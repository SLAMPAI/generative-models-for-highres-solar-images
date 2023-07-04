"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import json
import os
import torch
import numpy as np
import torch as th
from guided_diffusion import logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.train_util import get_step
import torchvision
from torchvision.transforms import functional as TF
from metrics import metrics
import horovod.torch as hvd
import time
import joblib
from guided_diffusion.image_datasets import ImageDataset
from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

class Normalize:

    def __init__(self, dl):
        self.dl = dl

    def __iter__(self):
        for x, y in self.dl:
            x = (x+1) / 2
            if x.shape[1] == 1:
                x = x.repeat(1,3,1,1)
            yield x

@torch.no_grad()
def main():
    hvd.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(hvd.local_rank())
    
    args = create_argparser().parse_args()
    print(args.use_ddim)
    logger.configure(dir=args.log_folder if args.log_folder else "results")
    real = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size_for_metric,
        image_size=args.image_size,
        class_cond=args.class_cond,
        nb_input_channels=args.nb_input_channels,
        deterministic=True,
        random_flip=False,
        random_crop=False,
    )
    real = Normalize(real)
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    step = get_step(args.model_path) 
    if hvd.rank() == 0:
        print("Step", step)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    class Sampler:

        def __iter__(self):
            self.samples = []
            i = 0
            nb = 0
            while nb < args.num_samples:
                logger.log("sampling...")
                t0 = time.time()
                model_kwargs = {}
                if args.class_cond:
                    classes = th.randint(
                        low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
                    )
                    model_kwargs["y"] = classes
                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                kw = {}
                if args.use_ddim and args.sampling_method:
                    kw['method'] = args.sampling_method
                sample = sample_fn(
                    model,
                    (args.batch_size, args.nb_input_channels, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    **kw
                )
                # bring to 0..1
                sample = (sample+1) / 2

                if args.force_grayscale:
                    sample = rgb_to_grayscale(sample)

                # gather generated images from all GPU workers
                gathered_samples = hvd.allgather(sample).cpu()
                for i in range(0, len(gathered_samples), args.batch_size_for_metric):
                    sample = gathered_samples[i:i+args.batch_size_for_metric] 
                    sample = sample.to(device)
                    yield sample
                if hvd.rank() == 0 and args.save_samples:
                    print(f"time per step {time.time() - t0} for generating {gathered_samples.shape}")
                    self.samples.append(gathered_samples)
                nb += len(gathered_samples)
            if hvd.rank() == 0 and args.save_samples:
                # in order to save them to disk if needed
                self.samples = torch.cat(self.samples)

    fake = Sampler()
    if args.compute_metric:
        compute = getattr(metrics, f"compute_{args.metric}")
        score = compute(
            real, fake, 
            device=device,
            real_data_stats_folder=args.real_data_stats_folder,
            pretrained_models_folder=args.pretrained_models_folder,
        )
        print(f"{args.metric}:", score)
    else:
        for x in fake:
            pass
    ddim = args.use_ddim
    respacing = args.timestep_respacing
    model_path = os.path.basename(args.model_path).replace(".pt", "")
    name = f"{model_path}_DDIM{ddim}{args.sampling_method}_T{respacing}"
    path = os.path.join(args.log_folder, f"{args.metric}_{name}.json")

    if hvd.rank() == 0 and args.compute_metric:
        with open(path, "w") as fd:
            if type(score) == dict:
                dic = score
            else:
                dic = {args.metric: score}
            s = json.dumps(dic)
            fd.write(s)

    if hvd.rank() == 0 and args.save_samples:
        torch.save(fake.samples, os.path.join(args.log_folder, f"samples_{name}.th"))
    

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    nb_input_channels=3,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=hvd.rank(),
        num_shards=hvd.size(),
        random_crop=random_crop,
        random_flip=random_flip,
        nb_input_channels=nb_input_channels,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False
        )
    return loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def rgb_to_grayscale(
    image: torch.Tensor, rgb_weights: torch.Tensor = torch.tensor([0.299, 0.587, 0.114])
) -> torch.Tensor:
    #From Kornia's code
    # https://github.com/kornia/kornia/blob/406f03aa3da709dddeee4a6992a1158d2054fc63/kornia/color/gray.py

    r"""Convert a RGB image to grayscale version of image.
    .. image:: _static/img/rgb_to_grayscale.png
    The image data is assumed to be in the range of (0, 1).
    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.
    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if not isinstance(rgb_weights, torch.Tensor):
        raise TypeError(f"rgb_weights is not a torch.Tensor. Got {type(rgb_weights)}")

    if rgb_weights.shape[-1] != 3:
        raise ValueError(f"rgb_weights must have a shape of (*, 3). Got {rgb_weights.shape}")

    r: torch.Tensor = image[..., 0:1, :, :]
    g: torch.Tensor = image[..., 1:2, :, :]
    b: torch.Tensor = image[..., 2:3, :, :]

    if not torch.is_floating_point(image) and (image.dtype != rgb_weights.dtype):
        raise TypeError(
            f"Input image and rgb_weights should be of same dtype. Got {image.dtype} and {rgb_weights.dtype}"
        )

    w_r, w_g, w_b = rgb_weights.to(image).unbind()
    return w_r * r + w_g * g + w_b * b

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        model_path="",
        log_folder="",
        batch_size_for_metric=8,
        data_dir="",
        sampling_method="",
        metric="fid",
        real_data_stats_folder="data/tfrecords_193A_40K_with_lev1.5_corrections/normalized_log_transform_stats",
        pretrained_models_folder=".",
        compute_metric=True,
        save_samples=False,
        force_grayscale=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
