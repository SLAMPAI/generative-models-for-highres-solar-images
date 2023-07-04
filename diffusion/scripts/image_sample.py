"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
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
import time
import horovod.torch as hvd

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    hvd.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(hvd.local_rank())
    args = create_argparser().parse_args()
    seed_everything(args.seed + hvd.rank())
    logger.configure(dir=args.log_folder if args.log_folder else "results")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    step = get_step(args.model_path) 
    if hvd.rank() == 0:
        print("Step", step)
        print("DDIM", args.use_ddim)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log("sampling...")
    all_images = []
    all_labels = []
    i = 0
    nb = 0
    while nb < args.num_samples:
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
        if args.use_ddim and args.method:
            kw['method'] = args.method
        sample = sample_fn(
            model,
            (args.batch_size, args.nb_input_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            **kw,
        )
        # sample = sample[:, 0:1].repeat(1,3,1,1)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        gathered_samples = hvd.allgather(sample)
        if hvd.rank() == 0:
            print("Generated", gathered_samples.shape, time.time() - t0)
            grid = torchvision.utils.make_grid(gathered_samples.cpu(), nrow=8)
            folder = logger.get_dir()
            TF.to_pil_image(grid).save(os.path.join(folder, f'{args.prefix}gen_{i}_step{step:010d}.png'))
        nb += len(gathered_samples)
        all_images.append(gathered_samples)
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(hvd.size())
            ]
            gathered_labels = hvd.allgather(gathered_labels)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        i += 1
    all_images = torch.cat(all_images)
    folder = logger.get_dir()
    torch.save(all_images, os.path.join(folder, f"{args.prefix}gen_step{step:010d}.th"))
    hvd.join()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        model_path="",
        log_folder="",
        method="",
        seed=0,
        prefix=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
