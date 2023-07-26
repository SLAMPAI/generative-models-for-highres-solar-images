# A Comparative Study on Generative Models for High Resolution Extreme Ultraviolet Solar Images
*by Mehdi Cherti, Alexander Czernik, Stefan Kesselheim, Frederic Effenberger, Jenia Jitsev * [\[arXiv\]](https://arxiv.org/abs/2304.07169)

- [Short version](https://ml4astro.github.io/icml2023/assets/24.pdf) of the paper accepted at [ICML 2023 Astro workshop](https://ml4astro.github.io/icml2023/#accepted-contributions)
  
[![Open In Colab][colab-badge]][colab-notebook]

[colab-notebook]: <https://colab.research.google.com/drive/1D-wB8OhHyb9Ag6bjGhVuDw5RTBK7ZKhb?usp=sharing>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

https://github.com/SLAMPAI/generative-models-for-highres-solar-images/assets/509507/0d015fd4-4903-4ed9-9ffa-de820106df9d

## Introduction

In this repository, we provide the code for *"A Comparative Study on Generative Models for High Resolution Extreme Ultraviolet Solar Images"* ([arXiv](https://arxiv.org/abs/2304.07169)).

## Installation

## Obtaining solar (SDO) data

The dataset is available here: <https://huggingface.co/datasets/slampai/solar-sdo>.
Download it and unzip using:

```bash
wget https://huggingface.co/datasets/slampai/solar-sdo/raw/main/image_folder_1024x1024_normalized_log_transform_193A_40K_with_lev1.5_corrections.zip
unzip image_folder_1024x1024_normalized_log_transform_193A_40K_with_lev1.5_corrections.zip
```

## Downloading models

### (Best) Diffusion model

```bash
wget https://huggingface.co/slampai/generative-models-for-highres-solar-images/resolve/main/diffusion/diffusion_1000t_lr0.0001_128ch_2bpr_horiz_flip/ema_0.9999_058000.pt --output-document=ema_0.9999_058000.pt
```

The full set of models is available at <https://huggingface.co/slampai/generative-models-for-highres-solar-images/tree/main/models/diffusion>.

### (Best) ProjectedGAN model

```bash
wget https://huggingface.co/slampai/generative-models-for-highres-solar-images/resolve/main/models/projgan/00017-stylegan2-proj_baseline/network-snapshot.pkl --output-document=projgan_best.pkl
```
The full set of models is available at <https://huggingface.co/slampai/generative-models-for-highres-solar-images/tree/main/models/projgan>.

## Sampling from models

For diffusion models, see [colab](https://colab.research.google.com/drive/1ETQ48vxhBFcTu4s-j6FAjCVe14rPg02h?usp=sharing).

For ProjectedGAN, see [colab](https://colab.research.google.com/drive/1D-wB8OhHyb9Ag6bjGhVuDw5RTBK7ZKhb?usp=sharing), also
latent space exploration included.

## Training experiments

TODO 

## Results

See [results.ipynb](results.ipynb).

## Citation

If you find this work helpful, please cite our paper:
```
@article{cherti2023comparative,
  title={A Comparative Study on Generative Models for High Resolution Solar Observation Imaging},
  author={Cherti, Mehdi and Czernik, Alexander and Kesselheim, Stefan and Effenberger, Frederic and Jitsev, Jenia},
  journal={arXiv preprint arXiv:2304.07169},
  year={2023}
}
```
