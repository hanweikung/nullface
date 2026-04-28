<div align="center">

<h1>NullFace: Training-Free Localized Face Anonymization</h1>

[Han-Wei Kung](https://hanweikung.github.io)<sup>1</sup> · [Tuomas Varanka](https://scholar.google.com/citations?user=5QWyHT4AAAAJ&hl=en)<sup>2</sup> · [Terence Sim](https://scholar.google.com/citations?user=AdEsZwsAAAAJ&hl=en)<sup>3</sup> · [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup> University of Trento, Italy | <sup>2</sup> University of Oulu, Finland | <sup>3</sup> National University of Singapore, Singapore

[![arXiv](https://img.shields.io/badge/arXiv-2503.08478-b31b1b?logo=arxiv&logoColor=white)](http://arxiv.org/abs/2503.08478) [![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=firefox&logoColor=white)](https://hanweikung.github.io/nullface/) [![FG 2026](https://img.shields.io/badge/FG%202026-Oral-gold)](https://fg2026.ieee-biometrics.org/)

Accepted for **oral presentation** at [IEEE FG 2026](https://fg2026.ieee-biometrics.org/).

<img src="assets/teaser.svg" width="100%">

</div>

Our method obscures identity while preserving attributes such as gaze, expressions, and head pose (in contrast to [Stable Diffusion Inpainting](https://github.com/CompVis/latent-diffusion)) and enables selective anonymization of specific facial regions.

## Environment setup

To install all required dependencies, create a new conda environment using the provided `environment.yml` file:

```sh
conda env create -f environment.yml
```

Then activate the environment:

```sh
conda activate nullface
```

## Usage

We include a sample image from the [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) dataset in the `my_dataset` folder to demonstrate example usage. The hyperparameters specified below are the ones used in our experiments for comparison with baseline methods.

```python
from anonymize_face import anonymize_face

output_img = anonymize_face(
    image_path="my_dataset/images/00080.png",
    mask_image_path="my_dataset/masks/00080/eyes_and_mouth.png",
    sd_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
    insightface_model_path="~/.insightface",
    device_num=0,
    guidance_scale=10.0,
    num_diffusion_steps=100,
    eta=1.0,
    skip=70,
    ip_adapter_scale=1.0,
    id_emb_scale=1.0,
    output_log_file="log.txt",
    det_thresh=0.1,
    det_size=640,
    seed=0,
    mask_delay_steps=10,
)

if output_img:
    output_path = "anonymized.png"
    output_img.save(output_path)
    print(f"Anonymized image saved to: {output_path}")
else:
    print(
        "Face could not be detected. Please check the output log file for more details."
    )
```

## Test set

For the quantitative comparisons against baseline methods in our paper, we selected:
- 4,852 test subjects from the [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) dataset
- 4,722 test subjects from the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset

For each subject, we created corresponding segmentation masks to selectively control the visibility of the eye and mouth areas if desired.

### Download

The list of selected test subjects and their corresponding segmentation masks are available for download at the [Hugging Face Hub](https://huggingface.co/datasets/hkung/nullface-test-set).

## Acknowledgements

This project is built upon [Diffusers](https://github.com/huggingface/diffusers) and [DDPM inversion](https://github.com/inbarhub/DDPM_inversion).
