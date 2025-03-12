# NullFace: Training-Free Localized Face Anonymization

[arXiv](http://arxiv.org/abs/2503.08478)

![](assets/teaser.svg)

Our method obscures identity while preserving attributes such as gaze, expressions, and head pose (in contrast to [Stable Diffusion Inpainting](https://github.com/CompVis/latent-diffusion)) and enables selective anonymization of specific facial regions.

## Test set

For the quantitative comparisons against baseline methods in our paper, we selected:
- 4,852 test subjects from the [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) dataset
- 4,722 test subjects from the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset

For each subject, we created corresponding segmentation masks to selectively control the visibility of the eye and mouth areas if desired.

### Download

The list of selected test subjects and their corresponding segmentation masks are available for download at the [Hugging Face Hub](https://huggingface.co/datasets/hkung/nullface-test-set).

## Code coming soon 