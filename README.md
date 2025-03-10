# NullFace: Training-Free Localized Face Anonymization

![](assets/teaser.svg)

Our method obscures identity while preserving attributes such as gaze, expressions, and head pose (in contrast to [Stable Diffusion Inpainting](https://github.com/CompVis/latent-diffusion)) and enables selective anonymization of specific facial regions.

## Test Set

We provide access to the test set used in our quantitative comparisons against baseline methods. This includes:
- [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) Test Set: 4,852 test subjects
- [FFHQ](https://github.com/NVlabs/ffhq-dataset) Test Set: 4,722 test subjects
- Segmentation Masks: Corresponding masks used in our experiments

### Download

You can download our test set and segmentation masks from the [Hugging Face Hub](https://huggingface.co/datasets/hkung/nullface-test-set).
