from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from PIL import Image
from torch import autocast, inference_mode

from ddm_inversion.inversion_utils import (
    inversion_forward_process,
    inversion_reverse_process,
)
from ddm_inversion.utils import image_grid
from prompt_to_prompt.ptp_classes import load_512
from utils.face_embedding import FaceEmbeddingExtractor


def anonymize_face(
    image_path: str,
    mask_image_path: str,
    sd_model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    insightface_model_path: str = "~/.insightface",
    device_num: int = 0,
    guidance_scale: float = 10.0,
    num_diffusion_steps: int = 100,
    eta: float = 1.0,
    skip: int = 70,
    ip_adapter_scale: float = 1.0,
    id_emb_scale: float = 1.0,
    output_log_file: str = "log.txt",
    det_thresh: float = 0.1,
    det_size: int = 640,
    seed: int = 0,
    mask_delay_steps: int = 10,
):
    """
    Anonymizes a facial image using Stable Diffusion and face embedding extraction.

    Args:
        image_path (str): Path to the input image to be anonymized.
        mask_image_path (str): Path to the mask image. If not found, a white mask is used.
        sd_model_path (str): Path to the Stable Diffusion 1.5 model.
        insightface_model_path (str): Path to the InsightFace model.
        device_num (int): CUDA device number to use.
        guidance_scale (float): Guidance scale for diffusion.
        num_diffusion_steps (int): Number of diffusion steps.
        eta (float): DDIM eta parameter.
        skip (int): Number of steps to skip in the reverse process.
        ip_adapter_scale (float): Controls the amount of text or image conditioning.
        id_emb_scale (float): Scaling factor for the identity embedding.
        output_log_file (str): Output text file to record images where faces could not be detected.
        det_thresh (float): Threshold for face detection.
        det_size (int): Size for face detection model input.
        seed (int): Seed for reproducible inference.
        mask_delay_steps (int): Number of diffusion steps to wait before applying the mask.

    Returns:
        PIL.Image.Image or None: The anonymized image as a PIL Image, or None if a face is not detected.
    """
    device = f"cuda:{device_num}"

    # load/reload model:
    ldm_stable = StableDiffusionInpaintPipeline.from_pretrained(
        sd_model_path, torch_dtype=torch.float16
    ).to(device)
    ldm_stable.load_ip_adapter(
        "h94/IP-Adapter-FaceID",
        subfolder=None,
        weight_name="ip-adapter-faceid_sd15.bin",
        image_encoder_folder=None,
    )
    ldm_stable.set_ip_adapter_scale(ip_adapter_scale)
    dtype = ldm_stable.dtype

    # Initialize FaceEmbeddingExtractor instance
    extractor = FaceEmbeddingExtractor(
        ctx_id=device_num,
        det_thresh=det_thresh,
        det_size=(det_size, det_size),
        model_path=insightface_model_path,
    )  # Use GPU (ctx_id>=0), or CPU with ctx_id=-1

    # Open the output file in write mode
    with open(output_log_file, "w") as f:
        # Extract embedding for the largest face
        try:
            id_embs_inv, id_embs = extractor.get_face_embeddings(
                image_path=image_path,
                is_opposite=True,
                seed=seed,
                scale_factor=id_emb_scale,
                dtype=dtype,
                device=device,
            )
        except ValueError as e:
            # Write the filename to the text file
            f.write(f"{e}\n")
            return None
        else:
            ldm_stable.scheduler = DDIMScheduler.from_config(
                sd_model_path, subfolder="scheduler"
            )

            ldm_stable.scheduler.set_timesteps(num_diffusion_steps)

            # load image
            offsets = (0, 0, 0, 0)
            x0 = load_512(image_path, *offsets, device).to(dtype=dtype)

            # Check if the mask path exists. If it does, load the mask image.
            # Otherwise, create a new white image with the same size as the image.
            if mask_image_path and Path(mask_image_path).is_file():
                mask_image = load_image(mask_image_path)
            else:
                print(f"Error: The file '{mask_image_path}' was not found.")
                height, width = x0.shape[-2:]
                mask_image = Image.new("RGB", (width, height), "white")

            # vae encode image
            with autocast("cuda"), inference_mode():
                w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).to(
                    dtype=dtype
                )

            # find Zs and wts - forward process
            wt, zs, wts = inversion_forward_process(
                ldm_stable,
                w0,
                etas=eta,
                prompt="",
                cfg_scale=guidance_scale,
                prog_bar=True,
                num_inference_steps=num_diffusion_steps,
                ip_adapter_image_embeds=[id_embs_inv],
            )

            generator = torch.manual_seed(seed)

            # reverse process (via Zs and wT)
            w0, _ = inversion_reverse_process(
                ldm_stable,
                xT=wts[num_diffusion_steps - skip],
                etas=eta,
                prompts=[""],
                cfg_scales=[guidance_scale],
                prog_bar=True,
                zs=zs[: (num_diffusion_steps - skip)],
                controller=None,
                ip_adapter_image_embeds=[id_embs],
                init_image=x0,
                mask_image=mask_image,
                generator=generator,
                mask_delay_steps=mask_delay_steps,
            )

            # vae decode image
            with autocast("cuda"), inference_mode():
                x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
            if x0_dec.dim() < 4:
                x0_dec = x0_dec[None, :, :, :]
            img = image_grid(x0_dec)
            return img
