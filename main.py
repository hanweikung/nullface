import argparse
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from PIL import Image
from torch import autocast, inference_mode
from tqdm import tqdm

from ddm_inversion.inversion_utils import (
    inversion_forward_process,
    inversion_reverse_process,
)
from ddm_inversion.utils import dataset_from_yaml, image_grid
from prompt_to_prompt.ptp_classes import load_512
from utils.face_embedding import FaceEmbeddingExtractor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd_model_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="Path to the Stable Diffusion 1.5 model",
    )
    parser.add_argument(
        "--insightface_model_path",
        type=str,
        default="~/.insightface",
        help="Path to the InsightFace model",
    )
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--dataset_yaml", default="test_my_dataset.yaml")
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--skip", type=int, default=70)
    parser.add_argument(
        "--ip_adapter_scale",
        type=float,
        default="1.0",
        help=(
            "Controls the amount of text or image conditioning to apply to the model."
            "A value of 1.0 means the model is only conditioned on the image prompt."
        ),
    )
    parser.add_argument(
        "--id_emb_scale",
        type=float,
        default=-1.0,
        help="Scaling factor for the identity embedding, with a default value of -1.0 for anonymization purposes.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="log.txt",
        help="The output text file records the images in which faces could not be detected.",
    )
    parser.add_argument(
        "--det_thresh",
        type=float,
        default=0.1,
        help="Set your desired threshold for face detection.",
    )
    parser.add_argument(
        "--det_size",
        type=int,
        default=640,
        help="The size for face detection model input",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="A seed for reproducible inference."
    )
    parser.add_argument(
        "--mask_delay_steps",
        type=int,
        default=0,
        help="The number of diffusion steps to wait before applying the mask.",
    )

    args = parser.parse_args()
    full_data = dataset_from_yaml(args.dataset_yaml)
    sd_model_path = args.sd_model_path
    device = f"cuda:{args.device_num}"
    guidance_scale = args.guidance_scale
    id_emb_scale = args.id_emb_scale
    eta = args.eta  # = 1
    skip_zs = [args.skip]

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
    ldm_stable.set_ip_adapter_scale(args.ip_adapter_scale)
    dtype = ldm_stable.dtype

    # Initialize FaceEmbeddingExtractor instance
    extractor = FaceEmbeddingExtractor(
        ctx_id=args.device_num,
        det_thresh=args.det_thresh,
        det_size=(args.det_size, args.det_size),
        model_path=args.insightface_model_path,
    )  # Use GPU (ctx_id>=0), or CPU with ctx_id=-1

    # Open the output file in write mode
    with open(args.output_file, "w") as f:
        for i in tqdm(range(len(full_data))):
            current_image_data = full_data[i]
            image_path = current_image_data["image"]
            mask_image_path = current_image_data["mask_image"]

            # Extract embedding for the largest face
            try:
                id_embs_inv, id_embs = extractor.get_face_embeddings(
                    image_path=image_path,
                    seed=args.seed,
                    scale_factor=args.id_emb_scale,
                    dtype=dtype,
                    device=device,
                )
            except ValueError as e:
                # Write the filename to the text file
                f.write(f"{e}\n")
            else:
                ldm_stable.scheduler = DDIMScheduler.from_config(
                    sd_model_path, subfolder="scheduler"
                )

                ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)

                # load image
                offsets = (0, 0, 0, 0)
                x0 = load_512(image_path, *offsets, device).to(dtype=dtype)

                # Check if the mask path exists. If it does, load the mask image.
                # Otherwise, create a new black image with the same size as the image.
                if mask_image_path and Path(mask_image_path).is_file():
                    mask_image = load_image(mask_image_path)
                else:
                    print(f"Error: The file '{mask_image_path}' was not found.")
                    # width and height are the dimensions of the image
                    height, width = x0.shape[-2:]
                    # Create a new image with a white background
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
                    num_inference_steps=args.num_diffusion_steps,
                    ip_adapter_image_embeds=[id_embs_inv],
                )

                for skip in skip_zs:
                    generator = torch.manual_seed(args.seed)

                    # reverse process (via Zs and wT)
                    w0, _ = inversion_reverse_process(
                        ldm_stable,
                        xT=wts[args.num_diffusion_steps - skip],
                        etas=eta,
                        prompts=[""],
                        cfg_scales=[guidance_scale],
                        prog_bar=True,
                        zs=zs[: (args.num_diffusion_steps - skip)],
                        controller=None,
                        ip_adapter_image_embeds=[id_embs],
                        init_image=x0,
                        mask_image=mask_image,
                        generator=generator,
                        mask_delay_steps=args.mask_delay_steps,
                    )

                    # vae decode image
                    with autocast("cuda"), inference_mode():
                        x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
                    if x0_dec.dim() < 4:
                        x0_dec = x0_dec[None, :, :, :]
                    img = image_grid(x0_dec)

                    # Replace dots with underscores.
                    # Format cfg to have at least 4 characters in total, including one digit after the decimal point, and pad it with leading zeros if necessary.
                    filename_wo_ext = f"{Path(image_path).stem}-cfg-{guidance_scale:04.1f}-skip-{skip}-id-{id_emb_scale}".replace(
                        ".", "_"
                    )

                    img.save(filename_wo_ext + ".png")
