import torch
import os
from tqdm import tqdm
from typing import List, Optional, Union
from diffusers.image_processor import PipelineImageInput
from diffusers.models import ImageProjection


def load_real_image(folder="data/", img_name=None, idx=0, img_size=512, device="cuda"):
    from ddm_inversion.utils import pil_to_tensor
    from PIL import Image
    from glob import glob

    if img_name is not None:
        path = os.path.join(folder, img_name)
    else:
        path = glob(folder + "*")[idx]

    img = Image.open(path).resize((img_size, img_size))

    img = pil_to_tensor(img).to(device)

    if img.shape[1] == 4:
        img = img[:, :3, :, :]
    return img


def mu_tilde(model, xt, x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps
    )
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev**0.5 * beta_t) / (1 - alpha_bar)) * x0 + (
        (alpha_t**0.5 * (1 - alpha_prod_t_prev)) / (1 - alpha_bar)
    ) * xt


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size,
    )

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(
        (
            num_inference_steps + 1,
            model.unet.in_channels,
            model.unet.sample_size,
            model.unet.sample_size,
        )
    ).to(dtype=x0.dtype, device=x0.device)
    xts[0] = x0
    for t in reversed(timesteps):
        idx = num_inference_steps - t_to_idx[int(t)]
        xts[idx] = (
            x0 * (alpha_bar[t] ** 0.5)
            + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        )

    return xts


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(
        model.scheduler.config.num_train_timesteps - 2,
        timestep
        + model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps,
    )

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(
        pred_original_sample, model_output, torch.LongTensor([next_timestep])
    )
    return next_sample


def get_variance(model, timestep):  # , prev_timestep):
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps
    )
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def inversion_forward_process(
    model,
    x0,
    etas=None,
    prog_bar=False,
    prompt="",
    cfg_scale=3.5,
    num_inference_steps=50,
    steps=None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
):
    if prompt:
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size,
    )
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]:
            etas = [etas] * model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(
            size=variance_noise_shape, dtype=model.dtype, device=model.device
        )
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    # op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)
    op = tqdm(timesteps) if prog_bar else timesteps

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = prepare_ip_adapter_image_embeds(
            model=model,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            device=model.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=prompt is not None,
        )
        added_uncond_kwargs = {"image_embeds": [image_embeds[0][:1, :]]}
        if prompt:
            added_cond_kwargs = {"image_embeds": [image_embeds[0][1:, :]]}
    else:
        added_uncond_kwargs = None
        if prompt:
            added_cond_kwargs = None

    for t in op:
        # idx = t_to_idx[int(t)]
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx + 1][None]
            # xt = xts_cycle[idx+1][None]

        with torch.no_grad():
            out = model.unet.forward(
                xt,
                timestep=t,
                encoder_hidden_states=uncond_embedding,
                added_cond_kwargs=added_uncond_kwargs,
            )
            if prompt:
                cond_out = model.unet.forward(
                    xt,
                    timestep=t,
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                )

        if prompt:
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample
        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else:
            # xtm1 =  xts[idx+1][None]
            xtm1 = xts[idx][None]
            # pred of x0
            pred_original_sample = (
                xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred
            ) / alpha_bar[t] ** 0.5

            # direction to xt
            prev_timestep = (
                t
                - model.scheduler.config.num_train_timesteps
                // model.scheduler.num_inference_steps
            )
            alpha_prod_t_prev = (
                model.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else model.scheduler.final_alpha_cumprod
            )

            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (
                0.5
            ) * noise_pred

            mu_xt = (
                alpha_prod_t_prev ** (0.5) * pred_original_sample
                + pred_sample_direction
            )

            z = (xtm1 - mu_xt) / (etas[idx] * variance**0.5)
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance**0.5) * z
            xts[idx] = xtm1

    if not zs is None:
        zs[0] = torch.zeros_like(zs[0])

    return xt, zs, xts


def reverse_step(model, model_output, timestep, sample, eta=0, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps
    )
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep)  # , prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (
        0.5
    ) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(
                model_output.shape, dtype=model.dtype, device=model.device
            )
        sigma_z = eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


def inversion_reverse_process(
    model,
    xT,
    etas=0,
    prompts="",
    cfg_scales=None,
    prog_bar=False,
    zs=None,
    controller=None,
    asyrp=False,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    init_image: Optional[torch.Tensor] = None,
    mask_image: PipelineImageInput = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    mask_delay_steps=0,
):
    # Define call parameters
    height, width = init_image.shape[-2:]
    do_classifier_free_guidance = prompts != None
    device = model._execution_device
    num_images_per_prompt = 1

    batch_size = len(prompts)

    cfg_scales_tensor = (
        torch.Tensor(cfg_scales)
        .view(-1, 1, 1, 1)
        .to(dtype=model.dtype, device=model.device)
    )

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None:
        etas = 0
    if type(etas) in [int, float]:
        etas = [etas] * model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0] :]) if prog_bar else timesteps[-zs.shape[0] :]

    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0] :])}

    # Prepare mask latent variables
    mask_condition = model.mask_processor.preprocess(
        mask_image, height=height, width=width, resize_mode="default", crops_coords=None
    ).to(device=device)

    masked_image = init_image * (mask_condition < 0.5)

    mask, masked_image_latents = model.prepare_mask_latents(
        mask_condition,
        masked_image,
        batch_size * num_images_per_prompt,
        height,
        width,
        model.dtype,
        device,
        generator,
        do_classifier_free_guidance,
    )
    if do_classifier_free_guidance:
        init_mask, _ = mask.chunk(2)
    else:
        init_mask = mask

    # Add image embeds for IP-Adapter
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = prepare_ip_adapter_image_embeds(
            model=model,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            device=model.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=prompts != None,
        )
        added_uncond_kwargs = {"image_embeds": [image_embeds[0][:1, :]]}
        if prompts:
            added_cond_kwargs = {"image_embeds": [image_embeds[0][1:, :]]}
    else:
        added_uncond_kwargs = None
        if prompts:
            added_cond_kwargs = None

    for t in op:
        idx = (
            model.scheduler.num_inference_steps
            - t_to_idx[int(t)]
            - (model.scheduler.num_inference_steps - zs.shape[0] + 1)
        )
        ## Unconditional embedding
        with torch.no_grad():
            uncond_out = model.unet.forward(
                xt,
                timestep=t,
                encoder_hidden_states=uncond_embedding,
                added_cond_kwargs=added_uncond_kwargs,
            )

        ## Conditional embedding
        if prompts:
            with torch.no_grad():
                cond_out = model.unet.forward(
                    xt,
                    timestep=t,
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                )

        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (
                cond_out.sample - uncond_out.sample
            )
            if t_to_idx[int(t)] >= mask_delay_steps:
                noise_pred = (
                    1 - init_mask
                ) * uncond_out.sample + init_mask * noise_pred
        else:
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1
        xt = reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z)
        if controller is not None:
            xt = controller.step_callback(xt)
    return xt, zs


def prepare_ip_adapter_image_embeds(
    model,
    ip_adapter_image,
    ip_adapter_image_embeds,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
):
    image_embeds = []
    if do_classifier_free_guidance:
        negative_image_embeds = []
    if ip_adapter_image_embeds is None:
        if not isinstance(ip_adapter_image, list):
            ip_adapter_image = [ip_adapter_image]

        if len(ip_adapter_image) != len(
            model.unet.encoder_hid_proj.image_projection_layers
        ):
            raise ValueError(
                f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(model.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
            )

        for single_ip_adapter_image, image_proj_layer in zip(
            ip_adapter_image, model.unet.encoder_hid_proj.image_projection_layers
        ):
            output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
            single_image_embeds, single_negative_image_embeds = model.encode_image(
                single_ip_adapter_image, device, 1, output_hidden_state
            )

            image_embeds.append(single_image_embeds[None, :])
            if do_classifier_free_guidance:
                negative_image_embeds.append(single_negative_image_embeds[None, :])
    else:
        for single_image_embeds in ip_adapter_image_embeds:
            if do_classifier_free_guidance:
                (
                    single_negative_image_embeds,
                    single_image_embeds,
                ) = single_image_embeds.chunk(2)
                negative_image_embeds.append(single_negative_image_embeds)
            image_embeds.append(single_image_embeds)

    ip_adapter_image_embeds = []
    for i, single_image_embeds in enumerate(image_embeds):
        single_image_embeds = torch.cat(
            [single_image_embeds] * num_images_per_prompt, dim=0
        )
        if do_classifier_free_guidance:
            single_negative_image_embeds = torch.cat(
                [negative_image_embeds[i]] * num_images_per_prompt, dim=0
            )
            single_image_embeds = torch.cat(
                [single_negative_image_embeds, single_image_embeds], dim=0
            )

        single_image_embeds = single_image_embeds.to(device=device)
        ip_adapter_image_embeds.append(single_image_embeds)

    return ip_adapter_image_embeds
