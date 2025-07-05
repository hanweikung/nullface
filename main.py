from anonymize_face import anonymize_face

output_img_path = anonymize_face(
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

if output_img_path:
    print(f"Anonymized image saved to: {output_img_path}")
else:
    print(
        "Face could not be detected. Please check the output log file for more details."
    )
