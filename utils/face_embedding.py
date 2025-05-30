import cv2
import numpy as np
import torch
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from utils.sample_vector import random_vector_within_angle


class FaceEmbeddingExtractor:
    """
    A class to extract identity embeddings of the largest face from an image using InsightFace.
    """

    def __init__(
        self, ctx_id=0, det_thresh=0.5, det_size=(640, 640), model_path="~/.insightface"
    ):
        """
        Initializes the InsightFace app for facial analysis.

        Args:
        ctx_id (int): GPU context ID (use -1 for CPU).
        det_size (tuple): The size for face detection model input.
        """

        # Initialize InsightFace app
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)

    def get_face_embeddings(
        self,
        image_path,
        max_angle=0.0,
        is_opposite=False,
        seed=None,
        scale_factor=1.0,
        dtype=torch.float32,
        device="cuda",
    ):
        """
        Extract the identity embeddings of the largest face from an image file, scale it, and concatenate it with a black image representing a negative embedding.

        Args:
        image_path (str): The file path of the image.
        max_angle (float): The maximum allowed angle (in degrees) from the direction of the given face embedding.
        is_opposite (bool): If true, retrieve the face embedding in the opposite direction of the given face embedding for anonymization.
        seed (int, optional): Seed for the random number generator.
        scale_factor (float): A factor to scale the embedding.
        dtype (torch.dtype): The desired data type of the resulting embedding.
        device (str or torch.device): The device on which to place the resulting embedding.

        Returns:
        concat_emb_inv (torch.Tensor): The embedding of the largest detected face for performing inversion
        concat_emb (torch.Tensor): The embedding of the largest detected face
        """

        # Load image
        image = load_image(image_path)
        ref_images_embeds = []

        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.app.get(image)

        if len(faces) == 0:
            raise ValueError(f"No face detected in {image_path}.")
        else:
            # Find the largest face by bounding box area
            largest_face = max(
                faces,
                key=lambda face: (face.bbox[2] - face.bbox[0])
                * (face.bbox[3] - face.bbox[1]),
            )
            normed_embedding = largest_face.normed_embedding

        # Get the embedding of the largest face
        embedding = torch.from_numpy(normed_embedding)

        embedding = -embedding if is_opposite else embedding

        # Sample a random embedding within a specified angle from the direction of the given embedding
        sampled_embedding = random_vector_within_angle(
            embedding.numpy(), max_angle, seed
        )

        # Convert the NumPy array to a PyTorch tensor
        sampled_embedding = torch.from_numpy(sampled_embedding)

        # Scale the embedding by the provided scale factor
        scaled_embedding = sampled_embedding * scale_factor
        ref_images_embeds.append(scaled_embedding.unsqueeze(0))
        ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)
        neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)
        concat_emb_inv = torch.cat([neg_ref_images_embeds, neg_ref_images_embeds]).to(
            dtype=dtype, device=device
        )
        concat_emb = torch.cat([neg_ref_images_embeds, ref_images_embeds]).to(
            dtype=dtype, device=device
        )

        return concat_emb_inv, concat_emb
