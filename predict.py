#
# File:   predict.py
# Author: Artem Suprun
# Date:   10/29/2026
# Description: Inference pipeline for a generic autoencoder.
#

import torch
from PIL import Image
from torchvision import transforms

from model import AutoEncoder


class AutoEncoderPredictor:

    def __init__(self, checkpoint_path, device=None):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        # Load model configuration
        config = checkpoint["config"]

        self.model = AutoEncoder(
            encoder_dim=config["encoder_dim"],
            activation=config["activation"],
            output_activation=config["output_activation"]
        ).to(self.device)

        # Load trained weights
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # Expected image dimensions
        self.input_dim = config["encoder_dim"][0]

        # Assume square images
        self.image_size = int(self.input_dim ** 0.5)

        if self.image_size ** 2 != self.input_dim:
            raise ValueError("This predictor currently supports square images only.")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

    def preprocess(self, image):

        if isinstance(image, str):
            image = Image.open(image)

        image = image.convert("RGB")
        image = self.transform(image)

        return image.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image):

        self.model.eval()

        inputs = self.preprocess(image)

        reconstruction = self.model(inputs)

        reconstruction = reconstruction.squeeze(0)

        return reconstruction.cpu()

    def predict_image(self, image, save_path=None):

        reconstruction = self.predict(image)

        # Convert tensor into PIL image
        output = transforms.ToPILImage()(reconstruction)

        if save_path:
            output.save(save_path)

        return output


if __name__ == "__main__":

    predictor = AutoEncoderPredictor(checkpoint_path="checkpoints/autoencoder.pth")

    reconstructed = predictor.predict_image(image="data/test.png", save_path="reconstructed.png")

    reconstructed.show()
