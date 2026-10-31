#
# File:   run.py
# Author: Artem Suprun
# Date:   10/29/2026
# Description: User-facing interface for running an autoencoder.
#

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from predict import AutoEncoderPredictor


def parse_args():

    parser = argparse.ArgumentParser(description="Run an autoencoder on an input image.")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/autoencoder.pth",
        help="Path to the trained model checkpoint."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the reconstructed image."
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not display the original and reconstructed images."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    # Validate paths
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize predictor
    predictor = AutoEncoderPredictor(checkpoint_path=str(checkpoint_path))

    # Load image
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        # Preprocess input for visualization
        original = predictor.preprocess(image).squeeze(0).cpu()

        # Run inference
        reconstruction = predictor.predict(image)

    # Convert tensors into displayable images
    original_image = transforms.ToPILImage()(original)
    reconstructed_image = transforms.ToPILImage()(reconstruction)

    # Save reconstruction if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        reconstructed_image.save(output_path)

        print(f"Reconstruction saved to: {output_path}")

    # Display results
    if not args.no_display:

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(original_image, cmap="gray")
        axes[0].set_title("Original")

        axes[1].imshow(reconstructed_image, cmap="gray")
        axes[1].set_title("Reconstruction")

        for axis in axes:
            axis.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
