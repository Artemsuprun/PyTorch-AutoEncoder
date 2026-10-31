
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from predict import AutoEncoderPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test an autoencoder using a random MNIST image."
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
        help="Optional path to save the comparison."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for the MNIST dataset."
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Optional specific MNIST dataset index."
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not display the comparison."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load MNIST
    dataset = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # Select a random image or use a specified index
    index = (
        args.index
        if args.index is not None
        else random.randrange(len(dataset))
    )

    if not 0 <= index < len(dataset):
        raise ValueError("MNIST index is out of range.")

    original, label = dataset[index]

    # Apply the same corruption used during training
    corruption = transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))

    corrupted = corruption(original)

    # Initialize predictor
    predictor = AutoEncoderPredictor(checkpoint_path=str(checkpoint_path))

    # Convert grayscale tensor to PIL for predictor preprocessing
    corrupted_image = transforms.ToPILImage()(corrupted)

    # Run inference
    reconstruction = predictor.predict(corrupted_image)

    # Prepare images for visualization
    original_image = original.squeeze().cpu()
    corrupted_image = corrupted.squeeze().cpu()
    reconstruction = reconstruction.squeeze().cpu()

    # Display comparison
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    images = [
        (original_image, "Original"),
        (corrupted_image, "Corrupted"),
        (reconstruction, "Reconstruction")
    ]

    for axis, (image, title) in zip(axes, images):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(f"MNIST Autoencoder | Label: {label}")
    plt.tight_layout()

    # Save comparison if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_path, bbox_inches="tight")
        print(f"Comparison saved to: {output_path}")

    if not args.no_display:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
