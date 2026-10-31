#
# File:   dataset.py
# Author: Artem Suprun
# Date:   10/29/2026
# Description: Dataset loading and preprocessing file for the autoencoder.
#

from torch.utils.data import Dataset


class AutoEncoderDataset(Dataset):

    def __init__(self, dataset, corruption=None):
        self.dataset = dataset
        self.corruption = corruption

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]

        # Preserve the original image as the target
        target = image.clone()

        # Corrupt the input image
        if self.corruption:
            inputs = self.corruption(image)
        else:
            inputs = image.clone()

        return inputs, target
