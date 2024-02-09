import os
import logging

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset


class RoadDataset(Dataset):
    def __init__(self, path: str, split: str):
        self.path = path
        self.split = split

        self.images = []
        self.labels = []
        self._load_data()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _load_data(self):
        with open(os.path.join(self.path, self.split + ".txt")) as f:
            lines = f.readlines()

        for line in lines:
            name = line.strip()
            image_path = os.path.join(self.path, "Images", name)
            label_path = os.path.join(self.path, "Annotations", name)

            # Check if the files exists
            if not os.path.isfile(image_path):
                logging.warning(f"Image file {image_path} does not exist")
                continue
            if not os.path.isfile(label_path):
                logging.warning(f"Label file {label_path} does not exist")
                continue

            self.images.append(image_path)
            self.labels.append(label_path)

        assert len(self.images) == len(self.labels), "Number of images and labels do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # Change the shape of the label to (H, W)
        label = label.squeeze(0).long()

        return image, label

    def compute_mean_and_std(self):
        """
        Compute the mean and standard deviation of the dataset.
        """

        mean = 0
        std = 0
        nb_samples = 0
        for data, _ in self:
            data = data.view(data.size(0), -1)
            mean += data.mean(1)
            std += data.std(1)
            nb_samples += 1

        mean /= nb_samples
        std /= nb_samples
        return mean, std


def visualize_samples(dataset, num_samples=3):
    """
    Visualize some samples from the dataset along with their labels.
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))

    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]

        # Convert tensors to numpy arrays
        image = image.permute(1, 2, 0).numpy()
        label = label.permute(1, 2, 0).numpy()

        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(label[:, :, 0], cmap='gray')
        axes[i, 1].set_title('Label')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_path = "/home/ales/school/robolab/road-segmentation/data/LandscapeDataset"
    train_ds = RoadDataset(dataset_path, "train")
    test_ds = RoadDataset(dataset_path, "test")
    train_mean, train_std = train_ds.compute_mean_and_std()
    print(f"Train mean: {train_mean}, Train std: {train_std}")
    test_mean, test_std = test_ds.compute_mean_and_std()
    print(f"Test mean: {test_mean}, Test std: {test_std}")
    mean, std = (train_mean + test_mean) / 2, (train_std + test_std) / 2
    print(f"Mean: {mean}, Std: {std}")
