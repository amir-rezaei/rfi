# src/core/learning/training/dataset.py

import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch



class ImageFolderDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images from a specified folder.

    This class scans a directory for image files, and provides a standard
    interface for a PyTorch DataLoader to access them. It handles opening,
    resizing, and transforming the images.

    Attributes:
        paths (list[str]): A list of file paths to the images in the dataset.
        transform (callable, optional): A function/transform to be applied to an image.
        img_size (int): The target size (width and height) to which images are resized.
        binarize (bool): If True, converts the image to a binary (0 or 1) tensor.
        threshold (float): The threshold used for binarization if `binarize` is True.
    """

    def __init__(
            self,
            root_dir: str,
            img_size: int = 64,
            transform: T.Compose = None,
            ext: str = 'png',
            binarize: bool = False,
            threshold: float = 0.5
    ):
        """
        Initializes the dataset.

        Args:
            root_dir: The path to the directory containing the images.
            img_size: The square dimension to resize images to.
            transform: Optional torchvision transforms to be applied. If None,
                       a default ToTensor transform is used.
            ext: The file extension of the images to be loaded (e.g., 'png', 'jpg').
            binarize: Whether to convert images to black and white.
            threshold: The brightness threshold for binarization.
        """
        self.paths = sorted(glob.glob(os.path.join(root_dir, f"*.{ext}")))
        if not self.paths:
            raise FileNotFoundError(f"No images with extension '.{ext}' found in directory '{root_dir}'")

        self.transform = transform
        self.img_size = img_size
        self.binarize = binarize
        self.threshold = threshold

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves an image by its index.

        This method is called by the DataLoader. It opens, resizes, transforms,
        and optionally binarizes the image at the given index.

        Args:
            idx: The index of the image to retrieve.

        Returns:
            A torch.Tensor representing the processed image.
        """
        # Open the image, convert to grayscale ('L' mode), and resize
        img = Image.open(self.paths[idx]).convert('L').resize((self.img_size, self.img_size))

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        else:
            # Apply default transformation if none is provided
            img = T.ToTensor()(img)

        # Binarize the image if requested
        if self.binarize:
            img = (img > self.threshold).float()

        return img



