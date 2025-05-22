# In dataloaders.py

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from os.path import join
import random
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

# --- Fix for Pickling Error: Define Callable Classes ---
class ReshapeMNIST:
    """Reshapes a tensor to (-1) for MNIST."""
    def __call__(self, x):
        # Assumes input x is the tensor right before reshaping
        return x.view(-1)

class IdentityTransform:
    """A transform that returns the input unchanged."""
    def __call__(self, x):
        return x
# --- End Fix ---

class ContrastiveDatasetFromImages(Dataset):
    def __init__(self, dataset, num_views=2, transform=None, contrastive=True, distinct_views=True):
        self.dataset = dataset
        self.num_views = num_views
        self.transform = transform
        self.distinct_views = distinct_views
        self.contrastive = contrastive

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Handle Subset case where the underlying dataset needs to be accessed
        if isinstance(self.dataset, Subset):
             # Access the original dataset and the specific index within it
             original_idx = self.dataset.indices[idx]
             img, label = self.dataset.dataset[original_idx]
        else:
             # Standard dataset access
             img, label = self.dataset[idx]

        # Ensure the image is in PIL format if it's a tensor
        if isinstance(img, torch.Tensor):
            # Check if it's already 3 channels (like CIFAR) or 1 channel (like MNIST)
            if img.ndim == 3 and img.shape[0] == 1: # Grayscale tensor
                img = to_pil_image(img.squeeze(0), mode='L') # Convert to L mode PIL
            elif img.ndim == 3: # Color tensor
                 img = to_pil_image(img)
            elif img.ndim == 2: # Single channel tensor without channel dim
                 img = to_pil_image(img, mode='L')
            # Add other potential tensor formats if needed

        # Apply the main transform to the PIL image
        transformed_img = self.transform(img)

        data = {"image": transformed_img, "label": label, "index": idx}

        # Apply transform again for contrastive views if needed
        if self.contrastive:
            for i in range(self.num_views - 1):
                 # Apply transform to the *original* PIL image again for distinct augmentations
                 data[f"image{i+1}"] = self.transform(img)

        return data


def get_dataloaders(
    batch_size=256,
    num_views=2,
    dataset_name='cifar10',
    num_workers=0, # Default to 0 for easier debugging, especially on Windows
    size=224,
    root='/datadrive/pytorch-data', # Adjust this path as needed
    with_augmentation=True,
    contrastive = True,
    unlabeled = True,
    shuffle_train=True,
    shuffle_test=True, # Will be set to False for the test loader below
    non_parametric = False,
    return_datasets=False,
    max_train_samples=None,
    pin_memory=False, # Set to False if running on CPU
    persistent_workers=False # Set True if num_workers > 0 and initialization is slow
    ):
    dataset_name = dataset_name.lower()
    # Define normalization parameters
    normalization_params = {
        "cifar10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
        "cifar100": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
        "tinyimagenet": {"mean": (0.4802, 0.4481, 0.3975), "std": (0.2302, 0.2265, 0.2262)},
        "imagenet": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        "stl10": {"mean":(0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        "mnist": {"mean": (0.1307,), "std": (0.3081,)},
        "oxfordpets": {"mean": (0.4467, 0.4398, 0.4066), "std": (0.2603, 0.2566, 0.2713)},
    }

    if dataset_name not in normalization_params:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Check available keys.")

    # Select normalization
    mean, std = normalization_params[dataset_name]["mean"], normalization_params[dataset_name]["std"]
    normalize = transforms.Normalize(mean=mean, std=std)

    # --- Use defined classes instead of lambda ---
    reshape_transform = ReshapeMNIST() if dataset_name == 'mnist' else IdentityTransform()

    # Define transformation pipelines as lists first
    train_transform_list = [
        transforms.RandomResizedCrop(size=size if dataset_name != 'mnist' else 28, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip() if dataset_name != 'mnist' else transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8) if dataset_name != 'mnist' else None,
        transforms.RandomGrayscale(p=0.2) if dataset_name != 'mnist' else None,
        transforms.ToTensor(),
        normalize,
        reshape_transform, # Use the class instance
    ] if with_augmentation else [
        transforms.Resize(size=(size, size) if dataset_name != 'mnist' else (28, 28)),
        transforms.ToTensor(),
        normalize,
        reshape_transform, # Use the class instance
    ]

    test_transform_list = [
        transforms.Resize(size=(size, size) if dataset_name != 'mnist' else (28, 28)),
        transforms.ToTensor(),
        normalize,
        reshape_transform, # Use the class instance
    ]

    # Remove None transforms and compose
    train_transform = transforms.Compose([t for t in train_transform_list if t is not None])
    test_transform = transforms.Compose([t for t in test_transform_list if t is not None])
    # --- End transform modification ---


    # Load datasets
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True)
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(root=root, train=True, download=True)
        test_dataset = datasets.CIFAR100(root=root, train=False, download=True)
    elif dataset_name == "tinyimagenet":
        data_dir = join(root, "tiny-imagenet-200")
        train_dataset = datasets.ImageFolder(root=join(data_dir, "train"))
        test_dataset = datasets.ImageFolder(root=join(data_dir, "val"))
    elif dataset_name == "stl10":
        train_dataset = datasets.STL10(root=root, split='train+unlabeled' if unlabeled else 'train', download=True)
        test_dataset = datasets.STL10(root=root, split='test', download=True)
    elif dataset_name == "mnist":
        # MNIST needs explicit transform during download IF ContrastiveDatasetFromImages doesn't handle PIL conversion
        # However, ContrastiveDatasetFromImages now handles PIL conversion, so no transform needed here.
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=None)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=None)
    elif dataset_name == "oxfordpets":
        train_dataset = datasets.OxfordIIITPet(root=root, split='trainval', target_types='category', download=True, transform=None)
        test_dataset = datasets.OxfordIIITPet(root=root, split='test', target_types='category', download=True, transform=None)
    elif dataset_name == "imagenet":
        data_dir = join(root, "imagenet2/ILSVRC/Data/CLS-LOC") # Check this path
        train_dir = join(data_dir, "train")
        test_dir = join(data_dir, "val") # Or "val2" depending on your setup
        train_dataset = datasets.ImageFolder(root=train_dir)
        test_dataset = datasets.ImageFolder(root=test_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")


    if max_train_samples is not None and max_train_samples < len(train_dataset):
        indices = random.sample(range(len(train_dataset)), max_train_samples)
        train_dataset = Subset(train_dataset, indices)

    # Wrap datasets
    train_dataset_wrapped = ContrastiveDatasetFromImages(train_dataset, num_views=num_views, transform=train_transform, contrastive=contrastive)
    test_dataset_wrapped = ContrastiveDatasetFromImages(test_dataset, num_views=1, transform=test_transform, contrastive=False) # num_views=1 for test


    # Effective batch size calculation - adjust if needed
    eff_batch_size = batch_size // num_views if contrastive else batch_size

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset_wrapped,
        batch_size=eff_batch_size,
        shuffle=shuffle_train,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory, # Use arg
        persistent_workers=persistent_workers if num_workers > 0 else False # Only if workers > 0
    )

    test_loader = DataLoader(
        test_dataset_wrapped,
        batch_size=eff_batch_size, # Use same effective batch size or full batch_size
        shuffle=False, # <-- Set shuffle=False for testing/validation
        drop_last=True, # Usually False for testing/validation
        num_workers=num_workers,
        pin_memory=pin_memory, # Use arg
        persistent_workers=persistent_workers if num_workers > 0 else False # Only if workers > 0
    )

    if return_datasets:
        return train_loader, test_loader, train_dataset_wrapped, test_dataset_wrapped
    else:
        return train_loader, test_loader
