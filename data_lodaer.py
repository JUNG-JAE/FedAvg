import torch
from torch.utils.data import Dataset
from skimage import io
from glob import glob
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from conf import settings

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * settings.CHANNEL_SIZE, (0.5,) * settings.CHANNEL_SIZE)])


class UserDataLoader(Dataset):
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = get_label(data_path_list)
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.path_list[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.label[idx])


def get_label(data_path_list):
    return [path.split('/')[-2] for path in data_path_list]


def worker_dataloader(worker_id):
    TRAINING_DATA_SET_PATH = glob(f"./data/{settings.DATA_TYPE}/{worker_id}/train/*/*[.png, .jpg]")
    TESTING_DATA_SET_PATH = glob(f"./data/{settings.DATA_TYPE}/{worker_id}/test/*/*[.png, .jpg]")

    train_loader = torch.utils.data.DataLoader(
        UserDataLoader(TRAINING_DATA_SET_PATH, settings.LABELS, transform=transform),
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        UserDataLoader(TESTING_DATA_SET_PATH, settings.LABELS, transform=transform),
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    return train_loader, test_loader


def source_dataloader():
    if settings.DATA_TYPE == "MNIST":
        train_set = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

    elif settings.DATA_TYPE == "FMNIST":
        train_set = datasets.FashionMNIST(root='./data/f_mnist', train=True, download=False, transform=transform)
        test_set = datasets.FashionMNIST(root='./data/f_mnist', train=False, download=False, transform=transform)

    elif settings.DATA_TYPE == "CIFAR10":
        train_set = datasets.CIFAR10(root='./data/cifar10', train=True, download=False, transform=transform)
        test_set = datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=transform)
    else:
        print("Input dataset is not supported")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=settings.BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=settings.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader
