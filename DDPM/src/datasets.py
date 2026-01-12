import torch
from torchvision import datasets, transforms
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def generate_data(N):
    scaler = StandardScaler()
    X, _ = make_moons(n_samples=N, noise=0.05, random_state=42)
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    return X


class MoonsDataset(torch.utils.data.Dataset):
    def __init__(self, N):
        self.X = generate_data(N)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    def samples(self):
        return self.X


class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=self.transform)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image

    def __len__(self):
        return len(self.dataset)
