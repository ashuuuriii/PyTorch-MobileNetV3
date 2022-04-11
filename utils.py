import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def calc_mean_sd(dataloader: DataLoader):
    ch_sum, ch_sum_sqrt, n_batches = 0, 0, 0
    for data, _ in dataloader:
        ch_sum += torch.mean(data, dim=[0, 2, 3])
        ch_sum_sqrt += torch.mean(torch.square(data), dim=[0, 2, 3])
        n_batches += 1

    mean = ch_sum / n_batches
    sd = (ch_sum_sqrt / n_batches - mean ** 2) ** 0.5
    return mean, sd


if __name__ == '__main__':
    torch.manual_seed(42)
    dataset = datasets.CIFAR100('data', train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=128)
    mean, sd = calc_mean_sd(dataloader)
    print(f"mean: {mean}, sd: {sd}")
