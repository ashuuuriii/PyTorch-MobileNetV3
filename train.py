import argparse
import yaml

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from model import MobileNetV3


def train():
    return


def train_epoch():
    return


def validate():
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help="Path to yaml config file")
    parser.add_argument('--force-cpu', action='store_true', help='Force PyTorch to use cpu when CUDA is available.')
    args = parser.parse_args()

    # Load config file
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        settings = cfg['settings']
        train_cfg = cfg['train_params']
        opt_cfg = cfg['optimizer_params']
        data_cfg = cfg['dataset_params']
        model_cfg = cfg['model_params']

    # Set device and random seed
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    torch.manual_seed(settings['seed'])
    torch.cuda.manual_seed(settings['seed'])

    # Create the dataloader
    augs = {'train': transforms.Compose([transforms.RandomCrop(data_cfg['img_size']),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(data_cfg['mean'], data_cfg['sd'])]),
            'val': transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(data_cfg['mean'], data_cfg['sd'])])}

    train_set = datasets.CIFAR100(root=data_cfg['path'],
                                  train=True,
                                  download=True,
                                  transform=augs['train'])
    val_set = datasets.CIFAR100(root=data_cfg['path'],
                                train=False,
                                download=True,
                                transform=augs['val'])

    train_loader = DataLoader(train_set, batch_size=data_cfg['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=data_cfg['batch_size'], shuffle=False)

    # Initialise the model
    model = MobileNetV3(model_size=model_cfg['model_size'],
                        n_classes=data_cfg['classes'],
                        head_type=model_cfg['classification_head'],
                        initialisation=model_cfg['initialisation_type'],
                        drop_rate=model_cfg['drop_out_probability'],
                        alpha=model_cfg['width_multiplier'])
    model = model.to(device)

    # Initialise the optimizer
    optimizer = optim.RMSprop(model.parameters(),
                              lr=opt_cfg['lr'],
                              momentum=opt_cfg['momentum'],
                              weight_decay=opt_cfg['weight_decay'])

    schdeduler = optim.lr_scheduler.StepLR(optimizer,
                                           opt_cfg['lr_decay_step'],
                                           gamma=opt_cfg['lr_decay'])

    loss_fn = nn.CrossEntropyLoss()  # standard loss function for classification
