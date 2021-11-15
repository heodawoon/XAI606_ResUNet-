from __future__ import print_function
import argparse
import numpy as np
import torch
import os


## Module
from utils import *
from train import train
from model import ResUNetPPlus
from dataLoader import get_datapath, DataSegmentationLoader

parser = argparse.ArgumentParser(description='Pytorch Brain Tumor Segmentation ResUNet++')

parser.add_argument('--epochs', default=50, type=int,
                    help='perturbation magnitude')
parser.add_argument('--nfold', default=5, type=int,
                    help='perturbation magnitude')
parser.add_argument('--savepath', default="./daheo/result", type=str,
                    help='the path for save results')
parser.set_defaults(argument=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


def main():
    # Import Data
    global args
    args = parser.parse_args()

    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)

    # Use GPU
    import torch.cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f'CUDA is available. Your device is {device}.')
    else:
        print(f'CUDA is not available. Your device is {device}. It can take long time training in CPU.')

        # Fix Seed
    random_state = 42
    seed_everything(random_state)

    # Dataload
    image, mask = get_datapath('./data/', random_state)

    dataloader = DataSegmentationLoader(image, mask)

    model = nn.DataParallel(ResUNetPPlus()).to(device)
    loss = DiceLoss()
    train(dataloader, model, loss, device, args.epochs, args.savepath, args.nfold)


if __name__ == '__main__':
    main()
