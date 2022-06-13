from pathlib import Path
import lmdb
import pyarrow
import six
import os
import torchvision
from torchvision import datasets, models, transforms
from utils.folder2lmdb import ImageFolderLMDB
import torch
import tqdm
import yaml
from utils.ImageFolderPaths import ImageFolderWithPaths


data_transforms = {'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if __name__ == '__main__':
    with open(os.path.join(os.path.abspath('.'), 'config/config.yml'), 'r', encoding='utf8') as fs:
        cfg = yaml.load(fs, Loader=yaml.FullLoader)
    tiles_dir = cfg['tiles_dir']

    if cfg['use_lmdb'] is False:
        image_datasets_dir = {x: ImageFolderWithPaths(os.path.join(tiles_dir, x),
                                                      data_transforms[x]) for x in ['train']}

        dataloaders_dir = {x: torch.utils.data.DataLoader(image_datasets_dir[x], batch_size=64,
                                                          shuffle=True, num_workers=8) for x in ['train']}

        for i, j, k in tqdm.tqdm(dataloaders_dir['train']):
            pass
    else:
        if not os.path.exists(os.path.join(tiles_dir, 'train.lmdb')):
            from utils.folder2lmdb import folder2lmdb
            folder2lmdb(tiles_dir, name="train")
            folder2lmdb(tiles_dir, name='test')

        dummy = True
        if dummy is True:
            image_datasets_lmdb_pkl = {x: ImageFolderLMDB(os.path.join(tiles_dir, x + '.lmdb'), data_transforms[x])
                                       for x in ['train']}
            dataloaders_lmdb_pkl = {x: torch.utils.data.DataLoader(image_datasets_lmdb_pkl[x], batch_size=64,
                                                               shuffle=True, num_workers=4)
                                for x in ['train']}
            for i, j, k in tqdm.tqdm(dataloaders_lmdb_pkl['train']):
                pass
