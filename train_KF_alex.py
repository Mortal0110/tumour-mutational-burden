from __future__ import print_function, division
import tensorflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
from utils.folder2lmdb import ImageFolderLMDB
import tqdm
import yaml
from utils.common import logger
from utils.ImageFolderPaths import ImageFolderWithPaths

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# KFold 
folder = input('KFlod(number0~4): ')

plt.ion()  # interactive mode
# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
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

with open(os.path.join(os.path.abspath('.'),'config/config.yml'), 'r', encoding='utf8') as fs:
    cfg = yaml.load(fs, Loader=yaml.FullLoader)
data_dir = cfg['tiles_dir']

if cfg['use_lmdb'] is True:
    image_datasets = {x: ImageFolderLMDB(os.path.join(data_dir, x + str(folder) + '.lmdb'),
                                     data_transforms[x])
                  for x in ['train']}
else:
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x + str(folder)),
                                          data_transforms[x])
                      for x in ['train']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # min=0,max=1
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated


# Get a batch of training data and show
inputs, classes, names = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

tb = SummaryWriter()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        phase = 'train'
        model.train()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs_, labels_, names_ in tqdm.tqdm(dataloaders[phase]):
            
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs_ = model(inputs_)
                _, preds = torch.max(outputs_, 1)
                loss = criterion(outputs_, labels_)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs_.size(0)
            running_corrects += torch.sum((preds == labels_.data).int())

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        scheduler.step()
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        tb.add_scalar("Train/Loss", epoch_loss, epoch)
        tb.add_scalar("Train/Accuracy", epoch_acc, epoch)
        tb.flush()

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}h {:.0f}m'.format(
        time_elapsed // 3600, (time_elapsed-time_elapsed // 3600) * 60))
    logger.info('Best train Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    model_ft = models.alexnet(pretrained=True)
    model_ft.classifier.add_module("add_linear",nn.Linear(1000,2))
#    num_ftrs = model_ft.fc.in_features
#    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion_ft = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion_ft, optimizer_ft, exp_lr_scheduler,
                           num_epochs=8)
    tb.close()

    torch.save(model_ft, 'tmp/data/model/model_alex_{}.pkl'.format(folder))
    logger.info('tmp/data/model/model_alex_{}.pkl'.format(folder))
