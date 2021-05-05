# Standard Python imports

import os

# Preliminaries
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils import *
from models import *
from datasets import *



#torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

DIM = (256, 256)

NUM_WORKERS = 1
TRAIN_BATCH_SIZE = 24
VALID_BATCH_SIZE = 16
EPOCHS = 30
SEED = 2020
LR = 3e-4

device = torch.device('cuda')

####################################### Scheduler and its params ############################################################
SCHEDULER = 'CosineAnnealingWarmRestarts' #'CosineAnnealingLR'
factor=0.2 # ReduceLROnPlateau
patience=4 # ReduceLROnPlateau
eps=1e-6 # ReduceLROnPlateau
T_max=10 # CosineAnnealingLR
T_0=4 # CosineAnnealingWarmRestarts
min_lr=1e-6



############################################## Model Params ###############################################################
model_params = {
    'n_classes':11014,
    'model_name':'efficientnet_b3', #efficientnet_b0-b7
    'use_fc':True,
    'fc_dim':512,
    'dropout':0.0,
    'loss_module':'arcface', # arcface, cosface, adacos
    's':30.0,
    'margin':0.50,
    'ls_eps':0.0,
    'theta_zero':0.785,
    'pretrained':True
}





def run():
    # Dataset Creation
    data_dir = 'input/shopee-product-matching/train_images'
    train_df = pd.read_csv('input/shopee-product-matching/train.csv')

    # setup ground truth as list of posting ids similar to submisison.csv
    # dictionary that maps sample # to posting_id's that match
    tmp = train_df.groupby('label_group').posting_id.agg('unique').to_dict()
    train_df['target'] = train_df.label_group.map(tmp)

    # add filepath as dataframe member for easy loading
    train_df['file_path'] = train_df.image.apply(lambda x: os.path.join(data_dir, x))

    # Prep labels to be from 0 to N-1 groups
    le = LabelEncoder()
    train_df.label_group = le.fit_transform(train_df.label_group)

    # Validation set is first 1000 samples
    train_df_val = train_df[0:1000]

    # training is the rest of the data
    train_df = train_df[1000:]

    # Create SHOPEEDataset
    train_dataset = ShopeeDataset(train_df, transforms=get_train_transforms())
    valid_dataset = ShopeeDataset(train_df_val, transforms=get_valid_transforms())

    # Dataloaders for torch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Device
    device = torch.device("cuda")

    torch.cuda.empty_cache()

    # Create model on GPU
    model = ShopeeNet(**model_params)
    model.to(device)

    # Defining criterion
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # Defining Optimizer with weight decay to params other than bias and layer norms
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.Adam(optimizer_parameters, lr=LR)

    # Defining LR SCheduler
    scheduler = fetch_scheduler(optimizer)

    # THE ENGINE LOOP
    best_loss = 10000
    for epoch in range(EPOCHS):

        # train on training set
        train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch)

        # evaluate on validation set
        valid_loss = eval_fn(valid_loader, model, criterion, device)

        if valid_loss.avg < best_loss:

            # save off model if we beat performance
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), f'model_{model_params["model_name"]}_IMG_SIZE_{DIM[0]}_{model.loss_module}.bin')
            print('best model found for epoch {}'.format(epoch))


if __name__ == '__main__':
    run()