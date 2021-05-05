import random
import os
import numpy as np
import torch
from tqdm import tqdm
import gc

# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.neighbors import NearestNeighbors

# Setup seed for experiments
def seed_everything(seed):
    random.seed(seed)  # python random seed

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)  # numpy seed
    torch.manual_seed(seed)  # torch seed
    torch.cuda.manual_seed(seed)  # torch seed

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f'Setting all seeds to be {seed} to reproduce...')


# def seed_torch(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_train_transforms(DIM):
    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0],DIM[1],always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            #albumentations.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
            #albumentations.ShiftScaleRotate(
              #  shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            #),
            albumentations.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms(DIM):

    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0],DIM[1],always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )





def row_wise_f1_score(labels, preds):
    scores = []
    for label, pred in zip(labels, preds):
        n = len(np.intersect1d(label, pred))
        score = 2 * n / (len(label)+len(pred))
        scores.append(score)
    return scores, np.mean(scores)


# check if truth matches prediction
def match(actual, predicted):
    if actual == predicted:
        return True
    else:
        return False

# takes list of strings and join into one big string
def target_string(list_):
    return ' '.join(list_)



def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    loss_score = AverageMeter()

    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, (images, targets) in tk0:

        batch_size = images[0].shape[0]

        # to device
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # forward pass
        output = model(images, targets)

        # compute loss
        loss = criterion(output, targets)

        # backward pass and optimize params
        loss.backward()
        optimizer.step()

        # rolling window of losses
        loss_score.update(loss.detach().item(), batch_size)

        # Set tqdm bar info
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

    # step the scheduler
    if scheduler is not None:
        scheduler.step()

    # final loss
    return loss_score


def eval_fn(data_loader, model, criterion, device):

    loss_score = AverageMeter()
    model.eval()
    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))

    with torch.no_grad():
        for batch_idx, (images, targets) in tk0:
            batch_size = images[0].size()[0]

            # to device
            images = images.to(device)
            targets = targets.to(device)

            # forward pass but in eval mode
            output = model(images, targets)

            # compute loss
            loss = criterion(output, targets)

            # update loss meter
            loss_score.update(loss.detach().item(), batch_size)

            # update tqdm bar
            tk0.set_postfix(Eval_Loss=loss_score.avg)

    return loss_score


def get_neighbors(df, embeddings, KNN=50, image=True, GET_CV=True):
    '''
    https://www.kaggle.com/ragnar123/unsupervised-baseline-arcface?scriptVersionId=57121538
    '''

    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    # Iterate through different thresholds to maximize cv, run this in interactive mode, then replace else clause with a solid threshold
    if GET_CV:
        if image:
            thresholds = list(np.arange(2, 4, 0.1))
        else:
            thresholds = list(np.arange(0.1, 1, 0.1))
        scores = []
        for threshold in thresholds:
            predictions = []
            for k in range(embeddings.shape[0]):
                idx = np.where(distances[k,] < threshold)[0]
                ids = indices[k, idx]
                posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
                predictions.append(posting_ids)
            df['pred_matches'] = predictions
            df['f1'] = row_wise_f1_score(df['matches'], df['pred_matches'])
            score = df['f1'].mean()
            print(f'Our f1 score for threshold {threshold} is {score}')
            scores.append(score)
        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'Our best score is {best_score} and has a threshold {best_threshold}')

        # Use threshold
        predictions = []
        for k in range(embeddings.shape[0]):
            # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
            if image:
                idx = np.where(distances[k,] < 2.7)[0]
            else:
                idx = np.where(distances[k,] < 0.60)[0]
            ids = indices[k, idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)

    # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
    else:
        predictions = []
        for k in tqdm(range(embeddings.shape[0])):
            if image:
                idx = np.where(distances[k,] < 2.7)[0]
            else:
                idx = np.where(distances[k,] < 0.60)[0]
            ids = indices[k, idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)

    del model, distances, indices
    gc.collect()
    return df, predictions
