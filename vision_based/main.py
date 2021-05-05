# Standard Python imports

# Standard vanilla pip libraries
import pandas as pd


# Torch imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Fancy imports
import albumentations


# Sklearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda')

# Apply random seed
from utils import *
from models import *
from datasets import *


if __name__ == '__main__':
    seed_everything(42)

    # load training data
    data_dir = 'input/shopee-product-matching/train_images'
    df_train = pd.read_csv('input/shopee-product-matching/train.csv')

    # add filepath as dataframe member
    df_train['file_path'] = df_train.image.apply(lambda x: os.path.join(data_dir, x))

    # Create Cross-validation folds
    gkf = GroupKFold(n_splits=5)
    df_train['fold'] = -1
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(df_train, None, df_train.label_group)):
        df_train.loc[valid_idx, 'fold'] = fold



    # ----- CONFIGURATIONS ----- #
    image_size = 512
    batch_size = 10
    n_worker = 1
    init_lr = 3e-4
    n_epochs = 6 # from my experiments, use > 25 when margin = 0.5
    fold_id = 0
    holdout_id = 0
    valid_every = 5
    save_after = 10
    accumulation_step = 1
    margin = 0.5 # 0 for faster convergence, larger may be beneficial
    search_space = np.arange(40, 100, 10) # in my experiments, thresholds should be between 40 - 90 (/100) for cosine similarity
    use_amp = False # todo: figure how to work with pytorch native amp
    debug = True # set this to False to train in full
    kernel_type = 'baseline'
    model_dir = 'weights'


    # Data augmentation transformers
    transforms_train = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        albumentations.HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
        albumentations.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
        albumentations.CoarseDropout(p=0.5),
        albumentations.Normalize()
    ])

    transforms_valid = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])


    # Create the dataset from dataframe
    dataset = SHOPEEDataset(df_train, 'train', transform=transforms_train)


    # Create model
    model = SHOPEEDenseNet(image_size, df_train.label_group.nunique())

    print(f'Creating model: image_size:{image_size} labels:{df_train.label_group.nunique()}')

    model.to(device)


    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    df_train_this = df_train[df_train['fold'] != fold_id]
    df_valid_this = df_train[df_train['fold'] == fold_id]

    df_valid_this['count'] = df_valid_this.label_group.map(df_valid_this.label_group.value_counts().to_dict())

    dataset_train = SHOPEEDataset(df_train_this, 'train', transform = transforms_train)
    dataset_valid = SHOPEEDataset(df_valid_this, 'test', transform = transforms_valid)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers = n_worker)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers = n_worker)

    def train_func(train_loader):
        model.train()
        bar = tqdm(train_loader)
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        losses = []
        for batch_idx, (images, targets) in enumerate(bar):

            images, targets = images.to(device), targets.to(device).long()

            if debug and batch_idx == 100:
                print('Debug Mode. Only train on first 100 batches.')
                break

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(images, targets)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % accumulation_step == 0) or ((batch_idx + 1) == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(images, targets)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])

            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

        loss_train = np.mean(losses)
        return loss_train


    def valid_func(valid_loader):
        model.eval()
        bar = tqdm(valid_loader)

        TARGETS = []
        losses = []
        PREDS = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(bar):
                images, targets = images.to(device), targets.to(device).long()

                logits = model(images, targets)

                PREDS += [torch.argmax(logits, 1).detach().cpu()]
                TARGETS += [targets.detach().cpu()]

                loss = criterion(logits, targets)
                losses.append(loss.item())

                bar.set_description(f'loss: {loss.item():.5f}')

        PREDS = torch.cat(PREDS).cpu().numpy()
        TARGETS = torch.cat(TARGETS).cpu().numpy()
        accuracy = (PREDS == TARGETS).mean()

        loss_valid = np.mean(losses)
        return loss_valid, accuracy


    def generate_test_features(test_loader):
        model.eval()
        bar = tqdm(test_loader)

        FEAS = []

        with torch.no_grad():
            for batch_idx, (images) in enumerate(bar):
                images = images.to(device)

                features = model(images)

                FEAS += [features.detach().cpu()]

        FEAS = torch.cat(FEAS).cpu().numpy()

        return FEAS


    def row_wise_f1_score(labels, preds):
        scores = []
        for label, pred in zip(labels, preds):
            n = len(np.intersect1d(label, pred))
            score = 2 * n / (len(label)+len(pred))
            scores.append(score)
        return scores, np.mean(scores)


    def find_threshold(df, lower_count_thresh, upper_count_thresh, search_space):
        '''
        Compute the optimal threshold for the given count threshold.
        '''
        score_by_threshold = []
        best_score = 0
        best_threshold = -1
        for i in tqdm(search_space):
            sim_thresh = i / 100
            selection = ((FEAS @ FEAS.T) > sim_thresh).cpu().numpy()
            matches = []
            oof = []
            for row in selection:
                oof.append(df.iloc[row].posting_id.tolist())
                matches.append(' '.join(df.iloc[row].posting_id.tolist()))
            tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
            df['target'] = df.label_group.map(tmp)
            scores, score = row_wise_f1_score(df.target, oof)
            df['score'] = scores
            df['oof'] = oof

            selected_score = df.query(f'count > {lower_count_thresh} and count < {upper_count_thresh}').score.mean()
            score_by_threshold.append(selected_score)
            if selected_score > best_score:
                best_score = selected_score
                best_threshold = i

        plt.title(f'Threshold Finder for count in [{lower_count_thresh},{upper_count_thresh}].')
        plt.plot(score_by_threshold)
        plt.axis('off')
        plt.show()
        print(f'Best score is {best_score} and best threshold is {best_threshold / 100}')


    for epoch in range(n_epochs):
        scheduler.step()
        loss_train = train_func(train_loader)
        if epoch % valid_every == 0:
            print('Now generating features for the validation set to simulate the submission.')
            FEAS = generate_test_features(valid_loader)
            FEAS = torch.tensor(FEAS).cuda()
            print('Finding Best Threshold in the given search space.')
            find_threshold(df = df_valid_this,
                   lower_count_thresh = 0,
                   upper_count_thresh = 999,
                   search_space = search_space)
            if epoch >= save_after:
                torch.save(model.state_dict(), f'{model_dir}{kernel_type}_fold{fold_id}_densenet_{image_size}_epoch{epoch}.pth')

