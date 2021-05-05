import os
import pandas as pd

from datasets import *
from utils import *
from models import *

from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':

    DIM = (256, 256)
    NUM_WORKERS = 1
    TRAIN_BATCH_SIZE = 24
    VALID_BATCH_SIZE = 16

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
    train_dataset = ShopeeDataset(train_df, transforms=get_train_transforms(DIM))
    valid_dataset = ShopeeDataset(train_df_val, transforms=get_valid_transforms(DIM))

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

    # Validation set is first 1000 samples
    train_df_val = train_df[0:1000]

    # training is the rest of the data
    train_df = train_df[1000:]

    # Create SHOPEEDataset
    train_dataset = SHOPEEDataset(train_df, mode='train', transform=get_train_transforms(DIM))
    valid_dataset = SHOPEEDataset(train_df_val, mode='test', transform=get_valid_transforms(DIM))

    device = torch.device('cuda')

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

    model = ShopeeNet(**model_params)
    model.load_state_dict(torch.load('model_efficientnet_b3_IMG_SIZE_256_arcface.bin'))
    model.to(device)


    # Compute embeddings on validation set
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(valid_loader):
            img = img.to(device)
            label = label.to(device)
            embedding_ = F.normalize(model.extract_feat(img))
            embeddings.append(embedding_.detach().cpu())


    embeddings_mat = torch.cat(embeddings).cpu().numpy()

    sim_thresh = 0.7

    # dot product with each other, only grab if above similarity threshold
    # since model output is normalize this is equivalent to cosine similarity
    selection = ((embeddings_mat @ embeddings_mat.T) > sim_thresh)




    # take target (list of strings) and collapse to string separated by spaces
    train_df_val['target_string'] = train_df_val.apply(lambda x: target_string(x['target']),axis=1)
    train_df_val.head(5)


    matches = []

    # loop through each row (each sample)
    for row in selection:

        # grab the samples that meet the threshold criteria, grab posting id and convert to list of strings
        posting_ids_that_match = train_df_val.iloc[row].posting_id.tolist()

        # collapse to one long string and add to list
        matches.append(posting_ids_that_match)


    tmp = dict( zip(range(len(matches), 2*len(matches)), matches) )
    train_df_val['pred'] = train_df_val.index.map(tmp)


    # convert list of strings for target into one long string
    train_df_val['target_string'] = train_df_val.apply(lambda x: target_string(x['target']), axis=1)

    # do the same for prediction
    train_df_val['pred_string'] = train_df_val.apply(lambda x: target_string(x['pred']), axis=1)


    # boolean for each target/pred pair if the strings match exactly
    train_df_val['prediction_accuracy'] = train_df_val.apply(lambda x: match(x['target_string'], x['pred_string']),axis=1)
    prediction_accuracy_v1 = train_df_val.groupby(['prediction_accuracy'])['posting_id'].count()

    ans = prediction_accuracy_v1[True] / (prediction_accuracy_v1[True] + prediction_accuracy_v1[False])

    print(f'Accuracy: {ans*100.0}%')




    scores, score = row_wise_f1_score(train_df_val.target, train_df_val.pred)

    print(f'Accuracy (shopee metric) {score}')