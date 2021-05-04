import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('train.csv')
tmp = train_df.groupby('label_group').posting_id.agg('unique').to_dict()
for k in tmp.keys():
    value = ' '.join(tmp[k])
    tmp[k] = value

train_df['target'] = train_df.label_group.map(tmp)
# print(train_df.iloc[0])
train_df.to_csv(f'train_target.csv')
print(len(train_df['posting_id'].unique()))
df = pd.read_csv('submission_index5000.csv')
print(df.shape)
names = ['submission_index10000', 'submission_index11000', 'submission_index15000',
         'submission_index18000', 'submission_index25000', 'submission_index28000', 'submission_index30000',
         'submission_needed']
tmp = [df]
for n in names:
    tmp.append(pd.read_csv(f'{n}.csv'))

df = pd.concat(tmp, ignore_index=True)
print(df.shape)
df = df.drop_duplicates(subset=['posting_id'])
print(df.shape)
print(df.loc[df['posting_id'] == 'train_1802986387']['matches'].iloc[0])
# common = train_df.merge(df, on=["posting_id"])
# needed = train_df[~train_df["posting_id"].isin(common["posting_id"])]

# print(needed.shape)
results_y = []
results_y_h = []

with tqdm(total=df.shape[0], position=0, leave=True) as pbar:
    for index, row in df.iterrows():
        # data = [str(row['posting_id'])]matches
        target = train_df.loc[train_df['posting_id'] == row['posting_id']]['target'].iloc[0]
        target = target.split(' ')
        matches = set(row['matches'].split(' '))
        y = 1
        for t in range(len(target)):
            if target[t] not in matches:
                y = 0
                break
        results_y_h.append(y)
        results_y.append(1)
        pbar.update(1)

