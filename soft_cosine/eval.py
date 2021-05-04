import numpy as np
import pandas as pd
from tqdm import tqdm
from statistics import mean
from sklearn.metrics import accuracy_score

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
train_df = pd.read_csv(f'train_target.csv')
print(df.head())


def f1_score(true_labels, prd_labels):
    n = len(np.intersect1d(true_labels, prd_labels))
    return 2 * n / (len(true_labels) + len(prd_labels))


#
#
N = 1000
scores = []
with tqdm(total=N, position=0, leave=True) as pbar:
    for index in range(N):
        row = train_df.iloc[index]
        matches = df.loc[df['posting_id'] == row['posting_id']]['matches'].iloc[0]
        target = row['target'].split(' ')
        matches = matches.split(' ')
        scores.append(f1_score(target, matches[:len(target)]))
        pbar.update(1)
print(mean(scores))