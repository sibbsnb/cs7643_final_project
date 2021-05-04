import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import KeyedVectors
from multiprocessing import cpu_count

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import MatrixSimilarity, WmdSimilarity, SoftCosineSimilarity
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np

from tqdm import tqdm

np.random.seed(2018)
from gensim.models import Word2Vec
import nltk


stemmer = SnowballStemmer('english')
from sklearn.metrics.pairwise import cosine_similarity
from gensim.similarities import WmdSimilarity
from numpy import dot
from numpy.linalg import norm
from multiprocessing import Pool

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

print("finish loading...")
nltk.download('wordnet')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            if token == 'xxxx':
                continue
            result.append(lemmatize_stemming(token))

    return result


def word2vec_model(processed_docs):
    w2v_model = Word2Vec(min_count=1,
                         window=3,
                         vector_size=50,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20)

    w2v_model.build_vocab(processed_docs)
    w2v_model.train(processed_docs, total_examples=w2v_model.corpus_count, epochs=300, report_delay=1)

    return w2v_model


def query_similarity(processed_docs):
    model = Word2Vec(min_count=1,
                     window=3,
                     vector_size=50,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20)
    wmd = WmdSimilarity(processed_docs, model, num_best=50)
    return wmd


def find_similarity(sen1, sen2, model):
    p_sen1 = preprocess(sen1)
    p_sen2 = preprocess(sen2)

    sen_vec1 = np.zeros(50)
    sen_vec2 = np.zeros(50)
    for val in p_sen1:
        sen_vec1 = np.add(sen_vec1, model[val])

    for val in p_sen2:
        sen_vec2 = np.add(sen_vec2, model[val])

    return dot(sen_vec1, sen_vec2) / (norm(sen_vec1) * norm(sen_vec2))


def find_similarity_v2(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def preprocess_v2(sen1, model):
    # p_sen1 = preprocess(sen1)
    sen_vec1 = np.zeros(50)
    for val in sen1:
        sen_vec1 = np.add(sen_vec1, model[val])
    return sen_vec1


def find_similarity_idx_v2(posting_id, similarity_matrix):
    return np.argsort(similarity_matrix[posting_id])[::-1]


def compute_pairwise_similarities(vec):
    return np.dot(vec, vec.T)


def vect_df(df):
    data = []
    for index, row in df.iterrows():
        vec = preprocess_v2(row['clean'], emb_vec)
        data.append([row['posting_id'], vec])
    print(len(data))
    new_df = pd.DataFrame(data, columns=['posting_id', 'vec'])
    return new_df


def softcossim(tfidf, dictionary, similarity_matrix, documents):
    index = SoftCosineSimilarity(tfidf[[dictionary.doc2bow(document) for document in documents]], similarity_matrix)
    return index


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def find_sim(df):
    df['query'].apply(lambda x: instance[x].argsort()[-50:][::-1])
    return df


train_df = pd.read_csv('C:/Users/khakem/PycharmProjects/shopee/train.csv')
processed_docs = train_df['title'].map(preprocess)
processed_docs = list(processed_docs)
train_df['clean'] = processed_docs
corpus = train_df['clean'].tolist()
print(len(corpus))
dictionary = Dictionary(corpus)
tfidf = TfidfModel(dictionary=dictionary)
w2v_model = Word2Vec(corpus,
                     min_count=5,
                     window=3,
                     vector_size=50,
                     sample=6e-5,
                     alpha=0.03,
                     workers=cpu_count(),
                     min_alpha=0.0007,
                     negative=20)
w2v_model.save('word2vec_model')
emb_vec = w2v_model.wv
emb_vec.save("emb_vec.wordvectors")
similarity_index = WordEmbeddingSimilarityIndex(w2v_model.wv)
similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf, nonzero_limit=100)
print("start soft cos")
instance = softcossim(tfidf, dictionary, similarity_matrix, corpus)
train_df['query'] = train_df['clean'].apply(lambda x: tfidf[dictionary.doc2bow(x)])
results = []
needed = pd.read_csv('needed.csv')
new_df = needed
with tqdm(total=new_df.shape[0], position=0, leave=True) as pbar:
    for index, row in new_df.iterrows():
        #data = [str(row['posting_id'])]
        query = train_df.loc[train_df['posting_id'] == row['posting_id']]['query'].iloc[0]
        #query = tfidf[dictionary.doc2bow(query)]
        sims = instance[query]
        indices = sims.argsort()[-50:][::-1] # top 50
        data = train_df.iloc[indices,:]['posting_id'].tolist()
        results.append([row['posting_id'], " ".join(data)])
        pbar.update(1)
df = pd.DataFrame(results, columns = ['posting_id','matches'])
df.to_csv(f'submission_needed.csv', index=False)
'''
new_df = train_df[15000:]
with tqdm(total=new_df.shape[0], position=0, leave=True) as pbar:
    for index, row in new_df.iterrows():
        #data = [str(row['posting_id'])]
        query = row['query']
        #query = tfidf[dictionary.doc2bow(query)]
        sims = instance[query]
        indices = sims.argsort()[-50:][::-1] # top 50
        data = train_df.iloc[indices,:]['posting_id'].tolist()
        results.append([row['posting_id'], "  ".join(data)])
        pbar.update(1)
        if index % 5000 == 0 and len(results) > 1:
            df = pd.DataFrame(results, columns = ['posting_id','matches'])
            df.to_csv(f'submission_index{index}.csv', index=False)
            results = []
df = pd.DataFrame(results, columns = ['posting_id','matches'])
df.to_csv(f'submission.csv', index=False)
#train_df['sims'] = train_df['query'].apply(lambda x: instance[x].argsort()[-50:][::-1])

train_df.to_csv(f'train_vec.csv', index=False)
'''