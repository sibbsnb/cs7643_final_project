from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from textblob import TextBlob
import nltk
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import matplotlib.pyplot as plt

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('train.csv')

df = df.iloc[:1000]
df['polarity'] = df['title'].map(lambda text: TextBlob(text).sentiment.polarity)
df['word_count'] = df['title'].apply(lambda x: len(str(x).split()))
text = " ".join(t for t in df.title)
stopwords = set(STOPWORDS)

# Generate a word cloud image
wordcloud = WordCloud(width=680, height=500, stopwords=stopwords, max_words=100, background_color="white").generate(
    text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# plt.show()
plt.savefig('world_cloud.png', bbox_inches='tight')

# text polarity
plt.clf()
ax = df['polarity'].plot.hist(bins=50)

plt.xlabel('polarity')
plt.ylabel('count')
plt.title('Sentiment Polarity Distribution')
ax.get_figure().savefig('text_polarity.png', bbox_inches='tight')

# word count
plt.clf()
ax = df['word_count'].plot.hist(bins=100)
plt.xlabel('word count')
plt.ylabel('count')
plt.title('product title Word Count Distribution')
ax.get_figure().savefig('word_count.png', bbox_inches='tight')

#


bi_dict = dict()
bg_measures = BigramAssocMeasures()
for idx, row in df.iterrows():
    words = nltk.word_tokenize(row['title'])
    bi_finder = BigramCollocationFinder.from_words(words)
    bi_collocs = bi_finder.nbest(bg_measures.likelihood_ratio, 10)
    for colloc in bi_collocs:
        bi_dict[colloc] += 1

sid = SentimentIntensityAnalyzer()
sentiment_summary = dict()
for idx, row in df.iterrows():
    sentences = nltk.tokenize.sent_tokenize(row['title'])
    for sentence in sentences:
        sentiment_score = sid.polarity_scores(sentence)
        if sentiment_score["compound"] == 0.0:
            sentiment_summary["neutral"] += 1
        elif sentiment_score["compound"] > 0.0:
            sentiment_summary["positive"] += 1
        else:
            sentiment_summary["negative"] += 1
