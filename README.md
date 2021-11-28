# FINAL PROJECT. PART 3.  Ranking


_The aim of this practice is to find all the documents that contain all the words in the query and sort them by their relevance with regard to the query, applying the classical TF-IDF + cosine similarity ranking function, our score + cosine similarity and word2vec + cosine similarity ranking functions. In other words, to rank a document corpus, which is a set of tweets from the World Health Organization (@WHO) account. We will figure out the performance of the system by comparing the three search engines, success or failure, based on how efficiently it is retrieving data for users. Finally, we will propose another word or document representations apart from the word2vec one and discuss their advantages and drawbacks._

## Starting 🚀

_This project consists of a dataset of tweets from the World Health Organization named: ‘dataset_tweets_WHO.txt’
Also includes a Jupyter notebook where the code is developed._


### Pre-requirements 📋

_Needed imports_

```
from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import time
import json
import re
import unidecode
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing
from gensim.models import Word2Vec
import codecs
import nltk
nltk.download('stopwords')
import gensim
from sklearn.manifold import TSNE
from datetime import datetime

```

### Instalation 🔧

```
pip install gensim

```

### Steps🔩

First part:

 * We built our search engine using two different ways of ranking: 
 	* a. TF-IDF + cosine similarity: Classical scoring.
 	
	* b. Your-Score + cosine similarity: We have chosen to take into consideration the following features in order to add points to the final score:
		- Followers Count of the account
		- Favorite Count + Retweet Count of the tweet
		- Verified account
		- Following by the user
		- Favorited/Retweeted by the user
		- Date time

Second part:

  * We returned a top-20 list of documents for each of the 5 queries, using word2vec + cosine similarity. 
  * In order to measure the quality of the search engine based on word2vec + cosine similarity, we recomputed the ground truth for each of the systems and applied the following evaluation techniques: the mean average precision, mean reciprocal rank and average NDCG metrics. Then, we used them to compare the accuracy of word2vec + cosine similarity, tf-idf + cosine similarity and our score + cosine similarity ranking functions.


## Built with 🛠️

* [Jupyter Notebook] (https://jupyter.org/) 
* [Python] (https://www.python.org/)


## Authors✒️

* **Irina Kireeva** 
* **Esther Flores** 
* **Frida Gloria**

