#!/usr/bin/env python

text = "Digital health can transform health care."
print(text)

print(text.replace("can transform", "is transforming"))

print(text.upper())

import re

words = re.sub(r'[^a-zA-Z0-9 ]', '', text)
print(words)

words_list = words.split()
print(words_list)

from collections import Counter
counts = Counter(words_list)
print(counts)

import numpy as np

array = np.array([["This sentence has 20 characters", "long"],
                  ["This 8", "short"]])
print(array)

import pandas as pd

data = {
        "Text": ["I am happy about the consultation",
                 "The doctor did not address my worries"],
        "Label": ["Good", "Bad"]
        }

df = pd.DataFrame(data)
print(df)


import nltk

nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

words = word_tokenize(text)
print(words)

from nltk.tokenize import sent_tokenize

text = "Digital health can transform health care. Hopefully in a better way."

sentences = sent_tokenize(text)
print(sentences)

import re

text = "T98his se$1£ntence! would ^%*9) benn333efit£ from so:;\/me: clea]ning;"
text_clean = re.sub(r'[^a-zA-Z\s]', '', text)
print(text_clean)


import nltk
from nltk.stem import PorterStemmer, LancasterStemmer

porter = PorterStemmer()
lancaster = LancasterStemmer()

words = ["running", "runner", "ran", "easily", "fairly", "fair"]

porter_stems = [porter.stem(w) for w in words]
lancaster_stems = [lancaster.stem(w) for w in words]

print("Original words: ", words)
print("Porter stems:   ", porter_stems)
print("Lancaster words:", lancaster_stems)

from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

text = "The pateint reported severe pain in the chest and arm."

text_filtered = " ".join([w for w in text.split() if w.lower() not in stop_words])

print(text_filtered)
