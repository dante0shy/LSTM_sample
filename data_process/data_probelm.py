import json
import numpy as np
import nltk
import os
from nltk.corpus import stopwords
# nltk.download()
if __name__=='__main__':
    input_file = '../data_for_train/train.json'
    output_dir = '../data_for_train'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    data = json.load(open(input_file,'r'))
    stemmer = nltk.stem.PorterStemmer()
    word_list = []
    max_len = 0
    for d,l in data:
        tokens = nltk.tokenize.word_tokenize(d)
        tokens = [stemmer.stem(token) for token in tokens]
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        max_len = max(max_len,len(tokens))
        word_list.extend(tokens)
    word_list = list(set(word_list))
    json.dump(word_list,open(os.path.join(output_dir,'word_list.json'),'w'))
    print max_len
    pass