import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd 

def clean_str(string):
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(data_file,testing=False):

    # Load data from files
    df = pd.read_csv(data_file,encoding = 'latin1')
    s_c = []
    sub_c = []
    for index,row in df.iterrows():
        if row['category'] not in s_c:
            s_c.append(row['category'])
        if row['subcategory'] not in sub_c:
            sub_c.append(str(row['subcategory']))
   
    print('Categories: ',s_c)
    # print('Sub Categories: ',sub_c)

    examples = []
    labels = []
    for index, row in df.iterrows():
        if testing:
            examples.append(str(row['short_description']).strip())
            labels.append(str(row['category']))
        else:
            for i in range(len(s_c)):
                if s_c[i] == str(row['category']):
                    examples.append(str(row['short_description']).strip())
                    l = np.zeros(len(s_c))
                    l[i] = 1.0
                    labels.append(l)
    return [examples, np.array(labels)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
