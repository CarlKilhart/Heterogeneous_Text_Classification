from utils.tfidf import tfidf_buf
import itertools
import math
from collections import defaultdict
from time import time
from tqdm import tqdm
import pickle
import argparse
import pandas as pd
import numpy as np
import dgl
import torch
import scipy.sparse as sp

def build_vocab(text):
    vocab_set = set()
    for t in text:
        words = t.split(' ')
        for w in words:
            vocab_set.add(w)
    print(f"Vocab size: {len(vocab_set)}")
    vocab_set = list(vocab_set)
    
    return vocab_set

def build_tfidf(vocab_set, text):
    buf = tfidf_buf(vocab_set)
    tfidf_count = buf.fit(text)
    
    # Get tfidf edge
    tfidf_count = tfidf_count.tocoo()
    row = tfidf_count.row
    col = tfidf_count.col
    data = tfidf_count.data
    col = col+len(text)
    
    return buf, row, col, data

def build_pmi(buf, text):
    u_list = []
    v_list = []
    w_list = []
    
    # Get window
    window_size = 20
    threshold = 0.0
    word_window_freq = defaultdict(int)  # w(i)  单词在窗口单位内出现的次数
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0
    for words in tqdm(text, desc="Split by window"):
        windows = list()

        if isinstance(words, str):
            words = words.split()
        length = len(words)

        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(list(set(window)))

        for window in windows:
            for word in window:
                word_window_freq[word] += 1

            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_len += len(windows)

    # Count pmi
    pmi_edge_lst = list()
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        # Cal_pmi
        p_i = word_freq_1 / windows_len
        p_j = word_freq_2 / windows_len
        p_i_j = W_i_j / windows_len
        pmi = math.log(p_i_j / (p_i * p_j))
        
        if pmi <= threshold:
            continue
        pmi_edge_lst.append([word_pair[0], word_pair[1], pmi])
        
    print("Total number of edges between word:", len(pmi_edge_lst))

    for edge_item in pmi_edge_lst:
        word_indx1 = buf.word2id[edge_item[0]]
        word_indx2 = buf.word2id[edge_item[1]]
        if word_indx1 == word_indx2:
            continue
        
        u_list.append(word_indx1+len(text))
        v_list.append(word_indx2+len(text))
        w_list.append(edge_item[2])
        
    return u_list, v_list, w_list
        
def build_textgcn_graph(dataset, read_vocab=False, ignore_ind=False):
    with open(f'data/dataset/{dataset}/dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    trainset = data['trainset']
    testset = data['testset']
    indset = data['indset']
    if ignore_ind:
        testset = pd.concat([testset, indset], axis=0)
    
    text = trainset['text'].str.lower().tolist()+testset['text'].str.lower().tolist()
    if read_vocab:
        vocab_set = pd.read_json(f'data/dataset/{dataset}/tokenizer/vocab.json', typ='series').index.tolist()
    else:
        vocab_set = build_vocab(text)

    # 获得tfidf权重矩阵
    start = time()
    buf, row, col, data = build_tfidf(vocab_set, text)
    tfidf_time = time() - start
    
    # Get pmi edge
    pmi_start = time()
    u_list, v_list, w_list = build_pmi(buf, text)
    pmi_time = time() - pmi_start
    print("pmi time:", pmi_time)
    
    # Merge
    u_arr = np.concatenate([np.array(u_list), row])
    v_arr = np.concatenate([np.array(v_list), col])
    w_arr = np.concatenate([np.array(w_list), data])
    
    # Symmetrization
    temp = sp.csr_matrix((w_arr, (u_arr, v_arr)), shape=(u_arr.max()+1, u_arr.max()+1))
    temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
    temp = temp.tocoo()
    u_arr = temp.row
    v_arr = temp.col
    w_arr = temp.data
    
    # Build graph
    graph = dgl.graph((u_arr, v_arr))
    graph.edata['eweight'] = torch.tensor(w_arr, dtype=torch.float32)
    
    # Save
    print("total time:", pmi_time + tfidf_time)
    with open(f'data/dataset/{dataset}/graph.pkl', 'wb') as f:
        pickle.dump({'graph_list':[graph], 'buf': buf}, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='The dataset name for textgcn model')
    args = parser.parse_args()
    
    build_textgcn_graph(args.dataset)