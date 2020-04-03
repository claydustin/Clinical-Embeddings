from collections import Counter, defaultdict, Iterable
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Raw_Datasets import Process_Raw_Inputs, Combine_Codes
import pandas as pd

list2d = [[1,2,3], [4,5,6], [7], [8,9]]

from sklearn.manifold import TSNE


def flatten(x):
    if isinstance(x, Iterable) and not isinstance(x, str):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

class GloveDataset:
   
    def __init__(self, patient_list):
        self._patient_list = patient_list
        self._codes = flatten(patient_list)
        word_counter = Counter()
        word_counter.update(self._codes)
        self._code2id = {w:i for i, (w,_) in enumerate(word_counter.most_common())} ##codes ordered from greatest to least
        self._id2code = {i:w for w, i in self._code2id.items()}
        self._vocab_len = len(self._code2id)
       
        self._id_patient_list = [[[self._code2id[code] for code in visit] for visit in patient if len(visit) > 1] for patient in self._patient_list]
       
        self._create_coocurrence_matrix()
       
        print("# of codes: {}".format(len(self._codes)))
        print("Number of Distinct Codes: {}".format(self._vocab_len))
       
    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)
        cp_patients= self._id_patient_list #create a lookup visit from original to pop codes from ensuring occurence matrix operations aren't doubled
        for patient in cp_patients:
            for visit in patient:      
                while len(visit) > 1:
                    i_code = visit.pop(0)
                    for j_code in visit:
                        cooc_mat[i_code][j_code]+=1

                   
        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()
       
        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)
               
        self._i_idx = torch.LongTensor(self._i_idx).cuda()
        self._j_idx = torch.LongTensor(self._j_idx).cuda()
        self._xij = torch.FloatTensor(self._xij).cuda()
   
   
    def get_batches(self, batch_size):
        #Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
       
        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]




class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)
       
        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()
       
    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
       
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
       
        return x

def weight_func(x, x_max, alpha):
    """weight_func: the weighting function
    sets a max weight based on x_max
    alpha: hyparameter to set variation in weights
    """
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx.cuda()


def wmse_loss(weights, inputs, targets):
    """linear weighting function of mse which uses weight_func
   
    """
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss).cuda()



if __name__ == '__main__':

    df = pd.read_csv("Data/visit_inputs.csv")
    df.columns = ['IDX', 'IDX_TYPE', 'BILLABLE_START_DT', 'CODE_TYPE', 'CODE_VALUE']
    inputs = Process_Raw_Inputs(df)
    with open('Data/visit_inputs_json.txt', 'r') as inFile:
        inputs = json.load(inFile)


    patientList= Combine_Codes(inputs, code_types=['ICD10'])

    EMBED_DIM = 300
    N_EPOCHS = 100
    BATCH_SIZE = 2048
    X_MAX = 1000
    ALPHA = 0.75
    dataset = GloveDataset(patientList)
    glove = GloveModel(dataset._vocab_len, EMBED_DIM)
    glove.cuda()

    optimizer = optim.Adagrad(glove.parameters(), lr=0.05)

    n_batches = int(len(dataset._xij) / BATCH_SIZE)
    loss_values = list()
    for e in range(1, N_EPOCHS+1):
        batch_i = 0
   
        for x_ij, i_idx, j_idx in dataset.get_batches(BATCH_SIZE):
       
            batch_i += 1
       
            optimizer.zero_grad()
       
            outputs = glove(i_idx, j_idx)
            weights_x = weight_func(x_ij, X_MAX, ALPHA)
            loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
       
            loss.backward()
       
            optimizer.step()
       
            loss_values.append(loss.item())
       
            if batch_i % 100 == 0:
                print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, N_EPOCHS, batch_i, n_batches, np.mean(loss_values[-20:])))  
   
        print("Saving model...")
        torch.save(glove.state_dict(), "text8.pt")
    loss_values
    plt.plot(loss_values)

    emb_i = glove.wi.weight.cpu().data.numpy()
    emb_j = glove.wj.weight.cpu().data.numpy()
    emb = emb_i + emb_j
    top_k = 300
    tsne = TSNE(metric='cosine', random_state=123)
    embed_tsne = tsne.fit_transform(emb[:top_k, :])
    fig, ax = plt.subplots(figsize=(14, 14))
    for idx in range(top_k):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(dataset._id2code[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
