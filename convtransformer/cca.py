import numpy as np
from scipy import linalg
import torch
import seaborn
import os
import natsort
import pickle
import matplotlib.pyplot as plt
from scipy.stats import entropy
import argparse

class CCA:
    def __init__(self):
        self.a = None
        self.b = None
    
    def train(self, X, Y):
        Nx, cx = X.shape
        Ny, cy = Y.shape
        
        X = (X - np.mean(X, 0)) / np.std(X, 0)
        Y = (Y - np.mean(Y, 0)) / np.std(Y, 0)
        
        data = np.concatenate([X, Y], axis = 1)
        cov = np.cov(data, rowvar=False)
        N, C = cov.shape
        Sxx = cov[0:cx, 0:cx]
        Syy = cov[cx:C, cx:C]
        Sxy = cov[0:cx, cx:C]
        Sxx_ = linalg.sqrtm(np.linalg.inv(Sxx))
        Syy_ = linalg.sqrtm(np.linalg.inv(Syy))
        M = Sxx_.T.dot(Sxy.dot(Syy_))
        U, S, V = np.linalg.svd(M, full_matrices=False)
        u = U[:, 0]
        v = V[0, :]
        self.a = Sxx_.dot(u)
        self.b = Syy_.dot(v)
        
    def predict(self, X, Y):
        X_ = X.dot(self.a)
        Y_ = Y.dot(self.b)
        return X_, Y_
    
    def cal_corrcoef(self, X, Y):
        X_, Y_ = self.predict(X, Y)
        return np.corrcoef(X_, Y_)[0,1]


parser = argparse.ArgumentParser(description="path to attention matrices")
parser.add_argument('--path_X', type=str, help='path to bilingual attention matrices')
parser.add_argument('--path_Y', type=str, help='path to multilingual attention matrices')

args = parser.parse_args()

path_X = args.path_X

samples_X = os.listdir(path_X)
samples_X = natsort.natsorted(samples_X,reverse=False)

for sample_X in samples_X:
    attn_X = []
    sample_X_attn = os.listdir(path_X+sample_X)
    for attn in sample_X_attn:
        if(os.path.splitext(attn)[-1] == ".pt" and os.path.splitext(attn)[0] != "self-attention"):
            attn_X.append(torch.load(path_X+sample_X+'/'+attn, map_location='cpu'))
            X = torch.cat(attn_X, dim=1)
            X = X[0,:,:].numpy().T
            pickle.dump(X,open(path_X+sample_X+'/'+sample_X+'_attn', 'wb')) 
        else:
            continue

path_Y = args.path_Y

samples_Y = os.listdir(path_Y)
samples_Y = natsort.natsorted(samples_Y, reverse=False)

for sample_Y in samples_Y:
    attn_Y = []
    sample_Y_attn = os.listdir(path_Y+sample_Y)
    for attn in sample_Y_attn:
        if(os.path.splitext(attn)[-1] == ".pt" and os.path.splitext(attn)[0] != "self-attention"):
            attn_Y.append(torch.load(path_Y+sample_Y+'/'+attn, map_location='cpu'))
            Y = torch.cat(attn_Y, dim=1)
            Y = Y[0,:,:].numpy().T
            pickle.dump(Y,open(path_Y+sample_Y+'/'+sample_Y+'_attn', 'wb')) 
        else:
            continue

for i in range(len(path_Y)):
    for sample in samples_X: 
        X = open(path_X + sample + '/' + sample + '_attn', 'rb')
        attn_matrix_X = pickle.load(X)
        X_row, X_col = attn_matrix_X.shape
        
        Y = open(path_Y[i] + sample + '/' + sample + '_attn', 'rb')
        attn_matrix_Y = pickle.load(Y)
        Y_row, Y_col = attn_matrix_Y.shape
        tar_min = min(X_col, Y_col)
        src_min = min(X_row, Y_row)
        
        X_final = attn_matrix_X[:src_min, :tar_min]
        Y_final = attn_matrix_Y[:src_min, :tar_min]
        
        pickle.dump(X_final, open(path_X+sample+'/'+sample+'_attn_final', 'wb'))
        pickle.dump(Y_final, open(path_Y[i]+sample+'/'+sample+'_attn_final', 'wb'))

for i in range(len(path_Y)):
    corrcoef_real = []
    corrcoef_imag = []
    corrcoef = []
    for sample in samples_X: 
        fX = open(path_X + sample + '/' + sample + '_attn_final', 'rb')
        attn_matrix_X = pickle.load(fX)
        fY = open(path_Y[i] + sample + '/' + sample + '_attn_final', 'rb')
        attn_matrix_Y = pickle.load(fY)
        clf = CCA()
        clf.train(attn_matrix_X, attn_matrix_Y)
        corrcoef_real.append(clf.cal_corrcoef(attn_matrix_X, attn_matrix_Y).real)
        corrcoef_imag.append(clf.cal_corrcoef(attn_matrix_X, attn_matrix_Y).imag)
        corrcoef.append(clf.cal_corrcoef(attn_matrix_X, attn_matrix_Y))  
    pickle.dump(corrcoef, open(path_Y[i]+'corrcoef', 'wb'))
    pickle.dump(corrcoef_real, open(path_Y[i]+'corrcoef_real', 'wb'))
    pickle.dump(corrcoef_imag, open(path_Y[i]+'corrcoef_imag', 'wb'))