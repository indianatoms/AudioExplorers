# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 22:46:54 2022

@author: psaff
"""
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def Kmeans(combinedMusic, combinedOther, combinedAll):
    print('computing kmeans: ')
    kmeans = KMeans(n_clusters =2, random_state=0).fit(combinedAll)
    # print(kmeans.cluster_centers_)

    print('MUSIC PREDICTION: ')
    print('here  everything should be music(0)')
    musicpredictions = kmeans.predict(combinedMusic)

    print('total: ', musicpredictions.shape[0])
    total = musicpredictions.shape[0]
    music = np.count_nonzero(musicpredictions == 0)
    other = np.count_nonzero(musicpredictions == 1)
    print('classified as music: ', music)
    print('classified as other: ', other)

    print('OTHER PREDICTION: ')
    print('here everything should be other (1)')
    otherpredictions = kmeans.predict(combinedOther)
    print('total: ', otherpredictions.shape[0])
    total = otherpredictions.shape[0]
    music = np.count_nonzero(otherpredictions == 0)
    other = np.count_nonzero(otherpredictions == 1)
    print('classified as music: ', music)
    print('classified as other: ', other)


