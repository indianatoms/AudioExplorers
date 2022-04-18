# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 22:46:54 2022

@author: psaff
"""
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import Kmeans

def Statistics(combinedMusic, combinedOther):
    otherMinfreqs = np.zeros(combinedOther.shape[1])
    otherMaxfreqs = np.zeros(combinedOther.shape[1])
    otherMedian = np.zeros(combinedOther.shape[1])
    otherAvg  = np.zeros(combinedOther.shape[1])
    otherStd = np.zeros(combinedOther.shape[1])

    musicMinfreqs = np.zeros(combinedMusic.shape[1])
    musicMaxfreqs = np.zeros(combinedMusic.shape[1])
    musicMedian = np.zeros(combinedMusic.shape[1])
    musicAvg  = np.zeros(combinedMusic.shape[1])
    musicStd = np.zeros(combinedMusic.shape[1])
    for freq in range(0, combinedMusic.shape[1] ):
        musicMaxfreqs[freq] = np.max(combinedMusic[:,freq])
        musicMinfreqs[freq] = np.min(combinedMusic[:,freq])
        musicAvg[freq] = np.average(combinedMusic[:, freq])
        musicMedian[freq] = np.median(combinedMusic[:, freq])
        musicStd[freq] = np.std(combinedMusic[:,freq])
    
        otherMaxfreqs[freq] = np.max(combinedOther[:,freq])
        otherMinfreqs[freq] = np.min(combinedOther[:,freq])
        otherAvg[freq] = np.average(combinedOther[:, freq])
        otherMedian[freq] = np.median(combinedOther[:, freq])
        otherStd[freq] = np.std(combinedOther[:,freq])
    print('-------------------------MUSIC----------------------')    
    print('Max: ', musicMaxfreqs)
    print('Min: ', musicMinfreqs)
    print('avg: ', musicAvg)
    print('median: ', musicMedian)
    print('std dev: ', musicStd)

    print('-------------------------OTHER----------------------')

    print('Max: ', otherMaxfreqs)
    print('Min: ', otherMinfreqs)
    print('avg: ', otherAvg)
    print('median: ', otherMedian)
    print('std dev: ', otherStd)

#PCA
def PCA_func(combinedMusic, combinedOther):

    combinedMusic = StandardScaler().fit_transform(combinedMusic)
    combinedOther = StandardScaler().fit_transform(combinedOther)
    pca = PCA(n_components=5)
    combinedMusic = pca.fit_transform(combinedMusic)
    combinedOther = pca.fit_transform(combinedOther)
    print("PCA variance: ", pca.explained_variance_ratio_)

    samples = combinedMusic.shape[0]; #1050
    frequencyBands = combinedMusic.shape[1]; #30
    combinedAll = np.vstack((combinedMusic, combinedOther))
    return combinedMusic, combinedOther, combinedAll


#combinedMusic = np.load('combinedMusicv2.npy') #all entries with music, shape 1500*79 x 30
#combinedOther = np.load('combinedOtherv2.npy')

#combinedMusic, combinedOther, combinedAll = PCA_func(combinedMusic, combinedOther)
#Statistics(combinedMusic, combinedOther)



#Kmeans.Kmeans(combinedMusic, combinedOther, combinedAll)


