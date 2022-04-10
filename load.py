# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:11:00 2022

@author: psaff
"""

import numpy as np
from sklearn.cluster import KMeans


musicData  = np.load('music_data.npy')
otherData = np.load('other_data.npy')


samples = musicData.shape[0]; #1050
frequencyBands = musicData.shape[1]; #30
timeframes = musicData.shape[2]; #79

# # --------------------------------------
# ### COMBINING THE DATA FOR KMEANS ###
#to compile all data at once:
    #takes 10+ minutes!
counter = 0
combinedMusic = musicData[counter,:,0]
combinedOther = otherData[counter,:,0]
np.save('combinedMusicv2.npy', combinedMusic)
np.save('combinedOtherv2.npy', combinedOther)
 
while counter< 10499:
    print('data from: ', counter)
    combinedMusic = musicData[counter,:,0]
    combinedOther = otherData[counter,:,0]
    print(combinedMusic.shape)
    for sample in range(counter+1, counter+499):
        for t in range(0, 79):
            combinedMusic = np.vstack((combinedMusic, musicData[sample, :, t]))
            combinedOther = np.vstack((combinedOther, otherData[sample, :, t]))
    
    #combinedAll = np.load('combinedAll.npy')
    combinedMusicAll = np.load('combinedMusicv2.npy')
    combinedOtherAll = np.load('combinedOtherv2.npy')
    print('sizes: ')
    print(combinedMusicAll.shape, ' ', combinedOtherAll.shape)
    
    combinedMusicComplete = np.vstack((combinedMusicAll, combinedMusic))
    combinedOtherComplete = np.vstack((combinedOtherAll, combinedOther))        
    #combined = np.vstack((combinedMusicAll, combinedOtherAll))
    # combinedAllComplete = np.vstack((combinedAll, combined))
    
    np.save('combinedMusicv2.npy', combinedMusicComplete)
    np.save('combinedOtherv2.npy', combinedOtherComplete)
    counter = counter + 500
    
    print('deleting')
    del combinedMusic, combinedOther, combinedMusicAll, combinedOtherAll, combinedMusicComplete, combinedOtherComplete
#---------------------
#---------------------
print('DONEE')
#combinedAll = np.load('combinedAll.npy')
combinedMusic = np.load('combinedMusic.npy') #all entries with music, shape 1500*79 x 30
combinedOther = np.load('combinedOther.npy')
combinedAll = np.vstack((combinedMusic, combinedOther))


print(combinedAll.shape, combinedMusic.shape, combinedOther.shape)
print('computing kmeans')
kmeans = KMeans(n_clusters =2, random_state=0).fit(combinedAll)
print(kmeans.cluster_centers_)

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

# print(combinedMusic.shape)

# for sample in range(0, samples ):
    
#     for band in range(0, frequencyBands):
    
#         for t in range(0, timeframes):

# for sample in range(0, samples ):
#     for t in range(0, timeframes):
#             freqsAtTimeframe = musicData[sample, :, t] # all frequecies at one timeframe of one sample
#             minfreq = np.min(freqsAtTimeframe)
#             maxfreq = np.max(freqsAtTimeframe)
#             diff = np.subtract(maxfreq, minfreq)

# print('standard deviation of one sample')
# allStdev = np.empty((samples, frequencyBands));
# for sample in range(0, samples ):
#     oneStdev = np.empty((frequencyBands));
#     for band in range(0, frequencyBands):
#         onefreqAtAllTimeframes = musicData[sample, band,:] #one frequency at all timeframes of one sample
#         #print(onefreqAtAllTimeframes)
#         stdev = np.std(onefreqAtAllTimeframes)
#         np.append(oneStdev[band], stdev)
#     np.append(allStdev[sample], oneStdev)
            
# print(diff)   
# print('standard dev: ')
# print(allStdev)  
# print(allStdev.shape)      
    
# print(i)
