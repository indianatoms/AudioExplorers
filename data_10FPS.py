import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statistics



combinedMusic = np.load('combinedMusicv2.npy') #all entries with music, shape 1500*79 x 30
combinedOther = np.load('combinedOtherv2.npy')

combinedMusic, combinedOther, combinedAll_data= statistics.PCA_func(combinedMusic, combinedOther)

i= 0
new_i = 0
framesToCombine = 5
music_10FPS = np.zeros((int(combinedMusic.shape[0]/(framesToCombine)), 5*framesToCombine)) # create array merged frames * total samples x PCA comp
other_10FPS = np.zeros((int(combinedMusic.shape[0]/(framesToCombine)), 5*framesToCombine))
frames = np.zeros((framesToCombine, 5))
frames_other = np.zeros((framesToCombine, 5))
print(combinedMusic.shape[1])
while i + framesToCombine +1 < combinedMusic.shape[0]:
    for cnt in range(i, i-1+framesToCombine):
        index = cnt-i
        frames[index] = combinedMusic[cnt,:]
        frames_other[index] = combinedOther[cnt,:]
    flatten = frames.flatten()
    flatten_other = frames_other.flatten()
    other_10FPS[new_i] = flatten_other;
    music_10FPS[new_i] = flatten;
    i = i+framesToCombine
    new_i = new_i +1

print(music_10FPS)
np.save('music5frames.npy', music_10FPS )
np.save('other5frames.npy', other_10FPS )