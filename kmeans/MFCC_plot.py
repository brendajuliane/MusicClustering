import librosa
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


audio_files = glob('../base/*.mp3')
print("Pasta com", len(audio_files), "audios carregados")

n_mfcc = 13 
n_clusters = 3  

all_mfccs = [] 

# Extrair MFCCs de cada arquivo de áudio
for audio_file in audio_files:
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    all_mfccs.append(mfcc_mean)

X = np.array(all_mfccs)

X = X[:,1:]


cluster = KMeans(n_clusters=3)
cluster.fit(X)
cluster_labels = cluster.labels_

data_3d = X[:, :3]

# Criando um gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['red'] * 10 + ['green'] * 10 + ['blue'] * 10

# Plotando os pontos
ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=colors)

plt.show()