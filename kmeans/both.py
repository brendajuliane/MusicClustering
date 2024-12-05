from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from glob import glob
import numpy as np
import librosa
import csv
from random import *

audio_files = glob('../base/different-singer/*.mp3')
audio_files = audio_files + glob('../base/*.mp3')
print("Pasta com", len(audio_files), "audios carregados")

features = []

# Extração de cada arquivo
for audio_file in audio_files:
    y, sr = librosa.load(audio_file)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    chromagram_mean = np.mean(chromagram, axis=1)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    mfcc_mean = np.mean(mfcc, axis=1)

    mfcc_mean_normalized = (mfcc_mean - np.min(mfcc_mean)) / (np.max(mfcc_mean) - np.min(mfcc_mean))

    features.append(np.append(mfcc_mean_normalized, chromagram_mean))

X = np.array(features)

# Removendo o primeiro MFCC
X = X[:,1:]

rs = randint(0, 1000)

for k in range(2,13): 
    cluster = KMeans(n_clusters=k, init="random", random_state=rs)
    cluster.fit(X)
    cluster_labels = cluster.labels_

    silhouette_avg = silhouette_score(X, cluster.labels_)
    davies_boudin = davies_bouldin_score(X, cluster.labels_)

    with open(f'added_base_modified_kmeans_both_results_rs_{rs}.csv', 'a', newline='') as csvfile:
        fieldnames = ['Cluster', 'Nome do Arquivo']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Cluster': k, 'Nome do Arquivo': silhouette_avg})

        for i, audio_file in enumerate(audio_files):
            audio_name = audio_file.replace('./base\\', '')
            writer.writerow({'Cluster': cluster_labels[i], 'Nome do Arquivo': audio_name})

    with open(f'added_base_modified_both_metrics_rs_{rs}.csv', 'a', newline='') as csvfile:
        fieldnames = ['Cluster', 'silhueta', 'davies']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if (k==2):
            writer.writeheader()
            
        writer.writerow({'Cluster': k, 'silhueta': silhouette_avg, 'davies': davies_boudin})