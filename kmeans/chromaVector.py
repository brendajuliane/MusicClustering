from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from glob import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import csv

audio_files = glob('../base/*.mp3')
print("Pasta com", len(audio_files), "audios carregados")

chromagrams = []

# Extração de cada arquivo
for audio_file in audio_files:
    y, sr = librosa.load(audio_file)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    chromagram_mean = np.mean(chromagram, axis=1)
    chromagrams.append(chromagram_mean)


X = np.array(chromagrams)

for k in range(2,13): 
    cluster = KMeans(n_clusters=k)
    cluster.fit(X)
    cluster_labels = cluster.labels_

    silhouette_avg = silhouette_score(X, cluster.labels_)

    print(f'\n\n\n------ Para {k} clusters, silhouette é {silhouette_avg} ------\n')

    with open('kmeans_chroma_results.csv', 'a', newline='') as csvfile:
        fieldnames = ['Cluster', 'Nome do Arquivo']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Cluster': k, 'Nome do Arquivo': silhouette_avg})

        for i, audio_file in enumerate(audio_files):
            audio_name = audio_file.replace('./base\\', '')
            writer.writerow({'Cluster': cluster_labels[i], 'Nome do Arquivo': audio_name})

    with open('metrics.csv', 'a', newline='') as csvfile:
        fieldnames = ['Cluster', 'silhueta', 'davies']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if (k==2):
            writer.writeheader()
            
        writer.writerow({'Cluster': k, 'silhueta': silhouette_avg, 'davies': davies_bouldin_score(X, cluster.labels_)})

