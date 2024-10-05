from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn_extra.cluster import KMedoids
from glob import glob
import numpy as np
import librosa
import csv
from random import *

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

# Geração de valor aleatório
rs = randint(0, 1000)

for k in range(2,13): 
    cluster = KMedoids(n_clusters=k, init="random", random_state=rs)
    cluster.fit(X)
    cluster_labels = cluster.labels_

    silhouette_avg = silhouette_score(X, cluster.labels_)

    print(f'\n\n\n------ Para {k} clusters, silhouette é {silhouette_avg} ------')
    print(f'------ Davies é {davies_bouldin_score(X, cluster.labels_)}')

    with open(f'kmedoids_chroma_results_rs_{rs}.csv', 'a', newline='') as csvfile:
        fieldnames = ['Cluster', 'Nome do Arquivo']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Cluster': k, 'Nome do Arquivo': silhouette_avg})

        for i, audio_file in enumerate(audio_files):
            audio_name = audio_file.replace('../base\\', '')
            writer.writerow({'Cluster': cluster_labels[i], 'Nome do Arquivo': audio_name})
