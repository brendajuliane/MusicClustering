import librosa
import numpy as np
from glob import glob
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import csv

audio_files = glob('../base/*.mp3')
print("Pasta com", len(audio_files), "audios carregados")

n_mfcc = 13 
all_mfccs = [] 

# Extrair MFCCs de cada arquivo de áudio
for audio_file in audio_files:
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=512)
    mfcc_mean = np.mean(mfcc, axis=1)
    all_mfccs.append(mfcc_mean)

X = np.array(all_mfccs) 

# Removendo primeiro MFCC
X = X[:,1:]

for k in range(2,13): 
    # Aplicar o K-Means
    cluster = AgglomerativeClustering(n_clusters=k)
    cluster.fit_predict(X)
    cluster_labels = cluster.labels_

    silhouette_avg = silhouette_score(X, cluster.labels_)

    print(f'\n\n\n------ Para {k} clusters, silhouette é {silhouette_avg} ------\n')

    with open('agglomerative_mfccs_results.csv', 'a', newline='') as csvfile:
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
