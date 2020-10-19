
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics


'''
Initial sample dots
'''
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.5,0.5, 0.5, 0.5]
sample_dots, cluster_id = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.5, 0.5, 0.5, 0.5], random_state =9)
x_location = sample_dots[:,0]
y_location = sample_dots[:,1]
plt.scatter(x_location,y_location, marker='o') # the dots before k-means

'''
K-Means algorithm using API
'''
# predict the cluster id
kmeans = KMeans(init='random',n_init = 1,n_clusters=4, random_state=9)
pred_cluster_id = kmeans.fit_predict(sample_dots)
plt.scatter(x_location,y_location, c = pred_cluster_id) # the dots after k-means 

# get the cluster centroids 
centroids = kmeans.fit(sample_dots).cluster_centers_
x_centroid = centroids[:,0]
y_centroid = centroids[:,1]
plt.scatter(x_centroid,y_centroid,s = 400,marker='*')
plt.show()

# compute the score of algorithm
score1 = metrics.calinski_harabasz_score(sample_dots, pred_cluster_id)
score2 = metrics.calinski_harabasz_score(sample_dots, cluster_id)
print(score1)
print(score2)

