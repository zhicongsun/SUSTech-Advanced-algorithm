
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans
# from sklearn import metrics


# '''
# Initial sample dots
# '''
# # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.5,0.5, 0.5, 0.5]
# sample_dots, cluster_id = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.5, 0.5, 0.5, 0.5], random_state =9)
# x_location = sample_dots[:,0]
# y_location = sample_dots[:,1]
# plt.scatter(x_location,y_location, marker='o') # the dots before k-means

# '''
# K-Means algorithm using API
# '''
# # predict the cluster id
# kmeans = KMeans(init='random',n_init = 1,n_clusters=4, random_state=9)
# pred_cluster_id = kmeans.fit_predict(sample_dots)
# plt.scatter(x_location,y_location, c = pred_cluster_id) # the dots after k-means 

# # get the cluster centroids 
# centroids = kmeans.fit(sample_dots).cluster_centers_
# x_centroid = centroids[:,0]
# y_centroid = centroids[:,1]
# plt.scatter(x_centroid,y_centroid,s = 400,marker='*')
# plt.show()

# # compute the score of algorithm
# score1 = metrics.calinski_harabasz_score(sample_dots, pred_cluster_id)
# score2 = metrics.calinski_harabasz_score(sample_dots, cluster_id)
# print(score1)
# print(score2)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics


# '''
# K-Means algorithm using API
# '''
# # predict the cluster id
# kmeans = KMeans(init='random',n_init = 1,n_clusters=4, random_state=9)
# pred_cluster_id = kmeans.fit_predict(sample_dots)
# plt.scatter(x_location,y_location, c = pred_cluster_id) # the dots after k-means 

# # get the cluster centroids 
# centroids = kmeans.fit(sample_dots).cluster_centers_
# x_centroid = centroids[:,0]
# y_centroid = centroids[:,1]
# plt.scatter(x_centroid,y_centroid,s = 400,marker='*')
# plt.show()

# # compute the score of algorithm
# score1 = metrics.calinski_harabasz_score(sample_dots, pred_cluster_id)
# score2 = metrics.calinski_harabasz_score(sample_dots, cluster_id)
# print(score1)
# print(score2)




# def k_means(sites, k, init_centers):
#     centers = init_centers[:]
#     while True:
#         new_centers = []
#         changed_centers = []
#         # Assign the center to the site
#         for site in sites:
#             center = assign_center(centers, site)[1]
#             site.center = center
#             center.sites.append(site)
#         # Recalculate center
#         for center in centers:
#             new_center = cal_center(center.sites)
#             new_centers.append(new_center)
#             if new_center.location != center.location:
#                 changed_centers.append(new_center)
#         if len(changed_centers) == 0:
#             return centers
#         centers = new_centers[:]

class Site:
    def __init__(self,x_location,y_location):
        self.x_location = x_location
        self.y_location = y_location

class Center:
    def __init__(self,x_location,y_location):
        self.x_location = x_location
        self.y_location = y_location
        self.sites = []

if __name__ == "__main__":
    '''
    Initial sample sites and initial centers
    '''
    # define super param
    CENTERS=[[-1,-1], [0,0], [1,1], [2,2]]
    K = len(CENTERS)
    N_SAMPLES = 1000
    CLUSTER_STD = [0.2, 0.2, 0.2, 0.2]
    
    # inital sample sites
    Object_sites = []
    sample_sites_locations = []
    temp_sample_sites_locations, cluster_id = make_blobs(n_samples=N_SAMPLES, n_features=2, centers = CENTERS, cluster_std=CLUSTER_STD, random_state =9)
    # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.5,0.5, 0.5, 0.5]
    x_sample_location = temp_sample_sites_locations[:,0]
    x_sample_location = x_sample_location.tolist()
    y_sample_location = temp_sample_sites_locations[:,1]
    y_sample_location = y_sample_location.tolist()

    for n_sample in range(N_SAMPLES):
        sample_sites_locations.append([x_sample_location[n_sample],y_sample_location[n_sample]])
        Object_sites.append( Site(sample_sites_locations[n_sample][0],sample_sites_locations[n_sample][1]) )

    # inital centers
    Object_centers = []
    initial_centers_locations= []
    x_center_location = []
    y_center_location = []
    rand_squence = np.random.randint(0,N_SAMPLES,K)
    
    for k in range(K):
        x_center_location.append(x_sample_location[rand_squence[k]])
        x_sample_location.pop(rand_squence[k])
        y_center_location.append(y_sample_location[rand_squence[k]])
        y_sample_location.pop(rand_squence[k])
        initial_centers_locations.append([x_center_location[k],y_center_location[k]])
        Object_centers.append( Center(initial_centers_locations[k][0],initial_centers_locations[k][1]) )

    plt.scatter(x_sample_location,y_sample_location, marker='o') # the sites before k-means
    plt.scatter(x_center_location,y_center_location,s = 400,marker='*')
    plt.show()


# a = []
# class A:
#     def __init__(self):
#         self.k = 1
#         print('a')
# for i in range(3):
#     a.append(A())
# print(a[1].k)
