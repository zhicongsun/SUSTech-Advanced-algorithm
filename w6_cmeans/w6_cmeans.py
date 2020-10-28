import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib import animation
from PIL import Image
import matplotlib.patches as mpatches
from math import pi
from numpy import cos, sin
import copy

def cal_center(sites,initial_centers,nu_matrix,m):
    centers = copy.deepcopy(initial_centers)
    n_centers = nu_matrix.shape[1]
    n_sites = len(sites)
    for j in range(n_centers):
        up_xtotal = 0.0
        down_xtotal = 0.0
        up_ytotal = 0.0
        down_ytotal = 0.0
        for i in range(n_sites):
            up_xtotal = up_xtotal + sites[i].x_location * nu_matrix[i][j] ** m
            up_ytotal = up_ytotal + sites[i].y_location * nu_matrix[i][j] ** m
            down_xtotal = down_xtotal + nu_matrix[i][j] ** m
            down_ytotal = down_ytotal + nu_matrix[i][j] ** m
        centers[j].x_location = up_xtotal / down_xtotal
        centers[j].y_location = up_ytotal / down_ytotal
    return centers

def cal_numatrix(distance_matrix,centers,m,n,K):
    nu_matrix = np.zeros((n,K))
    for i in range(n):
        for j in range(K):
            total = 0
            for k in range(K):
                total = total + (distance_matrix[i][j] / distance_matrix[i][k]) ** (2/(m-1))
            total = total ** (-1)
            nu_matrix[i][j] = total
    return nu_matrix

def cal_distance(sites,centers):
    n_sites = len(sites)
    n_centers = len(centers)
    distance_matrix = np.zeros((n_sites,n_centers))
    for i in range(n_sites):
        for j in range(n_centers):
            distance_matrix[i][j] = ( ((sites[i].x_location - centers[j].x_location) **2)+((sites[i].y_location - centers[j].y_location) **2) ) **0.5
    return distance_matrix

def assign_center(site,centers,distance_matrix):
    # return the id of center which has the sortest distance from site to this center
    site_id = site.id
    min_distance = [distance_matrix[site_id][0],0]
    for j in range(distance_matrix.shape[1]):# compare distances from site to K centers
        if distance_matrix[site_id][j] < min_distance[0]:
            min_distance[0] = distance_matrix[site_id][j]
            min_distance[1] = j
    return centers[min_distance[1]]

def c_means(sites, init_centers,algorithm_kind,m):
    centers = copy.deepcopy(init_centers) 
    n_fig = 0
    while True:
        distance_matrix = cal_distance(sites,centers)
        changed_centers = []
        # assign the center to the site
        for site in sites:
            center = assign_center(site, centers, distance_matrix)
            site.center = center.id
            center.sites.append(site)
        # recalculate center
        nu_matrix = cal_numatrix(distance_matrix,centers,m,len(sites),len(centers))
        new_centers = copy.deepcopy(cal_center(sites,centers,nu_matrix,m))

        for i in range(len(centers)):
            if ((abs(centers[i].x_location-new_centers[i].x_location)>0.01) or (abs(centers[i].y_location-new_centers[i].y_location)>0.01)):
                changed_centers.append(centers[i])
        if len(changed_centers) == 0:
            return centers,n_fig
        centers = copy.deepcopy(new_centers) 

        plt.clf()
        color_squence = ['darkorchid','limegreen','sandybrown','lightslategrey','rosybrown','sienna','seagreen']
    
        #draw sites,each site belongs to a center and has only one color 
        for j in range(len(centers)):
            x_sample_location = []
            y_sample_location = []
            for i in range(len(sites)):
                if sites[i].center == j:
                    x_sample_location.append(sites[i].x_location)
                    y_sample_location.append(sites[i].y_location)  
                    plt.scatter(sites[i].x_location,sites[i].y_location,marker='o',c = color_squence[j%7],alpha = max(nu_matrix[i][:]))

        x_center_location = []
        y_center_location = []
        for i in range(len(centers)):
            x_center_location.append(centers[i].x_location)
            y_center_location.append(centers[i].y_location)    
        plt.scatter(x_center_location,y_center_location,s=400,marker='*',c='red')
        plt.savefig(str(algorithm_kind)+ '_' + str(n_fig)+'.png')
        n_fig = n_fig+1
        plt.pause(0.01)

class Site:
    center = 0
    def __init__(self,id,x_location,y_location):
        self.id = id
        self.x_location = x_location
        self.y_location = y_location

class Center:
    sites = []
    def __init__(self,id,x_location,y_location):
        self.id = id
        self.x_location = x_location
        self.y_location = y_location

if __name__ == "__main__":

    '''
    Define the clusters using super param
    '''   
    algorithm_kind = 'CMeans'
    CENTERS=[[-1,1], [-1,-2], [1,1], [1,-1],[2,2]]
    K = len(CENTERS)
    N_SAMPLES = 300 # numbel of samples, K samples is used for initial centers
    CLUSTER_STD = [0.8, 0.8, 0.5, 0.5, 0.8] # std of each cluster
    M = 1.5
    '''
    Initial sample sites
    '''
    Object_sites = []
    sample_sites_locations = []
    temp_sample_sites_locations, cluster_id = make_blobs(n_samples=N_SAMPLES, n_features=2, centers = CENTERS, cluster_std=CLUSTER_STD, random_state =9)
    x_sample_location = temp_sample_sites_locations[:,0]
    x_sample_location = x_sample_location.tolist()
    y_sample_location = temp_sample_sites_locations[:,1]
    y_sample_location = y_sample_location.tolist()

    for n_sample in range(N_SAMPLES):
        # get sites locations [[x,y],[]...],then initial sites objects
        sample_sites_locations.append([x_sample_location[n_sample],y_sample_location[n_sample]])
        Object_sites.append( Site(n_sample,sample_sites_locations[n_sample][0],sample_sites_locations[n_sample][1]) )

    '''
    Init nu matrix and centers 
    '''
    #############################################################################
    Object_centers = []
    initial_centers_locations= []
    x_center_location = []
    y_center_location = []

    # init nu matrix
    nu_matrix = np.zeros((N_SAMPLES,K))
    for i in range(N_SAMPLES):
        for j in range(K):
            nu_matrix[i][j] = np.random.uniform(0,1)
    row_total = []
    for i in range(N_SAMPLES):
        row_total.append(sum(nu_matrix[i][:]))
    for i in range(N_SAMPLES):
        for j in range(K):
            nu_matrix[i][j] = nu_matrix[i][j] / row_total[i]

    # init centers
    for k in range(K):
        Object_centers.append( Center(k,0,0) )
    Object_centers = cal_center(Object_sites,Object_centers,nu_matrix,M)

    for k in range(K):
        x_center_location.append(Object_centers[k].x_location)
        y_center_location.append(Object_centers[k].y_location)
        initial_centers_locations.append([x_center_location[k],y_center_location[k]])
    
    # show the initial situation
    plt.scatter(x_sample_location,y_sample_location, marker='o') # the sites before k-means
    plt.scatter(x_center_location,y_center_location,s = 300,marker='*',c = 'red')
    plt.title('Init by '+ algorithm_kind)
    plt.savefig('CMeans_inital_centers_.png')

    '''
    C-Means and plt.show
    '''
    plt.ion()
    [Object_centers,n_fig] = c_means(Object_sites, Object_centers,algorithm_kind,M)
    plt.ioff()
    plt.show()
