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

def kmeans_cal_global_center(sites):
    sum_x = 0
    sum_y = 0
    cnt_sites = 0
    for i in range(len(sites)):
            cnt_sites = cnt_sites+1
            sum_x = sum_x + sites[i].x_location
            sum_y = sum_y + sites[i].y_location
    x_location = sum_x/cnt_sites
    y_location = sum_y/cnt_sites
    return x_location,y_location

def kmeans_cal_nextcenter(density,distance_matrix,centers,pre_siteid_of_center):
    distance_all_sites = []
    n_sites = len(density)
    n_centers = len(centers)
    for i in range(n_sites):
        total_distance = 1
        for j in range(n_centers):
            total_distance = total_distance * distance_matrix[i][pre_siteid_of_center[j]]
        distance_all_sites.append(total_distance)
        if density[i] == 0:
            distance_all_sites[i] = 0
    next_center_id = distance_all_sites.index(max(distance_all_sites))
    return next_center_id


def kmeans_assign_center(centers,site):
    min_distance = [( ((site.x_location-centers[0].x_location) ** 2) + ((site.y_location-centers[0].y_location)**2) ) ** 0.5,0]
    for i in range(len(centers)):
        distance = ( ((site.x_location-centers[i].x_location) ** 2) + ((site.y_location-centers[i].y_location) **2) ) ** 0.5
        if distance < min_distance[0]:
            min_distance[0] = distance
            min_distance[1] = centers[i].id
    return centers[min_distance[1]]

def kmeans_cal_center(id,sites):
    center = Center(id,0,0)
    sum_x = 0
    sum_y = 0
    cnt_sites = 0
    for i in range(len(sites)):
        if sites[i].center == id:
            cnt_sites = cnt_sites+1
            sum_x = sum_x + sites[i].x_location
            sum_y = sum_y + sites[i].y_location
    center.x_location = sum_x/cnt_sites
    center.y_location = sum_y/cnt_sites
    return center

def k_means(sites, init_centers,algorithm_kind):
    centers = init_centers[:]
    n_fig = 0
    while True:
        new_centers = []
        changed_centers = []
        # assign the center to the site
        for site in sites:
            center = kmeans_assign_center(centers, site)
            site.center = center.id
            center.sites.append(site)
        # recalculate center
        for center in centers:
            new_center = kmeans_cal_center(center.id, center.sites)
            new_centers.append(new_center)
            # if ((new_center.x_location != center.x_location) or (new_center.y_location != center.y_location)):
            if (((new_center.x_location-center.x_location)>0.01) or ((new_center.y_location-center.y_location)>0.01)):
                changed_centers.append(new_center)

        if len(changed_centers) == 0:
            return centers,n_fig
        centers = new_centers[:]

        plt.clf()
        color_squence = ['darkorchid','limegreen','sandybrown','lightslategrey','rosybrown','sienna','seagreen']
        for j in range(len(centers)):
            x_sample_location = []
            y_sample_location = []
            for i in range(len(sites)):
                if sites[i].center == j:
                    x_sample_location.append(sites[i].x_location)
                    y_sample_location.append(sites[i].y_location)  
            plt.scatter(x_sample_location,y_sample_location,marker='o', c = 'white',edgecolors = color_squence[j%7])

        x_center_location = []
        y_center_location = []
        for i in range(len(centers)):
            x_center_location.append(centers[i].x_location)
            y_center_location.append(centers[i].y_location)    
        plt.scatter(x_center_location,y_center_location,s=400,marker='*',c='red')
        plt.title(algorithm_kind)
        plt.xlabel('Number of iterations:' + str(n_fig+1))
        plt.savefig(str(algorithm_kind)+ '_' + str(n_fig+1)+'.png')
        n_fig = n_fig+1
        plt.pause(0.01)

class Site:
    center = 0
    # density = 0
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
    CENTERS=[[-1.15,-1.15], [-1.15,1.15], [2.5,2.5], [1.15,-1.15],[1.15,1.15]]
    K = len(CENTERS)
    N_SAMPLES = 2000 # numbel of samples, K samples is used for initial centers
    CLUSTER_STD = [0.6, 0.6, 0.05, 0.6, 0.6] # std of each cluster

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

    ###############################################################################################################
    # K-Means
    '''
    Inital centers
    '''
    # Random
    Object_centers = []
    initial_centers_locations= []
    x_center_location = []
    y_center_location = []

    # initial centers randomly
    algorithm_kind = 'K-Means '
    rand_squence = np.random.randint(0,N_SAMPLES,K) # choose sites randomly
    for k in range(K):
        # choose sites as centers from samples
        x_center_location.append(x_sample_location[rand_squence[k]])
        y_center_location.append(y_sample_location[rand_squence[k]])
        # pop sites which is chosed as the centers
        x_sample_location.pop(rand_squence[k])
        y_sample_location.pop(rand_squence[k])
        sample_sites_locations.pop(rand_squence[k])
        Object_sites.pop(rand_squence[k])
        # get centers locations [[x,y],[]...],then initial centers objects
        initial_centers_locations.append([x_center_location[k],y_center_location[k]])
        Object_centers.append( Center(k,initial_centers_locations[k][0],initial_centers_locations[k][1]) )

    fig = plt.figure(figsize=(5,5))
    plt.scatter(x_sample_location,y_sample_location, marker='o') # the sites before k-means
    plt.scatter(x_center_location,y_center_location,s = 300,marker='*',c = 'red')
    plt.title('Init by'+ algorithm_kind)
    plt.savefig('KMeans_inital_centers_.png')

    '''
    K-Means and plt.show
    '''
    plt.ion()
    [Object_centers,n_fig] = k_means(Object_sites, Object_centers,algorithm_kind)
    plt.ioff()
    plt.show()

    '''
    Save figs as gif
    '''
    im = Image.open(str(algorithm_kind) + "_1.png")
    images=[]
    for i in range(n_fig+1):
        if i!=0:
            fpath = str(algorithm_kind) + '_' + str(i) + ".png"
            images.append(Image.open(fpath))
    im.save(str(algorithm_kind) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)

