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

################################################################################################################################
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

################################################################################################################################
def cmeans_cal_center(sites,initial_centers,nu_matrix,m):
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

def cmeans_cal_distance(sites,centers):
    n_sites = len(sites)
    n_centers = len(centers)
    distance_matrix = np.zeros((n_sites,n_centers))
    for i in range(n_sites):
        for j in range(n_centers):
            distance_matrix[i][j] = ( ((sites[i].x_location - centers[j].x_location) **2)+((sites[i].y_location - centers[j].y_location) **2) ) **0.5
    return distance_matrix

def cmeans_assign_center(site,centers,distance_matrix):
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
        distance_matrix = cmeans_cal_distance(sites,centers)
        changed_centers = []
        # assign the center to the site
        for site in sites:
            center = cmeans_assign_center(site, centers, distance_matrix)
            site.center = center.id
            center.sites.append(site)
        # recalculate center
        nu_matrix = cal_numatrix(distance_matrix,centers,m,len(sites),len(centers))
        new_centers = copy.deepcopy(cmeans_cal_center(sites,centers,nu_matrix,m))

        for i in range(len(centers)):
            if ((abs(centers[i].x_location-new_centers[i].x_location)>0.01) or (abs(centers[i].y_location-new_centers[i].y_location)>0.01)):
                changed_centers.append(centers[i])
        if len(changed_centers) == 0:
            return centers,n_fig
        centers = copy.deepcopy(new_centers) 

        plt.clf()
        color_squence = ['darkorchid','limegreen','sandybrown','lightslategrey','rosybrown','sienna','seagreen']
        for j in range(len(centers)):
            x_sample_location = []
            y_sample_location = []
            for i in range(len(sites)):
                x_sample_location.append(sites[i].x_location)
                y_sample_location.append(sites[i].y_location)  
                plt.scatter(sites[i].x_location,sites[i].y_location,marker='o',c = 'white',alpha = nu_matrix[i][j],edgecolors=color_squence[j%7])
    
        x_center_location = []
        y_center_location = []
        for i in range(len(centers)):
            x_center_location.append(centers[i].x_location)
            y_center_location.append(centers[i].y_location)    
        plt.scatter(x_center_location,y_center_location,s=400,marker='*',c='red')
        plt.title(algorithm_kind + ' M=' + str(m))
        plt.xlabel('Number of iterations:' + str(n_fig+1))
        plt.savefig(str(algorithm_kind)+ '_' +'M=' + str(m) + '_' + str(n_fig+1)+'.png')
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
    CENTERS=[[-1.15,-1.15], [-1.15,1.15], [2.5,2.5], [1.15,-1.15],[1.15,1.15]]
    K = len(CENTERS)
    N_SAMPLES = 600 # numbel of samples, K samples is used for initial centers
    CLUSTER_STD = [0.6, 0.6, 0.05, 0.6, 0.6] # std of each cluster
    
     ###############################################################################################################
    # K-Means
    algorithm_kind = 'K-Means'
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
    plt.title('Init by '+ algorithm_kind)
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


    ###########################################################################################################################
    algorithm_kind = 'C-Means' 
    M = 1.1
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
    Object_centers = cmeans_cal_center(Object_sites,Object_centers,nu_matrix,M)

    for k in range(K):
        x_center_location.append(Object_centers[k].x_location)
        y_center_location.append(Object_centers[k].y_location)
        initial_centers_locations.append([x_center_location[k],y_center_location[k]])
    
    # show the initial situation
    fig = plt.figure(figsize=(5,5))
    plt.scatter(x_sample_location,y_sample_location, marker='o',edgecolor = 'green',c = 'white') # the sites before k-means
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

    '''
    Save figs as gif
    '''
    im = Image.open(str(algorithm_kind)+ '_' +'M=' + str(M) + "_1.png")
    images=[]
    for i in range(n_fig+1):
        if i>1:
            fpath = str(algorithm_kind)+ '_' +'M=' + str(M) + '_' + str(i)+ ".png"
            images.append(Image.open(fpath))
    im.save(str(algorithm_kind)+ '_' +'M=' + str(M) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)

    ###########################################################################################################################
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
    Object_centers = cmeans_cal_center(Object_sites,Object_centers,nu_matrix,M)

    for k in range(K):
        x_center_location.append(Object_centers[k].x_location)
        y_center_location.append(Object_centers[k].y_location)
        initial_centers_locations.append([x_center_location[k],y_center_location[k]])
    
    # show the initial situation
    fig = plt.figure(figsize=(5,5))
    plt.scatter(x_sample_location,y_sample_location, marker='o',edgecolor = 'green',c = 'white') # the sites before k-means
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

    '''
    Save figs as gif
    '''
    im = Image.open(str(algorithm_kind)+ '_' +'M=' + str(M) + "_1.png")
    images=[]
    for i in range(n_fig+1):
        if i>1:
            fpath = str(algorithm_kind)+ '_' +'M=' + str(M) + '_' + str(i)+ ".png"
            images.append(Image.open(fpath))
    im.save(str(algorithm_kind)+ '_' +'M=' + str(M) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)

    ###########################################################################################################################
    M = 2
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
    Object_centers = cmeans_cal_center(Object_sites,Object_centers,nu_matrix,M)

    for k in range(K):
        x_center_location.append(Object_centers[k].x_location)
        y_center_location.append(Object_centers[k].y_location)
        initial_centers_locations.append([x_center_location[k],y_center_location[k]])
    
    # show the initial situation
    fig = plt.figure(figsize=(5,5))
    plt.scatter(x_sample_location,y_sample_location, marker='o',edgecolor = 'green',c = 'white') # the sites before k-means
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

    '''
    Save figs as gif
    '''
    im = Image.open(str(algorithm_kind)+ '_' +'M=' + str(M) + "_1.png")
    images=[]
    for i in range(n_fig+1):
        if i>1:
            fpath = str(algorithm_kind)+ '_' +'M=' + str(M) + '_' + str(i)+ ".png"
            images.append(Image.open(fpath))
    im.save(str(algorithm_kind)+ '_' +'M=' + str(M) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)

    ###########################################################################################################################
    M = 3
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
    Object_centers = cmeans_cal_center(Object_sites,Object_centers,nu_matrix,M)

    for k in range(K):
        x_center_location.append(Object_centers[k].x_location)
        y_center_location.append(Object_centers[k].y_location)
        initial_centers_locations.append([x_center_location[k],y_center_location[k]])
    
    # show the initial situation
    fig = plt.figure(figsize=(5,5))
    plt.scatter(x_sample_location,y_sample_location, marker='o',edgecolor = 'green',c = 'white') # the sites before k-means
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

    '''
    Save figs as gif
    '''
    im = Image.open(str(algorithm_kind)+ '_' +'M=' + str(M) + "_1.png")
    images=[]
    for i in range(n_fig+1):
        if i>1:
            fpath = str(algorithm_kind)+ '_' +'M=' + str(M) + '_' + str(i)+ ".png"
            images.append(Image.open(fpath))
    im.save(str(algorithm_kind)+ '_' +'M=' + str(M) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)
