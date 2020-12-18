
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
from matplotlib import animation
from PIL import Image
import matplotlib.patches as mpatches
from math import pi
from numpy import cos, sin

def cal_global_center(sites):
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

def cal_nextcenter(density,distance_matrix,centers,pre_siteid_of_center):
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

def cal_distance(sites):
    distance_list = []
    n_sites = len(sites)
    distance_matrix = np.zeros((n_sites,n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            distance_matrix[i][j] = ( ((sites[i].x_location - sites[j].x_location) **2)+((sites[i].y_location - sites[j].y_location) **2) ) **0.5
            distance_matrix[j][i] = distance_matrix[i][j]
            distance_list.append(distance_matrix[i][j])
    return distance_matrix,distance_list

def cal_density(sites,distance_matrix,radius):
    density = []
    n_sites = len(sites)
    for i in range(n_sites):
        cnt_site = 0
        for j in range(n_sites):
            if distance_matrix[i][j] < radius:
                cnt_site = cnt_site + 1
        density.append(cnt_site)
    return density


def assign_center(centers,site):
    min_distance = [( ((site.x_location-centers[0].x_location) ** 2) + ((site.y_location-centers[0].y_location)**2) ) ** 0.5,0]
    for i in range(len(centers)):
        distance = ( ((site.x_location-centers[i].x_location) ** 2) + ((site.y_location-centers[i].y_location) **2) ) ** 0.5
        if distance < min_distance[0]:
            min_distance[0] = distance
            min_distance[1] = centers[i].id
    return centers[min_distance[1]]

def cal_center(id,sites):
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
            center = assign_center(centers, site)
            site.center = center.id
            center.sites.append(site)
        # recalculate center
        for center in centers:
            new_center = cal_center(center.id, center.sites)
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
            plt.scatter(x_sample_location,y_sample_location,marker='o',c = color_squence[j%7])

        x_center_location = []
        y_center_location = []
        for i in range(len(centers)):
            x_center_location.append(centers[i].x_location)
            y_center_location.append(centers[i].y_location)    
        plt.scatter(x_center_location,y_center_location,s=400,marker='*',c='red')
        plt.title('Init by '+ algorithm_kind)
        plt.xlabel('Number of iterations:' + str(n_fig+1))
        plt.savefig(str(algorithm_kind)+ '_' + str(n_fig)+'.png')
        n_fig = n_fig+1
        plt.pause(0.5)

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
    CENTERS=[[-1,1], [-1,-2], [1,1], [1,-1],[2,2]]
    # CENTERS=[[-1,-1], [-1,1]]
    K = len(CENTERS)
    N_SAMPLES = 300 # numbel of samples, K samples is used for initial centers
    CLUSTER_STD = [0.8, 0.8, 0.5, 0.5, 0.8] # std of each cluster
    # CLUSTER_STD = [0.2, 0.2]


    '''
    Initial sample sites
    '''

    Object_sites = []
    sample_sites_locations = []
    temp_sample_sites_locations, cluster_id = make_blobs(n_samples=N_SAMPLES, n_features=2, centers = CENTERS, cluster_std=CLUSTER_STD, random_state =9)
    print(type(temp_sample_sites_locations))
    print(temp_sample_sites_locations)
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
    #############################################################################
    # Random
    Object_centers = []
    initial_centers_locations= []
    x_center_location = []
    y_center_location = []

    # initial centers randomly
    algorithm_kind = 'Random Initialization '
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
    plt.savefig('inital_centers_Random.png')

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
    im = Image.open(str(algorithm_kind) + "_0.png")
    images=[]
    for i in range(n_fig):
        if i!=0:
            fpath = str(algorithm_kind) + '_' + str(i) + ".png"
            images.append(Image.open(fpath))
    im.save(str(algorithm_kind) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)


    #############################################################################
    # density center
    Object_centers = []
    initial_centers_locations= []
    x_center_location = []
    y_center_location = []

    # iniital centers using density
    algorithm_kind = 'DBCIM'
    [distance_matrix,distance_list] = cal_distance(Object_sites)
    radius = 0.5 * max(distance_list) / K
    density = cal_density(Object_sites,distance_matrix,radius)
    mindoc = sum(density) / len(density)

    for i in range(len(density)):
        if density[i] < mindoc:
            density[i] = 0
    pre_siteid_of_center = []
    pre_siteid_of_center.append(density.index(max(density)))
    x_center_location.append(x_sample_location[pre_siteid_of_center[0]])
    y_center_location.append(y_sample_location[pre_siteid_of_center[0]])
    initial_centers_locations.append([x_center_location[0],y_center_location[0]])
    Object_centers.append( Center(0,x_center_location[0],y_center_location[0]))
    # plot the initial centers and sites
    angles_circle = [i * pi / 180 for i in range(0, 360)]  # i change to double
    x = cos(angles_circle)
    y = sin(angles_circle)
    fig = plt.figure(figsize=(5,5))
    plt.scatter(x_sample_location,y_sample_location, marker='o') # the sites before k-means
    plt.scatter(x_center_location,y_center_location,s = 300,marker='*',c = 'red')
    plt.plot(x+x_center_location[0],y+y_center_location[0],'r')

    for i in range(K-1):
        pre_siteid_of_center.append(cal_nextcenter(density,distance_matrix,Object_centers,pre_siteid_of_center))
        x_center_location.append(x_sample_location[pre_siteid_of_center[i+1]])
        y_center_location.append(y_sample_location[pre_siteid_of_center[i+1]])
        initial_centers_locations.append([x_center_location[i+1],y_center_location[i+1]])
        Object_centers.append( Center(i+1,x_center_location[i+1],y_center_location[i+1]))

        # # plot the initial centers and sites
        plt.scatter(x_sample_location,y_sample_location, marker='o') # the sites before k-means
        plt.scatter(x_center_location,y_center_location,s = 300,marker='*',c = 'red')
        plt.plot(x+x_center_location[i+1],y+y_center_location[i+1],'r')
          
    plt.title('Init by '+ algorithm_kind)
    plt.savefig('inital_centers_DBCIM.png')

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
    im = Image.open(str(algorithm_kind) + "_0.png")
    images=[]
    for i in range(n_fig):
        if i!=0:
            fpath = str(algorithm_kind) + '_' + str(i) + ".png"
            images.append(Image.open(fpath))
    im.save(str(algorithm_kind) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)
    

    #############################################################################
    # part 
    algorithm_kind = 'Equally Divided Partition'

    [x_global_center,y_global_center] = cal_global_center(Object_sites)
    [distance_matrix,distance_list] = cal_distance(Object_sites)
    max_x_location = max(x_sample_location)
    min_x_location = min(x_sample_location)
    max_distance = max_x_location-min_x_location
    divide_distance = max_distance/K
    divide_location = []
    for i in range(K+1):
        divide_location.append(min_x_location + i * divide_distance)

    Object_centers = []
    initial_centers_locations= []
    x_center_location = []
    y_center_location = []

    for k in range(K):
        x_center_location.append( (divide_location[k] + divide_location[k+1])/2 )
        y_center_location.append( y_global_center )
        initial_centers_locations.append([x_center_location[k],y_center_location[k]])
        Object_centers.append( Center(k,initial_centers_locations[k][0],initial_centers_locations[k][1]) )

    fig = plt.figure(figsize=(5,5))
    plt.scatter(x_sample_location,y_sample_location, marker='o') # the sites before k-means
    plt.scatter(x_center_location,y_center_location,s = 300,marker='*',c = 'red')
    plt.title('Init by '+ algorithm_kind)
    plt.savefig('inital_Partition.png')

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
    im = Image.open(str(algorithm_kind) + "_0.png")
    images=[]
    for i in range(n_fig):
        if i!=0:
            fpath = str(algorithm_kind) + '_' + str(i) + ".png"
            images.append(Image.open(fpath))
    im.save(str(algorithm_kind) + '.gif', save_all=True, append_images=images,loop=1000,duration=500)
