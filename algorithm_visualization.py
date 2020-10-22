#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 22:16:17 2020

@author: apple
"""


import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches

# class Node:
#     def __init__(self):
#         pass
#     def show_node(self):
#         pass
# if if __name__ == "__main__":
#     pass

fig,ax = plt.subplots()
xy_circle = [0.1,0.1]
radius_circle = 0.05
circle = mpatches.Circle(xy_circle,radius_circle,facecolor = 'white',edgecolor = 'black')
ax.add_artist(circle)
ax.text(xy_circle[0],xy_circle[1],'1',
        horizontalalignment = 'center',
        verticalalignment = 'center',
        transform = ax.transAxes
        )
ax.axis('off')
ax.set_aspect('equal')
# plt.show()

xy_circle = [0.5,0.5]
radius_circle = 0.05
circle = mpatches.Circle(xy_circle,radius_circle,facecolor = 'white',edgecolor = 'black')
ax.add_artist(circle)
ax.text(xy_circle[0],xy_circle[1],'2',
        horizontalalignment = 'center',
        verticalalignment = 'center',
        transform = ax.transAxes
        )
plt.show()
