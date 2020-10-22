#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:10:51 2020

@author: apple
"""

import random
import numpy as np

if __name__ == "__main__":
    TTstar = []
    def loopoftest():
        num_machine = 2
        print('number of machine is:',num_machine)
        num_job = 5
        print('number of job is:',num_job)
        weight_machine = [1,2]
        print('weight of each machine is:',weight_machine)
        tw_machine = []
        temp_tw_machine = []
        for i in range(num_machine):
            tw_machine.append(0)
            temp_tw_machine.append(0)
        print('time of each Machine:',tw_machine)
        time_job = []
        sorted_time_job = []
        for i in range(num_job):
            time_job.append(random.randint(1,5))
            sorted_time_job.append(0)
        print('time of each Job is:',time_job)

        # weight_machine = np.mat(weight_machine)
        # time_job = np.mat(time_job)
        # tw_job = weight_machine.T*time_job #2x5
        # tw_job = tw_job.reshape(1,num_job*num_machine)
        # tw_job = tw_job.tolist()
        # tw_job = tw_job[0]
        # print('Tw is',tw_job)

        # sort the time of job, and pop the max one, then find out the min tw machine, make that machine tw update
        time_job.sort(reverse = False)
        sorted_time_job = time_job.copy()
        print('sorted time of job:',sorted_time_job)
        
        # weight-influenced sorted greedy algorithm
        while(sorted_time_job):
            maxtime_job = sorted_time_job.pop()
            print('\n#####################')
            print('pop max time job:',maxtime_job)
            maxtime_after_weight = [i * maxtime_job for i in weight_machine]
            print('max time after weight:',maxtime_after_weight)
            for i in range(num_machine):
                temp_tw_machine[i] = tw_machine[i] + maxtime_after_weight[i]
            id_mintw_machine = temp_tw_machine.index(min(temp_tw_machine))
            print('id of sutable machine:',id_mintw_machine)
            tw_machine[id_mintw_machine] = tw_machine[id_mintw_machine] + maxtime_after_weight[id_mintw_machine]
            print('now tw:',tw_machine)
        Tw_WISG = max(tw_machine)
        print('\n#####################')
        print('Tw of WISG is:',Tw_WISG)

        # brute force algorithm
        temp_tw_bf = []
        sorted_time_job = time_job.copy()
        print('\n')
        for i in range(int('11111',2)):
            tw_machine[0] = 0
            tw_machine[1] = 0
            total = 0
            bin_i = '{:05b}'.format(i)
            for j in range(5):
                if(bin_i[j] == '1'):
                    tw_machine[0] += sorted_time_job[j]
                else:
                    pass
            for ele in range(len(sorted_time_job)):
                total += sorted_time_job[ele]
            tw_machine[1] = weight_machine[1] * (total-tw_machine[0]) 
            temp_tw_bf.append(max(tw_machine))
        Tw_BF = min(temp_tw_bf)
        id_tw_bf = temp_tw_bf.index(Tw_BF)
        id_tw_bf = '{:05b}'.format(id_tw_bf)
        print('Tw of BF',Tw_BF,id_tw_bf)
        TTstar.append(Tw_WISG/Tw_BF)

    total_TTstar = 0
    for i in range(1000):
        loopoftest()
    for i in range(1000):
        total_TTstar += TTstar[i]
    print('\nThrough 1000 tests,T/Tstar:',total_TTstar/len(TTstar))







    

        
