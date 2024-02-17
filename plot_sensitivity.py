"""
Created on Sat Jan 13 15:43:54 2024

@author: jaker
"""

import pickle
import copy
import matplotlib as mpl
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def plotAllSensitivity():

    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Routing_Sensitivity/0/Results_Demand1_AdditionalToll_crash.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [A_stats, A_t, A_depart, A_arrive] = pickle.load(f)
        
    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Routing_Sensitivity/0.1/Results_Demand1_AdditionalToll_crash.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [B_stats, B_t, B_depart, B_arrive] = pickle.load(f)
        
    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Routing_Sensitivity/0.2/Results_Demand1_AdditionalToll_crash.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [C_stats, C_t, C_depart, C_arrive] = pickle.load(f)
    
    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Routing_Sensitivity/0.3/Results_Demand1_AdditionalToll_crash.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [D_stats, D_t, D_depart, D_arrive] = pickle.load(f)
    
    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Routing_Sensitivity/0.4/Results_Demand1_AdditionalToll_crash.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [E_stats, E_t, E_depart, E_arrive] = pickle.load(f)
    
    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Routing_Sensitivity/0.5/Results_Demand1_AdditionalToll_crash.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [F_stats, F_t, F_depart, F_arrive] = pickle.load(f)
    

    fig, ax = plt.subplots()
    #ax.plot(so_base_t, (np.array(so_base_mean_depart)/100000).tolist())
    #ax.plot(so_mon_t, (np.array(so_mon_mean_depart)/100000).tolist())
    #ax.plot(ue_mon_t, (np.array(ue_mon_mean_depart)/100000).tolist())
    ax.plot(A_t, (np.array(A_depart)/1000000).tolist())
    
    ax.plot(A_t, (np.array(A_arrive)/1000000).tolist())
    ax.plot(B_t, (np.array(B_arrive)/1000000).tolist())
    ax.plot(C_t, (np.array(C_arrive)/1000000).tolist())
    ax.plot(D_t, (np.array(D_arrive)/1000000).tolist())
    ax.plot(E_t, (np.array(E_arrive)/1000000).tolist())
    ax.plot(F_t, (np.array(F_arrive)/1000000).tolist())
    
    plt.legend(["Departures", "0", "0.2", "0.4", "0.6", "0.8", "1"], loc ="lower right", fontsize = 12)

    plt.xlabel('Time (hrs)', fontweight ='bold', fontsize = 15)
    plt.xticks(fontsize = 12)
    ax.yaxis.offsetText.set_fontsize(15)
    plt.ylabel('Million People', fontweight ='bold', fontsize = 15)
    plt.yticks(fontsize = 12)
    ax.set_ylim(0, 2.5)
    ax.set_xlim(0, 120)
    sns.set_style("whitegrid")
        
    plt.savefig("Out/20240116/Plots/AllSensitivity.png", dpi=1400, bbox_inches="tight")
    plt.show()
  
    

def plotSensitivityTrends():
    Crash_ue_x =[]
    Crash_ue_y = []
    
    for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        name = '//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Crash_Sensitivity/' + str(i) + '/Results_Demand1_AdditionalToll_ue.txt'
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            [stats, t, depart, arrive] = pickle.load(f)
    
        Crash_ue_x.append(1-i)
        Crash_ue_y.append(stats["last arrival mean"])
        
    Crash_so_x =[]
    Crash_so_y = []
    
    for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        name = '//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Crash_Sensitivity/' + str(i) + '/Results_Demand1_AdditionalToll_so.txt'
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            [stats, t, depart, arrive] = pickle.load(f)
    
        Crash_so_x.append(1-i)
        Crash_so_y.append(stats["last arrival mean"])
        
    Crash_routing_x =[]
    Crash_routing_y = []
    
    for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        name = '//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Crash_Sensitivity/' + str(i) + '/Results_Demand1_AdditionalToll_routing.txt'
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            [stats, t, depart, arrive] = pickle.load(f)
    
        Crash_routing_x.append(1-i)
        Crash_routing_y.append(stats["last arrival mean"])
        


    Routing_base_x =[]
    Routing_base_y = []
    
    for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        name = '//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Routing_Sensitivity/' + str(i) + '/Results_Demand1_AdditionalToll_base.txt'
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            [stats, t, depart, arrive] = pickle.load(f)
    
        Routing_base_x.append(i)
        Routing_base_y.append(stats["last arrival mean"])
        
    Routing_mon_x =[]
    Routing_mon_y = []
    
    for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        name = '//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Routing_Sensitivity/' + str(i) + '/Results_Demand1_AdditionalToll_mon.txt'
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            [stats, t, depart, arrive] = pickle.load(f)
    
        Routing_mon_x.append(i)
        Routing_mon_y.append(stats["last arrival mean"])

    Routing_crash_x =[]
    Routing_crash_y = []
    
    for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        name = '//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240116/Routing_Sensitivity/' + str(i) + '/Results_Demand1_AdditionalToll_crash.txt'
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            [stats, t, depart, arrive] = pickle.load(f)
    
        Routing_crash_x.append(i)
        Routing_crash_y.append(stats["last arrival mean"])
    
    fig, ax = plt.subplots()
    #ax.plot(so_base_t, (np.array(so_base_mean_depart)/100000).tolist())
    #ax.plot(so_mon_t, (np.array(so_mon_mean_depart)/100000).tolist())
    #ax.plot(ue_mon_t, (np.array(ue_mon_mean_depart)/100000).tolist())
    ax.plot(Crash_ue_x, Crash_ue_y)
    ax.plot(Crash_routing_x, Crash_routing_y)
    ax.plot(Crash_so_x, Crash_so_y)
    plt.legend(["UE (No TMDs)", "0.5 (Predicted TMD Influence)", "ESO (Ideal)"], loc ="lower left", fontsize = 12)
    
    plt.xlabel('Impact of TMD on Crashes', fontweight ='bold', fontsize = 15)
    plt.xticks(fontsize = 12)
    ax.yaxis.offsetText.set_fontsize(15)
    plt.ylabel('Time (hrs)', fontweight ='bold', fontsize = 15)
    plt.yticks(fontsize = 12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 140)
    sns.set_style("whitegrid")
        
    plt.savefig("Out/20240116/Plots/CrashesSensitivity.png", dpi=1400, bbox_inches="tight")
    plt.show()
    
    
    fig, ax = plt.subplots()
    #ax.plot(so_base_t, (np.array(so_base_mean_depart)/100000).tolist())
    #ax.plot(so_mon_t, (np.array(so_mon_mean_depart)/100000).tolist())
    #ax.plot(ue_mon_t, (np.array(ue_mon_mean_depart)/100000).tolist())
    ax.plot(Routing_base_x, Routing_base_y)
    ax.plot(Routing_crash_x, Routing_crash_y)
    ax.plot(Routing_mon_x, Routing_mon_y)
    plt.legend(["Base Crashes (No TMDs)", "0.2 (Predicted TMD Influence)", "No Crashes (Ideal)"], loc ="lower left", fontsize = 12)
    
    plt.xlabel('Impact of TMD on Routing', fontweight ='bold', fontsize = 15)
    plt.xticks(fontsize = 12)
    ax.yaxis.offsetText.set_fontsize(15)
    plt.ylabel('Time (hrs)', fontweight ='bold', fontsize = 15)
    plt.yticks(fontsize = 12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 140)
    sns.set_style("whitegrid")
        
    plt.savefig("Out/20240116/Plots/RoutingSensitivity.png", dpi=1400, bbox_inches="tight")
    plt.show()
    
def plotAllCurves():

    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240119/Prediction_Results/AdditionalToll_Demand2/Results_Demand1_AdditionalToll_ue_base.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [A_stats, A_t, A_depart, A_arrive] = pickle.load(f)
        
    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240119/Prediction_Results/AdditionalToll_Demand2/Results_Demand1_AdditionalToll_ue_mon.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [B_stats, B_t, B_depart, B_arrive] = pickle.load(f)
        
    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240119/Prediction_Results/AdditionalToll_Demand2/Results_Demand1_AdditionalToll_so_base.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [C_stats, C_t, C_depart, C_arrive] = pickle.load(f)
    
    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240119/Prediction_Results/AdditionalToll_Demand2/Results_Demand1_AdditionalToll_so_mon.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [D_stats, D_t, D_depart, D_arrive] = pickle.load(f)    

    fig, ax = plt.subplots()
    #ax.plot(so_base_t, (np.array(so_base_mean_depart)/100000).tolist())
    #ax.plot(so_mon_t, (np.array(so_mon_mean_depart)/100000).tolist())
    #ax.plot(ue_mon_t, (np.array(ue_mon_mean_depart)/100000).tolist())
    ax.plot(A_t, (np.array(A_depart)/100000).tolist())
    
    ax.plot(A_t, (np.array(A_arrive)/100000).tolist())
    ax.plot(B_t, (np.array(B_arrive)/100000).tolist())
    ax.plot(C_t, (np.array(C_arrive)/100000).tolist())
    ax.plot(D_t, (np.array(D_arrive)/100000).tolist())
    
    plt.legend(["Departures", "No Monitoring", "Monitoring Incidents", "Monitoring Routing", "Monitoring Both"], loc ="lower right", fontsize = 12)

    plt.xlabel('Time (hrs)', fontweight ='bold', fontsize = 15)
    plt.xticks(fontsize = 12)
    ax.yaxis.offsetText.set_fontsize(15)
    plt.ylabel('100 Thousand People', fontweight ='bold', fontsize = 15)
    plt.yticks(fontsize = 12)
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, 80)
    sns.set_style("whitegrid")
        
    plt.savefig("Out/20240116/Plots/AllCurves.png", dpi=1400, bbox_inches="tight")
    plt.show()
    
def Validation():
    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240119/NO_Heuristic/Results_Demand1_AdditionalToll_ue_base.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [Ideal_stats, Ideal_t, Ideal_depart, Ideal_arrive] = pickle.load(f)
        
    with open('//wsl.localhost/Ubuntu/home/jr73453/capacity_full/Out/20240119/Prediction_Results/AdditionalToll_Demand1/Results_Demand1_AdditionalToll_ue_base.txt', 'rb') as f:  # Python 3: open(..., 'rb')
        [Heuristic_stats, Heuristic_t, Heuristic_depart, Heuristic_arrive] = pickle.load(f)
    

    fig, ax = plt.subplots()
    #ax.plot(so_base_t, (np.array(so_base_mean_depart)/100000).tolist())
    #ax.plot(so_mon_t, (np.array(so_mon_mean_depart)/100000).tolist())
    #ax.plot(ue_mon_t, (np.array(ue_mon_mean_depart)/100000).tolist())
    ax.plot(Heuristic_t, (np.array(Heuristic_depart)/1000000).tolist())
    
    ax.plot(Ideal_t, (np.array(Ideal_arrive)/1000000).tolist())
    ax.plot(Heuristic_t, (np.array(Heuristic_arrive)/1000000).tolist())
    
    plt.legend(["Departures", "Ideal", "Heuristic"], loc ="lower right", fontsize = 12)

    plt.xlabel('Time (hrs)', fontweight ='bold', fontsize = 15)
    plt.xticks(fontsize = 12)
    ax.yaxis.offsetText.set_fontsize(15)
    plt.ylabel('Million People', fontweight ='bold', fontsize = 15)
    plt.yticks(fontsize = 12)
    ax.set_ylim(0, 2.5)
    ax.set_xlim(0, 140)
    sns.set_style("whitegrid")
        
    plt.savefig("Out/20240116/Plots/Validation.png", dpi=1400, bbox_inches="tight")
    plt.show()
    
#plot_all_curves(Ideal_t, Ideal_depart, Ideal_arrive, Heuristic_t, Heuristic_depart, Heuristic_arrive)


#plotAllSensitivity()
#plotSensitivityTrends()
plotAllCurves()
#Validation()