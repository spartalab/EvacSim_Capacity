# ------ Run max flow with different crash parameters

import utils
import convert_network
import pickle
from collections import defaultdict
import maxFlowIter
import time
import copy
import random
import statistics
import matplotlib as mpl
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import os # to call tap-b


########################Files##########################
#Input
COORDS_FILE = "Net/houston_input_AdditionalToll.nxy"
file_nl = "Net/houston_input_AdditionalToll.net"
file_dl = "Net/houston_input_1.ods"
DEMAND_FILE = utils.processFile("Net/Demand_Scenario_1.txt", "~")

#Temp
NET_FILE = "Net/Temp/houston_net_Demand1_AdditionalToll.txt"
LINKS_FILE = "Net/Temp/houston_links_Demand1_AdditionalToll.txt"
tntp_file = "Net/Temp/.tntp-net.txt"
tntp_params_file = "Net/Temp/.tntp-parameters.txt"
flows_file = "Net/Temp/.flows.txt"
link_data_file = "Net/Temp/.linkdata.txt" # for max-flow

#Output
OUT_FILE = 'Out/Results_Demand1_AdditionalToll.pkl'
critical_plot_root = "Out/critical_links" # suffix is "_{ue/so}.png"
critical_link_root = "Out/critical_links" # suffix is "_{ue/so}.txt"
stats_root = "Out/stats" # suffix is "_{ue/so}_{base/mon}.txt"
curve_plot_root = "Out/evacuation_curve" # suffix is "_{ue/so}_{base/mon}.png"


######## Monte Carlo Parameters ##########################
TICK_SIZE = 15 #15 minutes
Duration_Impact = 0.2   #Between 0 and 1, tells reduction in total incident durration due to crash
DEMAND_MULT = 1
num_samples = 2 # Number of Monte Carlo draws, must be > 1


base_params = { 'Collision Parameters': {
                   'time horizon': 250, 
                   'incident rate':200, #incidents per 100 million vmt
                   'mean duration':0.67,
                   'var duration':16.13,
                   'alpha capacity loss':4.05907,  #% capacity loss taken from ~Beta(alpha,beta)
                   'beta capacity loss':6.83057},
               'Disabled Parameters': {
                   'time horizon': 250, 
                   'incident rate':1000, # disabled
                   'mean duration':0.67,
                   'var duration':16.25,
                   'alpha capacity loss':5.19123,
                   'beta capacity loss':2.22481}}

No_params = { 'Collision Parameters': {
                   'time horizon': 0, 
                   'incident rate':0, #incidents per 100 million vmt
                   'mean duration':0,
                   'var duration':0,
                   'alpha capacity loss':0,  #% capacity loss taken from ~Beta(alpha,beta)
                   'beta capacity loss':0},
               'Disabled Parameters': {
                   'time horizon': 0, 
                   'incident rate':0, # disabled
                   'mean duration':0,
                   'var duration':0,
                   'alpha capacity loss':0,
                   'beta capacity loss':0}}

mon_params = { 'Collision Parameters': {
                   'time horizon': 250, 
                   'incident rate':200,
                   'mean duration':0.67*Duration_Impact,
                   'var duration':16.13*Duration_Impact,
                   'alpha capacity loss':4.05907,
                   'beta capacity loss':6.83057},
               'Disabled Parameters': {
                   'time horizon': 250, 
                   'incident rate':1000, # disabled
                   'mean duration':0.67*Duration_Impact,
                   'var duration':16.25*Duration_Impact,
                   'alpha capacity loss':5.19123,
                   'beta capacity loss':2.22481}}


################ Initialize Simulation ################################################
with open('Net/County_map.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    County_map = pickle.load(f)
    

coordinate_data = utils.processFile(COORDS_FILE)
coordinate = {}
for line in coordinate_data:
    fields = line.split()
    i = int(fields[0])
    x = float(fields[1])
    y = float(fields[2])
    coordinate[i] = (x,y)

Demand = defaultdict(dict)
for line in DEMAND_FILE:
    if line[0] == '<':
        continue
    fields = line.split()
    county = int(fields[0])
    sink = int(fields[1])
    T = int(fields[2])
    d = int(fields[3])
    ID = County_map[county]
    Demand[T][ID] = round(d*DEMAND_MULT)

convert_network.convertNewNet(TICK_SIZE, file_nl, file_dl, NET_FILE, LINKS_FILE)
convert_network.dta_to_tntp(file_nl, file_dl, tntp_file)

#############################################################################################    
            
def Run_sims(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, incident_params,
                  n, UEsim, demand_file, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot, critical_link):
    """
    incident_params = dictionary of parameters to use for Monte Carlo
        incident simulation; if dict is empty, assumes no incidents.
        This dict is passed directly to generate_incidents, see its
        docstring for details on keys
    n = # of Monte Carlo draws to do (if incident_params is empty,
        only do 1 run since the simulation is deterministic)
        is exponential, and capacity loss is normal.   
    """

    stats = []
    net = maxFlowIter.InitializeMaxFlowIter(copy.deepcopy(Demand), NET_FILE, LINKS_FILE, TICK_SIZE, incident_params, UEsim, demand_file, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot, critical_link)
    
    for i in range(n):
        a = time.time()
        print("Running iteration " + str(i+1) + " of " + str(n))
        run_stats = maxFlowIter.maxFlowIterRun(net, copy.deepcopy(Demand), TICK_SIZE, UEsim, demand_file, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot, critical_link)
        stats.append(run_stats)
        print("Time: " + str(time.time() - a))

    ################ COMPUTE STATISTICS ##########################
    final_stats = {}
    final_stats['num samples'] = n
    for key in stats[0]:
        if key == 'full profile': # Profile must be handled differently
            continue
        values = [s[key] for s in stats]
        final_stats[key + ' mean'] = statistics.mean(values)
        final_stats[key + ' stdev'] = statistics.stdev(values)
    last_tick = max(len(s['full profile']) for s in stats)
    profile_stats = []
    for tick in range(last_tick):
        departures = []
        arrivals = []
        queue = []
        for run in range(n):
            if tick >= len(stats[run]['full profile']):
                departures.append(stats[run]['full profile'][-1][1])
                arrivals.append(stats[run]['full profile'][-1][2])
                queue.append(stats[run]['full profile'][-1][3])
            else:
                t = stats[run]['full profile'][tick][0]
                departures.append(stats[run]['full profile'][tick][1])
                arrivals.append(stats[run]['full profile'][tick][2])
                queue.append(stats[run]['full profile'][tick][3])
        tick_stats = {
            't':t,
            'departure mean':statistics.mean(departures),
            'departure stdev':statistics.stdev(departures),
            'arrival mean':statistics.mean(arrivals),
            'arrival stdev':statistics.stdev(arrivals),
            'queue mean':statistics.mean(queue),
            'queue stdev':statistics.stdev(queue)
        }
        profile_stats.append(tick_stats)
    final_stats['full profile'] = profile_stats
    return final_stats

#maxFlowIter.run_max_flow(copy.deepcopy(Demand), NET_FILE, LINKS_FILE, TICK_SIZE, No_params, file_dl, tntp_file, flows_file, #tntp_params_file, link_data_file, coordinate)

#################### RUN SIMULATIONS ###############################
start = time.time()

print("Running SO Base")
so_base_stats = Run_sims(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, base_params, num_samples, 0,  file_dl, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot_root + "_so_base.png", critical_link_root + "_so_base.txt") #UEsim = 0
convert_network.format_stats(so_base_stats, stats_root + "_so_base.txt")
so_base_t, so_base_mean_depart, so_base_mean_arrive = convert_network.plot_curves(so_base_stats, curve_plot_root + "_so_base.png")
so_base_end = time.time()

print("Running SO Mon")
so_mon_stats = Run_sims(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, mon_params, num_samples, 0,  file_dl, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot_root + "_so_mon.png", critical_link_root + "_so_mon.txt") #UEsim = 0
convert_network.format_stats(so_mon_stats, stats_root + "_so_mon.txt")
so_mon_t, so_mon_mean_depart, so_mon_mean_arrive = convert_network.plot_curves(so_mon_stats, curve_plot_root + "_so_mon.png")
so_mon_end = time.time()

print("Running UE Base")
ue_base_stats = Run_sims(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, base_params, num_samples, 1,  file_dl, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot_root + "_ue_base.png", critical_link_root + "_ue_base.txt") #UEsim = 1
convert_network.format_stats(ue_base_stats, stats_root + "_ue_base.txt")
ue_base_t, ue_base_mean_depart, ue_base_mean_arrive = convert_network.plot_curves(ue_base_stats, curve_plot_root + "_ue_base.png")
ue_base_end = time.time()

print("Running UE Mon")
ue_mon_stats = Run_sims(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, mon_params, num_samples, 1,  file_dl, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot_root + "_ue_mon.png", critical_link_root + "_ue_mon.txt") #UEsim = 1
convert_network.format_stats(ue_mon_stats, stats_root + "_ue_mon.txt")
ue_mon_t, ue_mon_mean_depart, ue_mon_mean_arrive = convert_network.plot_curves(ue_mon_stats, curve_plot_root + "_ue_mon.png")
ue_mon_end = time.time()

convert_network.plot_all_curves(so_base_t, so_base_mean_depart, so_base_mean_arrive, so_mon_t, so_mon_mean_depart, so_mon_mean_arrive, ue_base_t, ue_base_mean_depart, ue_base_mean_arrive, ue_mon_t, ue_mon_mean_depart, ue_mon_mean_arrive, curve_plot_root + "All_Curves.png")

with open(OUT_FILE, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([so_base_stats, so_base_t, so_base_mean_depart, so_base_mean_arrive, so_mon_stats, so_mon_t, so_mon_mean_depart, so_mon_mean_arrive, ue_base_stats, ue_base_t, ue_base_mean_depart, ue_base_mean_arrive, ue_mon_stats, ue_mon_t, ue_mon_mean_depart, ue_mon_mean_arrive], f)

print("SO base time: " + str(so_base_end - start))
print("SO mon time: " + str(so_mon_end - so_base_end))
print("UE base time: " + str(ue_base_end - so_mon_end))
print("UE mon time: " + str(ue_mon_end - ue_base_end))
