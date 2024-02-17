# ------ Run max flow with different crash parameters

import utils
import convert_network
import gis_net
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
COORDS_FILE = "NetUpdate/houston_input_AdditionalToll.nxy"
file_nl = "NetUpdate/houston_input_AdditionalToll.net"
file_dl = "NetUpdate/houston_input_2.ods"
DEMAND_FILE = utils.processFile("NetUpdate/Demand_Scenario_2.txt", "~")

#Temp
NET_FILE = "NetUpdate/Temp/houston_net_Demand2_AdditionalToll.txt"
LINKS_FILE = "NetUpdate/Temp/houston_links_Demand2_AdditionalToll.txt"
tntp_file = "NetUpdate/Temp/.tntp-net.txt"
tntp_params_file = "NetUpdate/Temp/.tntp-parameters.txt"
flows_file = "NetUpdate/Temp/.flows.txt"
link_data_file = "NetUpdate/Temp/.linkdata.txt" # for max-flow

# ########################Files##########################
# #INPUT
# demand_file1 = 'GIS/Demand/demand_scenario1.xlsx'
# demand_file2 = 'GIS/Demand/demand_scenario2.xlsx'
# txt_demand_1 = 'GIS/Demand/Demand_Scenario_1.txt'
# txt_demand_2 ='GIS/Demand/Demand_Scenario_2.txt'
# GIS_nodes = 'GIS/GIS_nodes.xlsx'
# GIS_net = 'GIS/GIS_net.xlsx'

# #TEMP
# raw_link_file = 'GIS/Temp/Link1209_Base.csv' 
# node_file = 'GIS/Temp/Nodes1209_Base.csv'
# county_file = 'GIS/Temp/County_references.csv'
# Counties = 'GIS/Temp/County_map.pkl'        
# safe_node_file = 'GIS/Temp/safe_nodes.csv'
# demand_out1 = 'GIS/Temp/houston_input1.ods'
# demand_out2 = 'GIS/Temp/houston_input2.ods'
# out_nxy = 'GIS/Temp/houston_input.nxy'
# net_out = 'GIS/Temp/houston_input.net'
# links_upd_out = 'GIS/Temp/df_links_upd.csv'
# dup_out = 'GIS/Temp/duplicated_links.csv'
# ID_out = 'GIS/Temp/houston_ID.csv'

# NET_FILE = "GIS/Temp/houston_net.txt"
# LINKS_FILE = "GIS/Temp/houston_links.txt"
# tntp_file = "GIS/Temp/.tntp-net.txt"
# tntp_params_file = "GIS/Temp/.tntp-parameters.txt"
# flows_file = "GIS/Temp/.flows.txt"
# link_data_file = "GIS/Temp/.linkdata.txt" # for max-flow

# #Input to simulation
# COORDS_FILE = out_nxy
# file_nl = net_out
# file_dl = demand_out1  #or demand_out2
# DEMAND_FILE = utils.processFile(txt_demand_1, "~") #or txt_demand_2

######################################################################################
######################################################################################
######################################################################################

def Run_sims(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, incident_params, Routing_Impact, 
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
    net = maxFlowIter.InitializeMaxFlowIter(copy.deepcopy(Demand), NET_FILE, LINKS_FILE, TICK_SIZE, incident_params, Routing_Impact, UEsim, demand_file, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot, critical_link)
    
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



################ Initialize Simulation ################################################
   
#gis_net.BuildNetwork(demand_file1, demand_file2, GIS_nodes, GIS_net, raw_link_file, node_file, county_file, safe_node_file, demand_out1, demand_out2, out_nxy, net_out, links_upd_out, dup_out, ID_out, Counties)


######## Monte Carlo Parameters ##########################
TICK_SIZE = 15 #15 minutes
#Duration_Impact = 0.2   #Between 0 and 1, tells reduction in total incident durration due to crash
DEMAND_MULT = 1
num_samples = 100 # Number of Monte Carlo draws, must be > 1

# with open(Counties, 'rb') as f:  # Python 3: open(..., 'rb')
#     County_map = pickle.load(f)
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
    ID = County_map[county]#[0]
    Demand[T][ID] = round(d*DEMAND_MULT)

convert_network.convertNewNet(TICK_SIZE, file_nl, file_dl, NET_FILE, LINKS_FILE)
convert_network.dta_to_tntp(file_nl, file_dl, tntp_file)


######################################################################################
######################################################################################
######################################################################################
#######BEST ESTIMATE###################

Duration_Impact = 0.8
Routing_Impact = 0.5

#Output
OUT_FILE = 'Out/20240119/Prediction_Results/Results_Demand1_AdditionalToll' # suffix is "_{ue/so}.png"
critical_plot_root = 'Out/20240119/Prediction_Results/critical_links' # suffix is "_{ue/so}.png"
critical_link_root = 'Out/20240119/Prediction_Results/critical_links' # suffix is "_{ue/so}.txt"
stats_root = 'Out/20240119/Prediction_Results/stats' # suffix is "_{ue/so}_{base/mon}.txt"
curve_plot_root = 'Out/20240119/Prediction_Results/evacuation_curve' # suffix is "_{ue/so}_{base/mon}.png"

base_params = { 'Collision Parameters': {
                   'time horizon': 250, 
                   'incident rate':200, #incidents per 100 million vmt
                   'mean duration':40.3562,
                   'var duration':967.94,
                   'alpha capacity loss':4.05907,  #% capacity loss taken from ~Beta(alpha,beta)
                   'beta capacity loss':6.83057},
               'Disabled Parameters': {
                   'time horizon': 250, 
                   'incident rate':1000, # disabled
                   'mean duration':39.749,
                   'var duration':975.23,
                   'alpha capacity loss':5.19123,
                   'beta capacity loss':2.22481}}

mon_params = { 'Collision Parameters': {
                   'time horizon': 250, 
                   'incident rate':200,
                   'mean duration':40.3562*Duration_Impact,
                   'var duration':967.94*Duration_Impact,
                   'alpha capacity loss':4.05907,
                   'beta capacity loss':6.83057},
               'Disabled Parameters': {
                   'time horizon': 250, 
                   'incident rate':1000, # disabled
                   'mean duration':39.749*Duration_Impact,
                   'var duration':975.23*Duration_Impact,
                   'alpha capacity loss':5.19123,
                   'beta capacity loss':2.22481}}

#################### RUN SIMULATIONS ###############################

ue_base_stats = Run_sims(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, base_params, 0, num_samples, 1,  file_dl, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot_root + "_ue_base.png", critical_link_root + "_ue_base.txt") #UEsim = 1
convert_network.format_stats(ue_base_stats, stats_root + "_ue_base.txt")
ue_base_t, ue_base_mean_depart, ue_base_mean_arrive = convert_network.plot_curves(ue_base_stats, curve_plot_root + "_ue_base.png")

ue_mon_stats = Run_sims(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, mon_params, 0, num_samples, 1,  file_dl, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot_root + "_ue_mon.png", critical_link_root + "_ue_mon.txt") #UEsim = 1
convert_network.format_stats(ue_mon_stats, stats_root + "_ue_mon.txt")
ue_mon_t, ue_mon_mean_depart, ue_mon_mean_arrive = convert_network.plot_curves(ue_mon_stats, curve_plot_root + "_ue_mon.png")
 
so_base_stats = Run_sims(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, base_params, Routing_Impact, num_samples, 1,  file_dl, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot_root + "_so_base.png", critical_link_root + "_so_base.txt") #UEsim = 0
convert_network.format_stats(so_base_stats, stats_root + "_so_base.txt")
so_base_t, so_base_mean_depart, so_base_mean_arrive = convert_network.plot_curves(so_base_stats, curve_plot_root + "_so_base.png")

so_mon_stats = Run_sims(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, mon_params, Routing_Impact, num_samples, 1,  file_dl, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot_root + "_so_mon.png", critical_link_root + "_so_mon.txt") #UEsim = 0
convert_network.format_stats(so_mon_stats, stats_root + "_so_mon.txt")
so_mon_t, so_mon_mean_depart, so_mon_mean_arrive = convert_network.plot_curves(so_mon_stats, curve_plot_root + "_so_mon.png")

convert_network.plot_all_curves(so_base_t, so_base_mean_depart, so_base_mean_arrive, so_mon_t, so_mon_mean_depart, so_mon_mean_arrive, ue_base_t, ue_base_mean_depart, ue_base_mean_arrive, ue_mon_t, ue_mon_mean_depart, ue_mon_mean_arrive, curve_plot_root + "All_Curves.png")

with open(OUT_FILE, 'wb') as f:
    pickle.dump([so_base_stats, so_base_t, so_base_mean_depart, so_base_mean_arrive, so_mon_stats, so_mon_t, so_mon_mean_depart, so_mon_mean_arrive, ue_base_stats, ue_base_t, ue_base_mean_depart, ue_base_mean_arrive, ue_mon_stats, ue_mon_t, ue_mon_mean_depart, ue_mon_mean_arrive], f)
with open(OUT_FILE + "_ue_mon.txt", 'wb') as f:
    pickle.dump([ue_mon_stats, ue_mon_t, ue_mon_mean_depart, ue_mon_mean_arrive], f)
with open(OUT_FILE + "_ue_base.txt", 'wb') as f:
    pickle.dump([ue_base_stats, ue_base_t, ue_base_mean_depart, ue_base_mean_arrive], f)
with open(OUT_FILE + "_so_mon.txt", 'wb') as f:
    pickle.dump([so_mon_stats, so_mon_t, so_mon_mean_depart, so_mon_mean_arrive], f)
with open(OUT_FILE + "_so_base.txt", 'wb') as f:
    pickle.dump([so_base_stats, so_base_t, so_base_mean_depart, so_base_mean_arrive], f)
