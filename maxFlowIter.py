# ------ Run max flow

#import network
import iGraphNetwork
import utils
from collections import defaultdict
import numpy as np
import convert_network
import copy

def InitializeMaxFlowIter(Demand, NET_FILE, LINKS_FILE, TICK_SIZE, incident_params, Routing_Impact, UEsim, demand_file, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot, critical_link):
    #---------------------------------------------------------------------------
    print("Initializing...")
    ###################### Initialize Net ###############################  
    Excess = 0
    LastExcess = 0
    cumulative_depart = 0
    cumulative_arrive = 0
    queue_len = 0
    Critical = {}
    
    net = iGraphNetwork.iGraphNetwork(NET_FILE, LINKS_FILE, incident_params, TICK_SIZE)
    source = 0
    sink = 1

    sources = np.linspace(2,111,110)
    net.createShortestPaths(sources, sink)

    t = 1
    
    ThisIn = 0  
    for key in Demand[t]:
        ThisIn += Demand[t][key]    
        
    for i in range(net.numNodes):
        for j in range(net.numNodes):
            if net.matrix[i][j] == 1:
                Critical[i,j] = 0
    
    ################### Run Simulation ############################################
    
    while t<=288 or cumulative_arrive+0.01 < cumulative_depart:        
        ThisIn = 0
        for key in Demand[t]:
            ThisIn += Demand[t][key]
                 
        net.capacity = copy.deepcopy(net.BASE_capacity)
        
        net.UpdateCapacitiesFromDemand(demand_file, tntp_file, flows_file, tntp_params_file, t, Demand, 0, link_data_file)
        b1, flow1 = net.maxFlow(source, sink) #b is total flow, flow is flow on each link
        SO = copy.deepcopy(flow1)
        
        net.UpdateCapacitiesFromDemand(demand_file, tntp_file, flows_file, tntp_params_file, t, Demand, UEsim, link_data_file)
        #net.apply_incidents(t,UEsim)
        b2, flow2 = net.maxFlow(source, sink) #b is total flow, flow is flow on each link
        UE = copy.deepcopy(flow2)                
        
        net.LinearInterpolation(SO, UE, Routing_Impact)
        b, flow = net.maxFlow(source, sink) #b is total flow, flow is flow on each link
        
        net.Time_Flow[t] = copy.deepcopy(flow)
        net.Time_B[t] = copy.deepcopy(b)
        net.Time_Demand[t] = copy.deepcopy(Demand[t])
        #net.Time_Capacity[t] = copy.deepcopy(net.capacity)
    
    
        for (i,j) in Critical:
            if flow[i][j] == net.BASE_capacity[i][j]:
                Critical[i,j] += 1
                
        Excess = 0
        for key in Demand[t]:
            if flow[0][key] < Demand[t][key]:
                if ( t+1 in Demand ) & ( key in Demand[t+1] ):
                    Demand[t+1][key] = Demand[t+1][key] + (Demand[t][key] - flow[0][key])
                    Excess += (Demand[t][key] - flow[0][key])
                else:
                    Demand[t+1][key] = (Demand[t][key] - flow[0][key])
                    Excess += (Demand[t][key] - flow[0][key])
            

        cumulative_depart += ThisIn-LastExcess
        queue_len = max(0, queue_len + (ThisIn-LastExcess) - b)
        cumulative_arrive = cumulative_depart - queue_len

        LastExcess = Excess
        #print(t)
        t += 1
    
    convert_network.maxflow_to_plot(net,
                                Critical,
                                coordinate,
                                critical_plot)
    convert_network.maxflow_to_link(net,
                                Critical,
                                critical_link)
            
    return net

def maxFlowIterRun(net, Demand, TICK_SIZE, UEsim, demand_file, tntp_file, flows_file, tntp_params_file, link_data_file, coordinate, critical_plot, critical_link):
    #---------------------------------------------------------------------------
    ###################### Initialize Stats ###############################  
    Excess = 0
    LastExcess = 0
    cumulative_moving = 0
    cumulative_depart = 0
    cumulative_arrive = 0
    queue_len = 0
    total_delay = 0
    stats = {}
    stats['full profile'] = []
    EndTime = 288

    
    source = 0
    sink = 1
    t = 1
    
    ThisIn = 0  
    for key in Demand[t]:
        ThisIn += Demand[t][key]
    
    net.generate_incidents()
    
    ################### Run Simulation ############################################
    while t<=288 or cumulative_arrive < cumulative_depart:        
        ThisIn = 0
        update = 0
        for key in Demand[t]:
            ThisIn += Demand[t][key]

        if t in net.Time_Demand.keys():
            if Demand[t] == net.Time_Demand[t]:
                update = 0
            else:
                update = 1
        else: 
            update = 1

        net.capacity = copy.deepcopy(net.BASE_capacity)
        update = max(update, net.apply_incidents(t))  #apply_incidents updates net.capacity and returns whether this impacts flow at this timestep

        if update == 0 and t<288:   ######################### The demand and capacities are the same as the initilization and still within 3-day window#############################
            #net.capacity = net.Time_Capacity[t]
            b = net.Time_B[t]
            flow = net.Time_Flow[t]
        else:     #########################New Demands or Capacities, need to update (also here if outside 3 day window for consistency with no incidents solution) ##########################
            if t+130 in net.Time_Flow.keys(): #130 15-minute timesteps lookback heuristic calibrated to peak demand in scenario 1, can be adjusted based on scenario
                EndTime = t
                net.capacity = copy.deepcopy(net.Time_Flow[t])
                #net.UpdateCapacitiesFromBaseFlow(net.Time_Flow[t])
                net.UpdateCapacitiesFromNewDemand(t,Demand)
                net.apply_incidents(t)
                b, flow = net.maxFlow(source, sink) 
            else: 
                net.capacity = copy.deepcopy(net.Time_Flow[EndTime])
                #net.UpdateCapacitiesFromBaseFlow(net.Time_Flow[EndTime])
                net.UpdateCapacitiesFromNewDemand(t,Demand)
                net.apply_incidents(t)
                b, flow = net.maxFlow(source, sink) 
        for i in range(len(flow)):
            for j in range(len(flow[i])):        
                cumulative_moving += flow[i][j]*net.BASE_TT[i][j] 
        
        Excess = 0
        for key in Demand[t]:
            if flow[0][key] < Demand[t][key]:
                if ( t+1 in Demand ) & ( key in Demand[t+1] ):
                    Demand[t+1][key] = Demand[t+1][key] + (Demand[t][key] - flow[0][key])
                    Excess += (Demand[t][key] - flow[0][key])
                else:
                    Demand[t+1][key] = (Demand[t][key] - flow[0][key])
                    Excess += (Demand[t][key] - flow[0][key])

        cumulative_depart += ThisIn-LastExcess
        queue_len = max(0, queue_len + (ThisIn-LastExcess) - b)
        cumulative_arrive = cumulative_depart - queue_len
        total_delay += queue_len
        stats['full profile'].append((t*TICK_SIZE/60, cumulative_depart, cumulative_arrive,
                                      queue_len))

        LastExcess = Excess
        t += 1
        

    number, time, capacity = net.analyze_incidents()
    stats['total delay'] = total_delay * TICK_SIZE/60
    stats['total moving'] = cumulative_moving #sum(sum(tt@flow))#utils.dotproduct(net_data[0], net_data[1])
    stats['total time'] = stats['total delay'] + stats['total moving']
    stats['avg delay'] = stats['total delay'] / cumulative_arrive
    stats['avg moving'] = stats['total moving'] / cumulative_arrive
    stats['avg time'] = stats['total time'] / cumulative_arrive
    stats['last arrival'] = t*TICK_SIZE/60
    stats['total incidents'] = number
    stats['total incident time'] = time
    stats['total incident capacity drop'] = capacity
    
    return stats
