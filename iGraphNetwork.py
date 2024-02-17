# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:52:22 2023

@author: jaker
"""

import utils
import igraph
from igraph import Graph
import numpy as np
import copy
import random
import math
import convert_network
import os # to call tap-b

class BadNetworkException(Exception):
    """
    You can raise this exception if the network is invalid in some way (e.g.,
    the network matrix is not square.)
    """
    pass

class iGraphNetwork:
        
    def maxFlow(self, source, sink):
        """
        This method finds a maximum flow in the network.  Return a tuple 
        containing the total amount of flow shipped, and the flow on each link
        in a list-of-lists (which takes the same form as the adjacency matrix).
        These are already created for you at the start of the method, fill them
        in with your implementation of minimum cost flow.
        """
        g = igraph.Graph.Weighted_Adjacency(self.capacity, mode='directed', attr='capacity')
        Flow = g.maxflow(0,1,"capacity")
        g.es["flow"] = Flow.flow
        flowout = g.get_adjacency(attribute="flow")
        return Flow.value, np.array(flowout.data)

    def createShortestPaths(self,sources,sink):
        """
        Used to initialize all shortest paths
        """
        g = igraph.Graph.Weighted_Adjacency(self.BASE_TT, mode='directed', attr='TT')

        for source in sources:
            self.AllShortestPaths[int(source)] = g.get_shortest_paths(int(source), to=sink, weights=g.es["TT"])


    def getShortestPath(self,source, b, flow, maxFlow):
        """
        Retrieves the shortest path to the sink given a source node. 
        Calculates the capacity of a path based on the residual network. 
        Subtracts that much flow from the residual capacity of the path. 
        Adds new flow to paths 
        Adds new flow to total flow
        """
        capacity = maxFlow

        if source in self.AllShortestPaths.keys():
            path = self.AllShortestPaths[source][0]
            #print(path)
            for i in range(1, len(path)):
                capacity = min(capacity,self.capacity[path[i-1]][path[i]])
            #print("capacity: " + str(capacity))
            if capacity < 0:
                print("CAPACITY ERRORR")

            #update flow between supersource and origin, then update flow on every link on the path. 
            self.capacity[0][path[0]] -= capacity
            flow[0][path[0]] += capacity
            b += capacity
            for i in range(1, len(path)):
                self.capacity[path[i-1]][path[i]] -= capacity
                flow[path[i-1]][path[i]] += capacity
                #b += capacity
        else: 
            print("ERROR")
        return b, flow

    def readNetworkFile(self, networkFileName):
        """
        Reads a network adjacency matrix from the given file and set up
        other network data structures.
        """
        networkLines = utils.processFile(networkFileName)
        self.matrix = list()
        for rawLine in networkLines:
            matrixEntries = list(map(int,rawLine.split(",")))
            self.matrix.append(matrixEntries)
        self.numNodes = len(self.matrix)
        self.dimensionNetwork()

    def dimensionNetwork(self):
        self.cost = list()
        self.capacity = list()
        self.length = list()

        for i in range(self.numNodes):
            self.cost.append([0] * self.numNodes)
            self.capacity.append([0] * self.numNodes)
            self.length.append([0] * self.numNodes)
        self.supply = [0] * self.numNodes

    def readLinkDataFile(self, linkDataFileName):
        """
        Reads link data from a given file.  Format of this file is a series
        of rows, one for each link:

        tail,head,cost,capacity,length
        """
        linkLines = utils.processFile(linkDataFileName)
        for link in linkLines:
            try:
                tail, head, cost, capacity, length = link.split(",")
                tail = int(tail)
                head = int(head)
                if (tail < 0 or tail >= self.numNodes):
                    print(tail)
                    print(head)
                    print(self.numNodes)
                    raise BadNetworkException

                if (head < 0 or head >= self.numNodes):
                    raise BadNetworkException
                    print(tail)
                    print(head)
                    print(self.numNodes)

                if (self.matrix[tail][head] != 1):
                    print(tail)
                    print(head)
                    print(self.matrix[tail][head])
                    raise BadNetworkException

                self.cost[tail][head] = float(cost)
                self.capacity[tail][head] = int(capacity)
                self.length[tail][head] = float(length)
            except:
                print("Row %s in link file does not correspond to a valid "
                      "link." % link)
                raise BadNetworkException
                    
    def LinearInterpolation(self, SO, UE, Routing_Impact):
        """
        updates capacities based on linear interpolation of UE and SO
        """
        for i in range(len(SO)):
            for j in range(len(SO[i])): 
                self.capacity[i][j] = (Routing_Impact)*SO[i][j] + (1-Routing_Impact)*UE[i][j]

    def UpdateCapacitiesFromBaseFlow(self, BaseFlow):
        """
        updates capacities based on base flow. Not pretty but should lead to flows <= base from each origin. 
        """
        AllHeads = np.linspace(2,111,110)
        for head in AllHeads:
            self.capacity[0][int(head)] = BaseFlow[0][int(head)]

    def UpdateCapacitiesFromNewDemand(self, t, Demand):
        for head in Demand[t]:
            self.capacity[0][head] = Demand[t][head]

    def UpdateCapacitiesFromDemand(self, demand_file, tntp_file, flows_file, tntp_params_file, t, Demand, UEsim, link_data_file):
        """
        updates capacities based on demand

        tail,head,cost,capacity

        demand_file
        tntp_file
        flows_file
        tntp_params_file

        """
        if UEsim == 1:
            convert_network.demand_adjustments(demand_file, t, Demand)
            #Solve UE
            with open(tntp_params_file, "w") as tpf:
                tpf.write(f"<NETWORK FILE> {tntp_file}\n"
                    f"<TRIPS FILE> {demand_file}\n"
                    f"<FLOWS FILE> {flows_file}\n"    #flows and travel times
                    f"<DATA PATH>\n"
                    f"<FILE PATH>\n"
                    f"<CONVERGENCE GAP> 1e-2\n"
                    f"<MAX ITERATIONS> 10")
            os.system(f"./tap {tntp_params_file}")

            #Convert UE data to max-flow files, update to network. 
            convert_network.tntp_flows_to_links(flows_file,
                                                   tntp_file,
                                                   demand_file,
                                                   link_data_file) #UE max-flow input (capacities are flows in UE)
            self.dimensionNetwork()            
            self.readLinkDataFile(link_data_file)


        #if ( t in Demand ):
        for head in Demand[t]:
            self.capacity[0][head] = Demand[t][head]




    def generate_incidents(self):
        """
        Generate a list of incidents in the format needed by compute_stats.
        In the current implementation, we assume the number of incidents
        is Poisson, duration is exponential, and capacity loss is normally
        distributed, and required keys are:
            'incident rate' - avg incident rate (incidents per hour)
            'time horizon' - hours (latest time at which an incident can
                             begin... this should be later than when the
                             queue finally clears.  There is no harm in
                             having incidents after this fact, they will
                             not affect the analysis in any way.)
            'mean duration' - avg duration
            'mean capacity loss' - avg capacity loss
            'stdev capacity loss' - stdev for capacity loss
        """     
        random.seed(self.counter)

        self.counter = self.counter + 1
        self.incident = {}
        self.number = 0
        for i in range(111, self.numNodes):
            for j in range(111, self.numNodes):
                if self.matrix[i][j] ==1:
                    ####################### GENERATE COLLISIONS ########################
                    params = self.All_Params['Collision Parameters']
                    if params['incident rate'] == 0:
                        t = 0
                    else:
                        t = random.expovariate(params['incident rate']/100000000*self.BASE_capacity[i][j]*self.BASE_length[i][j]/5280)
                    while t < params['time horizon']:
                        inc ={'start time':random.uniform(0, params['time horizon']),
                             'duration':random.gammavariate(params['mean duration']*params['mean duration']/params['var duration'], params['var duration']/params['mean duration'])/60, #mean = alpha*beta; variance = alpha*beta**2
                             'capacity loss':random.betavariate(params['alpha capacity loss'],params['beta capacity loss'])
                              }
                        self.number += 1
                        
                        idx = math.ceil(inc['start time']*60/self.TICK_SIZE) #next 15 minute increment
                        
                        while inc['start time'] <= idx*self.TICK_SIZE/60 <= inc['start time'] + inc['duration']:
                            if ( idx in self.incident ):
                                self.incident[idx][(i,j)] = inc['capacity loss']
                            else:
                                self.incident[idx] = {}
                                self.incident[idx][(i,j)] = inc['capacity loss']
                            idx += 1
                            
                        t += random.expovariate(params['incident rate']/100000000*self.BASE_capacity[i][j]*self.BASE_length[i][j]/5280)


                    ####################### GENERATE DISABLED/ABANDONED ########################
                    
                    params = self.All_Params['Disabled Parameters']
                    if params['incident rate'] == 0:
                        t = 0
                    else:
                        t = random.expovariate(params['incident rate']/100000000*self.BASE_capacity[i][j]*self.BASE_length[i][j]/5280)
                    while t < params['time horizon']:
                        inc ={'start time':random.uniform(0, params['time horizon']),
                             'duration':random.gammavariate(params['mean duration']*params['mean duration']/params['var duration'], params['var duration']/params['mean duration'])/60, #mean = alpha*beta; variance = alpha*beta**2
                             'capacity loss':random.betavariate(params['alpha capacity loss'],params['beta capacity loss'])
                              }
                        self.number += 1
                        
                        idx = math.ceil(inc['start time']*60/self.TICK_SIZE) #next 15 minute increment
                        
                        while inc['start time'] <= idx*self.TICK_SIZE/60 <= inc['start time'] + inc['duration']:
                            if ( idx in self.incident ):
                                self.incident[idx][(i,j)] = inc['capacity loss']
                            else:
                                self.incident[idx] = {}
                                self.incident[idx][(i,j)] = inc['capacity loss']
                            idx += 1
                            
                        t += random.expovariate(params['incident rate']/100000000*self.BASE_capacity[i][j]*self.BASE_length[i][j]/5280)

    
    def apply_incidents(self,t):
        #Reset from last run to base network
        #if t-1 in self.incident.keys():
        #    for (i,j) in self.incident[t-1].keys():
        #        self.capacity[i][j] = self.BASE_capacity[i][j]

        #Update with new incidents
        update = 0
        if t in self.incident.keys():
            for (i,j) in self.incident[t].keys():
                self.capacity[i][j] = int(max(0,min(self.capacity[i][j], self.BASE_capacity[i][j]*self.incident[t][(i,j)]))) #Boyles used self.BASE_capacity[i][j] - self.incident[t][(i,j)]
                if t in self.Time_Flow.keys():
                    if self.Time_Flow[t][i][j] > self.capacity[i][j]:
                        update = 1
                else:
                    update = 1
        return update
    
                



    def analyze_incidents(self):
        cap = 0
        time = 0
        
        for t in self.incident.keys():
            for (i,j) in self.incident[t].keys():
                cap += self.BASE_capacity[i][j] - self.BASE_capacity[i][j]*self.incident[t][(i,j)]
                time += self.TICK_SIZE/60
        
        return self.number, time, cap
                
        
    def __init__(self, networkFile, linkDataFile, incident_params, TICK_SIZE):
        self.counter = 1
        self.numNodes = 0
        self.matrix = list()
        self.cost = list()  #now fftt in hrs
        self.capacity = list()
        self.length = list()
        self.supply = list()
        self.incident = {}
        self.TICK_SIZE = TICK_SIZE
        self.number = 0
        self.moving = 0
        
        self.AllShortestPaths = {}
        self.residualDemand = {}
        #self.residualCapacity = list()

        self.Time_Flow = {}
        self.Time_B = {}
        self.Time_Demand = {}
        self.Time_Capacity = {}
        self.Time_Cost = {}

        self.readNetworkFile(networkFile)
        self.readLinkDataFile(linkDataFile) 
            
        self.BASE_capacity = copy.deepcopy(self.capacity)
        self.BASE_length = copy.deepcopy(self.length)
        self.BASE_TT = copy.deepcopy(self.cost)
        
        self.All_Params = incident_params
