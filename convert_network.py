"""
Convert network from the evacsim format to that used by your 367R
max-flow code.  Adds a "supersource" node as node 0, and preserves
node ordering from the other files (which is 1-based, 1 is sink)
"""


import matplotlib as mpl
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import numpy as np
import utils

TICK_SIZE = 15 #15min
ARTIFICIAL_CAP = 0 # capacity for artificial links
TIME_HORIZON = 3 * 24 # hours (to convert capacity)
TIME_HORIZON = TICK_SIZE/60

def convertNewNet(TICK_SIZE, file_nl, file_dl, file_nf, file_lf):
	#ARTIFICIAL_CAP = 300

	network_lines = utils.processFile(file_nl, "~")
	demand_lines = utils.processFile(file_dl, "~")
	net_file = open(file_nf, "w")
	links_file = open(file_lf, "w")



	for line in network_lines:
	    if line.find("NUMBER OF NODES") >= 0:
	        num_nodes = int(line[17:])
	        break

	net_matrix = []
	for _ in range(num_nodes + 1):
	    net_matrix.append(["0"] * (num_nodes + 1))

	for line in network_lines:
	    if line[0] == '<':
	        continue
	    fields = line.split()
	    tail = int(fields[0])
	    head = int(fields[1])
	    capacity = int(int(fields[2])*(TICK_SIZE/60))
	    length = float(fields[3])
	    speed = int(fields[4])
	    tt = length/speed/5280
	    net_matrix[tail][head] = "1"
	    links_file.write(f"{tail},{head},{tt},{capacity},{length}\n")

	demand = 0
	for line in demand_lines:
	    if line[0] == '<':
	        continue
	    if line[:6] == 'Origin':
	        origin = int(line[6:])
	    if line[:2] == '1:':
	        demand = float(line[3:])
	    if demand > 0:
	        #print(f"Adding link from supersource to {origin}")
	        net_matrix[0][origin] = "1"
	        capacity = 0
	        links_file.write(f"0,{origin},0.000001,{capacity},{0.0}\n")  #TT 0.0000001 so does not affect solution but graph still connected in iGraph
	        demand = 0

	for i in range(num_nodes + 1):
	    net_file.write(",".join(net_matrix[i]) + "\n")

def dta_to_tntp(evacsim_net, evacsim_demand, new_net):
    """
    Convert network from the evacsim format to the TNTP format for
    static traffic assignment.

    evacsim_net = name of network file (.net) to convert
    evacsim_demand = name of static demand (.ods) file to convert
    new_net = name of file to contain TNTP network data

    Assumes the following
        -capacity in evacsim_net is in veh/hr
        -BPR alpha and beta values are as given below
        -free flow time = length / free flow speed
            (converting units to min)
        -other fields (toll, link_type) are arbitrary
    """

    BPR_ALPHA = 0.15
    BPR_BETA = 4

    network_lines = utils.processFile(evacsim_net, "~")
    demand_lines = utils.processFile(evacsim_demand, "~")
    net_file = open(new_net, "w")

    for line in network_lines:
        if line.find("NUMBER OF NODES") >= 0:
            num_nodes = int(line[17:])
        elif line.find("NUMBER OF LINKS") >= 0:
            num_links = int(line[17:])        
        elif line.find("NUMBER OF ZONES") >= 0:
            num_zones = int(line[17:])        
        elif line.find("END OF METADATA") >= 0:        
            # Input file metadata done, so can write all
            # output file metadata now.
            net_file.write(f"<NUMBER OF NODES> {num_nodes}\n")
            net_file.write(f"<NUMBER OF LINKS> {num_links}\n")
            net_file.write(f"<NUMBER OF ZONES> {num_zones}\n")
            net_file.write(f"<FIRST THRU NODE> {num_zones+1}\n")
            net_file.write(f"<END OF METADATA>\n")
            net_file.write("\n\n")
            net_file.write("~ init_node term_node capacity length "
                           "free_flow_time alpha beta speed toll link_type ;\n")
        else:
            # If none of the above cases apply, it's an
            # ordinary link to convert
            fields = line.split()
            tail = fields[0]
            head = fields[1]
            capacity = float(fields[2])
            length = float(fields[3])
            ffs = float(fields[4]) # free-flow speed
            net_file.write(f"\t{tail}"
                           f"\t{head}"
                           f"\t{capacity * TIME_HORIZON}"
                           f"\t{length}"
                           f"\t{(length/5280)/(ffs)*60}"
                           f"\t{BPR_ALPHA}"
                           f"\t{BPR_BETA}"
                           f"\t{ffs}"
                           f"\t{0}"
                           f"\t{0}"
                           "\t;\n")

def tntp_flows_to_links(tntp_flows, tntp_net, trips_file, link_data_file):
    """
    Convert the output of tap-b (static assignment) into the format
    needed for the max flow code.

    tntp_flows = UE link flows file (e.g., from tap-b)
    tntp_net = TNTP network file (needed to grab capacities)
    trips_file = static demand (.ods) file, used to determine
                 links from supersource
    link_data_file = file to write with link data, same as new_links
                     in dta_to_367r.  If None, doesn't write a file,
                     but still does the calculations below.

    Returns a list of tuples giving each link's flow and free-flow time,
    converted from minutes to hours.
    """
    flow_lines = utils.processFile(tntp_flows, "~")
    tntp_lines = utils.processFile(tntp_net, "~")
    demand_lines = utils.processFile(trips_file, "~")

    # This is an ugly hack and assumes the TNTP file was created by the
    # dta_to_tntp function in this file; therefore there are exactly 
    # 5 lines of header (skipping blanks) that need to be removed before
    # we get to the link data
    tntp_lines = tntp_lines[5:]
    # Turn off assertion until we can get tap-b to suppress artificial links
    #assert(len(tntp_lines) == len(flow_lines))
    if link_data_file is not None:
        links_file = open(link_data_file, "w")

    for ij in range(len(tntp_lines)):
        flow_fields = flow_lines[ij].split(" ")
        net_fields = tntp_lines[ij].split()
        tail = net_fields[0]
        head = net_fields[1]
        flow = float(flow_fields[1])
        capacity = float(net_fields[2])
        free_flow_time = float(net_fields[4]) / 60 # convert to hours
        maxflow_cap = int(min(flow, capacity) / TIME_HORIZON *TICK_SIZE/60)#Do not need to convert to vph since using 15 min increment / TIME_HORIZON) # convert to vph
        if link_data_file is not None:
            links_file.write(f"{tail},{head},{free_flow_time},{maxflow_cap},{0}\n")
        
    demand = 0
    for line in demand_lines:
        if line[0] == '<':
            continue
        if line[:6] == 'Origin':
            origin = int(line[6:])
        if line[:2] == '1:':
            demand = float(line[3:])
        if demand > 0:
            #print(f"Adding link from supersource to {origin}")
            links_file.write(f"0,{origin},0,{ARTIFICIAL_CAP},{0}\n")
            demand = 0


def demand_adjustments(evacsim_demand, t, Demand):
    demand_lines = utils.processFile(evacsim_demand, "~")
    
    lines = []

    # Parse the file into lines
    for line in demand_lines:
        if line[:6] == 'Origin':
            origin = int(line[6:])
        if line[:2] == '1:':
            if origin in Demand[t].keys():
                line = '1: ' + str(Demand[t][origin])
            else:
                line = '1: ' + str(0.0)
        lines.append(line + '\n')
    
    # Write them back to the file
    with open(evacsim_demand, 'w') as f:
        f.writelines(lines)

def format_stats(stats, stats_file):
    """
    Output evacuation metric statistics in a text file.
    """
    with open(stats_file, "w") as sf:
        sf.write("Metric: mean (stdev)\n")
        sf.write(f"Total delay: {stats['total delay mean']}({stats['total delay stdev']})\n")
        sf.write(f"Total moving time: {stats['total moving mean']}({stats['total moving stdev']})\n")
        sf.write(f"Total time: {stats['total time mean']}({stats['total time stdev']})\n")
        sf.write(f"Average delay: {stats['avg delay mean']}({stats['avg delay stdev']})\n")
        sf.write(f"Average moving time: {stats['avg moving mean']}({stats['avg moving stdev']})\n")
        sf.write(f"Average total time: {stats['avg time mean']}({stats['avg time stdev']})\n")
        sf.write(f"Last arrival time: {stats['last arrival mean']}({stats['last arrival stdev']})\n")
        sf.write(f"Total incidents: {stats['total incidents mean']}({stats['total incidents stdev']})\n")
        sf.write(f"Total incident time: {stats['total incident time mean']}({stats['total incident time stdev']})\n")
        sf.write(f"total incident capacity drop: {stats['total incident capacity drop mean']}({stats['total incident capacity drop stdev']})\n")

def plot_curves(stats, plot_file):
    """
    Plot departure/arrival curves (+/- 1 SD) in a PNG file.
    """
    t = [p['t'] for p in stats['full profile']]
    mean_depart = [p['departure mean'] for p in stats['full profile']]
    mean_arrive = [p['arrival mean'] for p in stats['full profile']]
    ub_arrive = [p['arrival mean'] + p['arrival stdev']
                 for p in stats['full profile']]
    lb_arrive = [p['arrival mean'] - p['arrival stdev']
                 for p in stats['full profile']]
    fig, ax = plt.subplots()
    ax.plot(t, mean_depart)
    ax.plot(t, mean_arrive)
    #ax.plot(t, ub_arrive)
    #ax.plot(t, lb_arrive)
    plt.xlabel('Time (hrs)', fontweight ='bold', fontsize = 15)
    plt.xticks(fontsize = 15)
    ax.yaxis.offsetText.set_fontsize(15)
    plt.ylabel('People', fontweight ='bold', fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.savefig(plot_file)
    return t, mean_depart, mean_arrive

def plot_all_curves(so_base_t, so_base_mean_depart, so_base_mean_arrive, so_mon_t, so_mon_mean_depart, so_mon_mean_arrive, ue_base_t, ue_base_mean_depart, ue_base_mean_arrive, ue_mon_t, ue_mon_mean_depart, ue_mon_mean_arrive, plot_file):

    fig, ax = plt.subplots()
    #ax.plot(so_base_t, (np.array(so_base_mean_depart)/100000).tolist())
    #ax.plot(so_mon_t, (np.array(so_mon_mean_depart)/100000).tolist())
    #ax.plot(ue_mon_t, (np.array(ue_mon_mean_depart)/100000).tolist())
    ax.plot(ue_base_t, (np.array(ue_base_mean_depart)/100000).tolist())
    
    ax.plot(so_base_t, (np.array(so_base_mean_arrive)/100000).tolist())
    ax.plot(so_mon_t, (np.array(so_mon_mean_arrive)/100000).tolist())
    ax.plot(ue_base_t, (np.array(ue_base_mean_arrive)/100000).tolist())
    ax.plot(ue_mon_t, (np.array(ue_mon_mean_arrive)/100000).tolist())

    plt.legend(["Departures", "SO Base", "SO Mon", "UE Base", "UE Mon"], loc ="lower right")

    plt.xlabel('Time (hrs)', fontweight ='bold', fontsize = 15)
    plt.xticks(fontsize = 15)
    ax.yaxis.offsetText.set_fontsize(15)
    plt.ylabel('100,000 People', fontweight ='bold', fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.savefig(plot_file, dpi=1200)


def maxflow_to_plot(net, Critical, coord, plot_file):
    """
    Create a plot of a network, highlighting links at capacity in a
    max-flow solution (the union of all minimum cuts).

    net = network object (see network.py)
    flow = flow matrix corresponding to net
    coord = dictionary of tuples with (x,y) coordinates for each node
    plot_file = name of PNG file to write
    """
    links = []
    cap_links = []
    # starting points for ranges...
    # 0 (or empty) includes supersource/sink, don't want this since the
    #              coordinates don't make sense
    # 2 plots all links except those connecting to supersource/sink
    # 111 plots all "real" links, excluding centroid connectors and artificial
    for i in range(111, net.numNodes):
        for j in range(111, net.numNodes):
            if net.matrix[i][j] == 1:
                links.append([coord[i],coord[j]])
                if (Critical[i,j] >= 1):
                    cap_links.append([coord[i],coord[j]])
                


    # Applied folium package to plot the map and draw polylines. 
    import folium
    my_map = folium.Map(location = [coord[2][1], coord[2][0]], zoom_start = 10, control_scale = True)

    # switch x and y for coordinates of links and cap_links
    links_upd = []
    cap_links_upd = []
    for i in range(len(links)):
        links_upd.append([(links[i][0][1], links[i][0][0]), (links[i][1][1], links[i][1][0])])
    for i in range(len(cap_links)):
        cap_links_upd.append([(cap_links[i][0][1], cap_links[i][0][0]), (cap_links[i][1][1], cap_links[i][1][0])])

    folium.PolyLine(links_upd, color = 'black').add_to(my_map)
    folium.PolyLine(cap_links_upd, color = 'red').add_to(my_map)
    my_map.save(plot_file + ".html")

    # lc = mc.LineCollection(links, color = (0,0,0,1), linewidths=1)
    # clc = mc.LineCollection(cap_links, color = (1,0,0,1), linewidths=3)

    # fig, ax = plt.subplots()  # a figure with a single Axes
    # ax.add_collection(lc)
    # ax.add_collection(clc)
    # ax.autoscale()
    # fig.savefig(plot_file)
    #lc = mc.LineCollection(links, color = (0,0,0,1), linewidths=1)
    #clc = mc.LineCollection(cap_links, color = (1,0,0,1), linewidths=3)

    #fig, ax = plt.subplots()  # a figure with a single Axes
    #ax.add_collection(lc)
    #ax.add_collection(clc)
    #ax.autoscale()
    #fig.savefig(plot_file)

def maxflow_to_link(net, Critical, link_file):
    """
    Create a list of links at capacity in a max-flow solution (links
    part of some minimum cut)

    net = network object (see network.py)
    link_file = name of text file to write
    """

    # For starting range see comments in maxflow_to_plot
    with open(link_file, "w") as lf:
        for i in range(111, net.numNodes):
            for j in range(111, net.numNodes):
                if (net.matrix[i][j] == 1):
                    if (Critical[i,j] >= 1):
                        lf.write(f"({i},{j}) {Critical[i,j]} \n")
