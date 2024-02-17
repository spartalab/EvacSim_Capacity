# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:35:31 2024

@author: jaker
"""
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import pickle
    
def BuildNetwork(demand_file1, demand_file2, GIS_nodes, GIS_net, raw_link_file, node_file, county_file, safe_node_file, demand_out1, demand_out2, out_nxy, net_out, links_upd_out, dup_out, ID_out, OUT_FILE):

    #### First: Check all nodes in the link file
    
    def nodes_not_in_network(raw_link_file, nodes_nxy):
        links        = pd.read_csv(raw_link_file, header=0)
        links_nodes  = links.iloc[:, 0:2]
        print(links_nodes.head())
        
        oldID = []
        for n in range(nodes_nxy.shape[0]):
            if nodes_nxy.iloc[n, 0] not in set(links_nodes.iloc[:, 0].tolist()):
                if nodes_nxy.iloc[n, 0] not in set(links_nodes.iloc[:, 1].tolist()):
                    oldID.append(nodes_nxy.iloc[n, 0])
        
        nodes_nxy = pd.DataFrame({'ID': oldID}) 
        return nodes_nxy
    
    
    def rmv_nodes_not_in_network(target_nodes, nodes_nxy):
        if len(target_nodes) != 0:
            for nid in target_nodes.iloc[:, 0]:
                print(nid)
                nodes_nxy = nodes_nxy[nodes_nxy['Node'] != nid]
            return nodes_nxy
        else:
            return nodes_nxy
        
    # HOUSTON Zones (sink node and counties) with a new ID list (OUTPUT: houston_zones)
    def create_zones(counties): 
        # counties
        county_file = pd.read_csv(counties, header=0)
        print('Duplicate counties:')
        duplicates = county_file[county_file['COUNTY'].duplicated()]
        print(duplicates)
        
        # add pseudo sink node
        sink_node = pd.DataFrame({'ID': [999999], 'COUNTY': ['Sink Node']})
        df = pd.concat([sink_node, county_file]).reset_index(drop = True)
        
        # create and rank a new ID list 
        df.insert(0, 'New_ID', range(1, len(df)+1))  
            
        return df
    
    # HOUSTON Interchanges ID ranking  (OUTPUT: houston_interchanges)
    def create_interchanges(zones, nodes_nxy):
        # all nodes
        print('Duplicate nodes:')
        duplicates = nodes_nxy[nodes_nxy.duplicated()]
        print(duplicates)
        
        nodes_nxy = nodes_nxy.iloc[:, 0:3]
        nodes_nxy = nodes_nxy.rename(columns = {"Node": "ID"})
        
        # find interchanges
        nodes = nodes_nxy['ID']
        print(f' Number of nodes {nodes.shape}')
        zones = zones['ID']
        print(f' Number of zones {zones.shape}')
        interchanges = pd.DataFrame(pd.concat([nodes, zones, zones]).drop_duplicates(keep=False)) # concat zones twice bc of sink node
        interchanges = interchanges.sort_values("ID").reset_index(drop = True)
        
        # rank interchanges
        interchanges.insert(0, 'New_ID', range(zones.shape[0] + 1, zones.shape[0] + len(interchanges) + 1)) #1 represents sink node  
        
        # insert one column
        interchanges.insert(2, 'COUNTY', 'Interchange')
        
        return interchanges
    
    
    def update_safe_nodes(interchanges, safe_nodes):
        shelters = pd.read_csv(safe_nodes, header=None)
        print('Duplicate safe nodes:')
        duplicates = shelters[shelters.duplicated()]
        print(duplicates)
        shelters = shelters.drop_duplicates()
        
        
        # replace interchange with safe nodes 
        for i in range(len(shelters)):
            interchanges.loc[interchanges['ID'] == shelters.iloc[i,0], 'COUNTY'] = 'Safe Node'
        
        interchange_duplicates = interchanges[interchanges.duplicated()]
        print('Duplicate interchanges:')
        print(interchange_duplicates)
        return interchanges
    
    # CONCAT Zones ID and Interchanges ID (OUTPUT: houston_ID)
    def concat_ID(ID1, ID2):
        new_ID = pd.concat([ID1, ID2])
        return new_ID
    
    def convert_to_df(lst):
        temp = []
        for i in range(len(lst)):
            print(lst[i].split('\t'))
            temp.append(lst[i].split('\t'))
        df = pd.DataFrame(temp)
        # First row as header
        df_header = df.iloc[0] 
        df = df[1:]
        df.columns = df_header
        
        return df
    
    def rearrange_colns(df):
        # Rearrange the columns order
        cols = df.columns.tolist()
        df = df.rename(columns={'Init_Node': 'Init_node', 'Term_Node':'Term_node'})
        new_cols = ['Init_node', 'Term_node', 'Capacity', 'Length(ft)', 'u_f', 'k_j']
        print(f'input cols {cols}\n new cols {new_cols}')
        
        df = df[new_cols]
        # Rename 
        df = df.rename(columns={'Length(ft)': 'Length (ft)', 'u_f': 'u_f (mph)', 'k_j': 'k_j (veh/mi)' })
    
        print(f'network size {df.shape}')
        return df
    
    def input_network_links(link_file):
        df = pd.read_csv(link_file, header=0)
        df = rearrange_colns(df)
        return df
        
    # MERGE New ID list (OUTPUT: links_ID_match)
    def merge_ID(df_links, newID_lst, dup_out):
        #     df_links['Init_node']=df_links['Init_node'].astype(int)
        #     df_links['Term_node']=df_links['Term_node'].astype(int)
        df_links['Init_node']=pd.to_numeric(df_links['Init_node'])
        df_links['Term_node']=pd.to_numeric(df_links['Term_node'])
    
        temp1 = newID_lst
        df1 = df_links.merge(temp1, left_on='Init_node', right_on='ID')
        df1 = df1.rename(columns={'New_ID':'From'})
        df1 = df1.loc[:, ['Init_node', 'Term_node', 'From']]
    
        temp2 = newID_lst
        df2 = df_links.merge(temp1, left_on='Term_node', right_on='ID')
        df2 = df2.rename(columns={'New_ID':'To'})
        df2 = df2.loc[:, ['Init_node', 'Term_node', 'To']]
    
        df = df1.merge(df2, how = 'inner')
    
        print(f'Duplicate links: {df[df.duplicated()].shape} ')
        duplicates = df[df.duplicated()]
        duplicates.to_csv(dup_out)
        print(duplicates)
        
        df = df.drop_duplicates()
        
        return df
        
    def update_network(new_ID, df_net, ID_match):
        # merge existing links
        df = new_ID.merge(df_net, how = 'inner')
        df = df.iloc[:, 2:]
        
        
    def update_network(new_ID, df_net, ID_match):
        # merge existing links
        df = new_ID.merge(df_net, how = 'inner')
        df = df.iloc[:, 2:]
        
        #     # Duplicates(Case I)
        #     print('Duplicates Case I: 332-333 (3 duplicates); 333-332 (3 duplicates)')
        #     print(f'{df[df.duplicated()]}\n')
        #     df = df.drop_duplicates() 
        
        #     # Duplicates (Case II)
        #     print('Duplicates Case II: different length(ft) 117-93 and 93-117')
        #     print(f'{df[(df["From"] == 93) & (df["To"] == 117)]}, {df[(df["From"] == 93) & (df["To"] == 117)].index}\n')
        #     print(f'{df[(df["From"] == 117) & (df["To"] == 93)]}, {df[(df["From"] == 117) & (df["To"] == 93)].index}\n')
        #     df = df.drop([df[(df["From"] == 93) & (df["To"] == 117)].index[1]])
        #     df = df.drop([df[(df["From"] == 117) & (df["To"] == 93)].index[1]])      
                  
        #     df.to_csv('check_df_net_1271.csv')
    
        #Check capacity and speed for centroid connectors of zones
    
        #Repace capacity for centroid connectors of zones with 5000
        df.loc[(df["From"] >= 2) & (df["From"] <= 110), 'Capacity'] = 5000
        #Repace speed for centroid connectors of zones with 70mph
        df.loc[(df["From"] >= 2) & (df["From"] <= 110), 'u_f (mph)'] = 70
    
        print(df[(df["From"] >= 2) & (df["From"] <= 110)])
        
        # add links from safe nodes to sink node
        safe_nodes_nID = ID_match[ID_match['COUNTY'] == 'Safe Node']
        for i in range(len(safe_nodes_nID)):  
            nrow = pd.DataFrame({'From':safe_nodes_nID.iloc[i, 0], 'To': 1, 'Capacity': 360000, 'Length (ft)': 10, 'u_f (mph)':70, 'k_j (veh/mi)': 40000}, index=[0])
            df = pd.concat([nrow,df])  
            
        # re-order based on From and To columns
        df = df.sort_values(['From', 'To']).reset_index(drop=True)
              
        print(f'network size {df.shape}')
        
        return df
    
    def write_net_file(zones, ID, network, net_out):  
        with open (net_out, 'w') as f:
            f.write(f"<NUMBER OF ZONES> {zones.shape[0]}\n") 
            f.write(f"<NUMBER OF NODES> {ID.shape[0]}\n")
            f.write(f"<NUMBER OF LINKS> {network.shape[0]}\n")
            f.write('<END OF METADATA>\n')
            f.write(f'\n')
            
            f.write('~ From To Capacity   Length (ft)  u_f (mph)  k_j (veh/mi)\n')
            for i in range(len(network)):
                f.write(f'  {network.iloc[i,0]}  {network.iloc[i,1]}  {network.iloc[i,2]}  {network.iloc[i,3]}  {network.iloc[i,4]}  {network.iloc[i,5]} ;\n')
    
    def write_nxy_file(houston_ID, nodes_nxy, out_nxy):
        # Input nodes file
        nodes_nxy = nodes_nxy.iloc[:, 0:3]
        nodes_nxy = nodes_nxy.rename(columns = {'Node' : 'ID'})
            
        # Merge
        df = nodes_nxy.merge(houston_ID, how = 'inner')
        
        
        # Add sink node 
        sink_node = pd.DataFrame({'New_ID': [1], 'X': [0], 'Y':[0] })
        df = pd.concat([sink_node, df]).reset_index(drop = True)
        
        # Sort ID 
        df = df.loc[:, ['New_ID', 'X', 'Y']].sort_values(['New_ID'])
        
        # Write to .nxy file
        df.to_csv(out_nxy, header = None, index = None, sep = ' ') # no header 
        print(df.shape)
    
    
    def write_ods_file(zones, scenario_file, out_file):
        num_zones  = zones.shape[0]
        
        scenario = pd.read_excel(scenario_file)
        df = zones.merge(scenario, how = 'left', on = 'COUNTY' )
        df['Evacuation Residents'] = df['Evacuation Residents'].fillna(0)
        df = df[(df['COUNTY'] != 'Interchange') & (df['COUNTY'] != 'Safe Node') & (df['COUNTY'] != 'Sink Node')]
        
        which_ods = scenario_file.split('.')[0][-1]
        
        with open(out_file, 'w') as f:
            f.write('<END OF METADATA>\n')
            f.write('\n')
            
            f.write(f'~ Number of sink node: Node 1 \n' 
              f'~ Number of origins (counties): {num_zones-1} (Node 2-110)\n')
            f.write('\n')
    
            for i in range(df.shape[0]):
                f.write(f'Origin {df.iloc[i, 0]} ~ {df.iloc[i, 2]} to the safe node\n')
                f.write(f'1: {df.iloc[i, 3]}\n')
    
                f.write('\n')
    
    def create_graph(df):
        graph = {}
        for i in range(len(df)):
            if df.iloc[i, 1] not in graph.keys(): 
                graph[df.iloc[i, 1]] = [df.iloc[i, 0]]
            else:
                graph[df.iloc[i, 1]].append(df.iloc[i, 0])
        return graph
    
    def graph_bfs(graph, visted, node):
        visited = [node]
        queue = [node]
        
        while queue:
            s = queue.pop(0)
            print (s, end = " ") 
            
            for neighbor in graph[s]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.append(neighbor)
        return visited
        
    def zones_not_connected(queue, zones):
        unconnect_zones = []
        for i in range(len(zones)):
            if zones.iloc[i, 0] not in queue:
                unconnect_zones.append(zones.iloc[i, 0])
        return unconnect_zones
    '''    
    #Input
    demand_file1 = 'GIS/Demand/demand_scenario1.xlsx'
    demand_file2 = 'GIS/Demand/demand_scenario2.xlsx'
    GIS_nodes = 'GIS/GIS_nodes.xlsx'
    GIS_net = 'GIS/GIS_net.xlsx'
    
    #Output
    raw_link_file = 'GIS/Temp/Link1209_Base.csv' 
    node_file = 'GIS/Temp/Nodes1209_Base.csv'
    county_file = 'GIS/Temp/County_references.csv'
    safe_node_file = 'GIS/Temp/safe_nodes.csv'
    
    
    #output for DTA
    demand_out1 = 'GIS/Temp/houston_input1.ods'
    demand_out2 = 'GIS/Temp/houston_input2.ods'
    out_nxy = 'GIS/Temp/houston_input.nxy'
    net_out = 'GIS/Temp/houston_input.net'
    links_upd_out = 'GIS/Temp/df_links_upd.csv'
    dup_out = 'GIS/Temp/duplicated_links.csv'
    ID_out = 'GIS/Temp/houston_ID.csv'
    '''
    GIS_nodes = pd.read_excel(GIS_nodes, header=0)
    GIS_net = pd.read_excel(GIS_net, header=0)
    
    GISsafe_nodes = GIS_nodes[GIS_nodes['Safe']==1]["ID"]
    GISsafe_nodes.to_csv(safe_node_file, index=False, header=False)
    
    GIScounty_nodes = GIS_nodes[GIS_nodes['CNTY01']==1][["ID", "COUNTY"]]
    GIScounty_nodes.to_csv(county_file, index=False)
    
    
    
    GISall_nodes = GIS_nodes[["ID", "LON_X", "LAT_Y"]]
    GISall_nodes = GISall_nodes.rename(columns={"ID": "Node", "LON_X": "X", "LAT_Y": "Y"})
    GISall_nodes.to_csv(node_file, index=False)
    
    
    
    GIS_net["CAPACITY_forward"] = GIS_net["CAPACITY_1"] + GIS_net["CAPACITY_3"]
    GIS_net["CAPACITY_reverse"] = GIS_net["CAPACITY_2"] + GIS_net["CAPACITY_4"]
    GIS_net["K_J_forward"] = GIS_net["K_J_AB_ADD"] + GIS_net["K_J_AB_TOL"]
    GIS_net["K_J_reverse"] = GIS_net["K_J_BA_ADD"] + GIS_net["K_J_BA_TOL"]
    GIS_net["SPD"] = GIS_net[["SPD_LMT", "SPEED_TOLL"]].max(axis=1, skipna=True)
    
    GIS_net1 = GIS_net[["FROM_ID", "TO_ID", "SPD", "CAPACITY_forward", "LENGTH_FT", "K_J_forward"]]
    GIS_net1 = GIS_net1.rename(columns={"FROM_ID": "Init_node", "TO_ID": "Term_node", "SPD": "u_f", "CAPACITY_forward": "Capacity", "LENGTH_FT": "Length(ft)", "K_J_forward": "k_j"})
    GIS_net2 = GIS_net[["TO_ID", "FROM_ID", "SPD", "CAPACITY_reverse", "LENGTH_FT", "K_J_reverse"]]
    GIS_net2 = GIS_net2.rename(columns={"TO_ID": "Init_node", "FROM_ID": "Term_node", "SPD": "u_f", "CAPACITY_reverse": "Capacity", "LENGTH_FT": "Length(ft)", "K_J_reverse": "k_j"})
    
    #GIS_net_all = GIS_net1.append(GIS_net2, ignore_index=True)
    GIS_net_all = pd.concat([GIS_net1, GIS_net2], ignore_index=True)
    GIS_net_all = GIS_net_all[GIS_net_all['Capacity']!=0]
    
    GIS_net_all.to_csv(raw_link_file, index=False)
    
    
    nodes_nxy = pd.read_csv(node_file, header=0)
    nodes_not_in_network = nodes_not_in_network(raw_link_file, nodes_nxy)
    nodes_nxy = rmv_nodes_not_in_network(nodes_not_in_network, nodes_nxy) 
    houston_zones = create_zones(county_file)
    all_interchanges = create_interchanges(houston_zones, nodes_nxy)
    houston_interchanges = update_safe_nodes(all_interchanges, safe_node_file)
    houston_ID = concat_ID(houston_zones, houston_interchanges)
    houston_ID.to_csv(ID_out)
    df_links = input_network_links(raw_link_file)
    links_ID_match = merge_ID(df_links, houston_ID, dup_out)
    df_links_upd = update_network(links_ID_match, df_links, houston_ID)
    df_links_upd.to_csv(links_upd_out)
    
    write_net_file(houston_zones, houston_ID, df_links_upd, net_out)
    
    print(df_links_upd.shape)
    df_links_upd.head(5)
    
    # Links from safe nodes to sink node
    print(df_links_upd[df_links_upd['To'] == 1].shape)
    df_links_upd[df_links_upd['To'] == 1]  
    
    
    write_nxy_file(houston_ID, nodes_nxy, out_nxy)
    
    
    write_ods_file(houston_zones, demand_file1, demand_out1)
    write_ods_file(houston_zones, demand_file2, demand_out2)
    
    
    merge = pd.merge(GIScounty_nodes, houston_ID, on='ID')
    County_map = merge[['ID', 'New_ID']].set_index('ID').T.to_dict('list')
    
    with open(OUT_FILE, 'wb') as f:
        pickle.dump(County_map, f)
