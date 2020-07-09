# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:21:56 2019

@author: saimunikoti
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import FormatStrFormatter
from numpy import linalg as la

###### robusness of the network
class randomattack():
    def __init__(self):
        print("random attack class is invoked")
        from .Robustness_HFGT_Main import new_resmetric
        self.resoutput = new_resmetric()

#%% complete node attack
    def nodeatkcomplete_sim(self, graph_org, n, acceptable_ratio):
        
        def get_completenodeattack(graph_org, acceptable_ratio):
            
            criticaln =0
            criticallcc=0
            sf2 = 0
            efr2 = 0
            ncc2flag = 0
            graph = graph_org.copy()

            tempcc= [len(xind) for xind in nx.weakly_connected_components(graph)]
            lcc = [max(tempcc)] # largest connected component
            ncc = [len(tempcc)] # no. of connected comp.
            
            sf  = [self.resoutput.get_servicefactor(graph,  weightflag=None)] # service factor
            sfwt  = [self.resoutput.get_servicefactor(graph,  weightflag="weight")] # service factor
            efr = [self.resoutput.get_edgerobustness(graph, weightflag=None)] # edge flow robust
            efrwt = [self.resoutput.get_edgerobustness(graph, weightflag=True)] # edge flow robust
            idr = [self.resoutput.indegree_robust(graph)] # indegree robustness metric
            # egr = [self.resoutput.get_egr(graph)] # effective graph conductance
            # ncr = [self.resoutput.get_weightedeff(graph)]  # network criticality
            countstages=0
            while(len(graph.nodes)>2):
                #plot for gif
                frames=True
                if frames == True:
                    plt.figure(len(lcc))
                    pos = nx.circular_layout(graph)
                    d = dict(graph.degree)
                    nodesplt = nx.draw_networkx_nodes(graph, pos, with_labels=True, node_size=[v * 100 for v in d.values()],
                                                      node_color="lightgreen")
                    labels = nx.draw_networkx_labels(graph, pos)

                    nx.draw_networkx_edges(graph, pos, edge_color='black', width=1, arrowstyle='->', arrowsize=15)
                    labels = nx.get_edge_attributes(graph, 'weight')
                    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
                    countfig = len(lcc)
                    plt.savefig(r'C:\Users\saimunikoti\Manifestation\Scn_Interdependency\Draft\GIF_Comp_Partial\CompRandom_' + str(
                        countfig) + '.jpg', format='jpg')  # save the figure to file
                    plt.close('all')

                nodeselected = np.random.choice(graph.nodes())
                recchildnodes = list(nx.dfs_preorder_nodes(graph, nodeselected)) # childnodes upto the end point 
                recchildnodes.remove(nodeselected)
                
                ##### node removal at primary level
                graph.remove_node(nodeselected) 
                print("noderemoval", countstages, nodeselected)
                #### effect of input on output- transitive property
                
                for childnode in recchildnodes:
                    indeg = graph.in_degree(childnode, weight="newweight")
                    # outdeg = graph.out_degree(childnode, weight="newweight")
                    
                    ## partial attack on the childnodes (find degradation ratio)         
                    try:
                        degradation_ratio= indeg/graph.nodes[childnode]['indegwt']
                    except:
                        continue # considering those nodes which have no indegree, i.e., they are not dependent on any other process.
                    
                    ## reduce the weight of all out edges of childnode if input weight varies
                    for (nodest, nodeed) in graph.edges(childnode):
                        graph[nodest][nodeed]['newweight'] =  degradation_ratio*(graph[nodest][nodeed]['newweight'])
                   
                    ##### node removal at secondary level
                    outdeg = graph.out_degree(childnode, weight="newweight")
                    if (indeg ==0) or (outdeg <= acceptable_ratio*graph.out_degree(childnode, weight="weight")):
                        graph.remove_node(childnode)
                        print("childnode", countstages,childnode)
                #### collecting metrics   
                tempcc = [len(xind) for xind in nx.weakly_connected_components(graph)]
                try:
                    lcc.append(max(tempcc))
                except :
                    break
                                    
                ncc.append(len(tempcc))
                tempidr = self.resoutput.indegree_robust(graph)
                idr.append(tempidr) 
                # tempegr = self.resoutput.get_egr(graph)
                # tempncr = self.resoutput.get_weightedeff(graph)
                # egr.append(tempegr)
                # ncr.append(tempncr)

                if len(graph.edges())>1:
                    tempsf = self.resoutput.get_servicefactor(graph, weightflag=None)
                    tempsfwt = self.resoutput.get_servicefactor(graph, weightflag="weight")
                    tempefr = self.resoutput.get_edgerobustness(graph, weightflag=None)
                    tempefrwt = self.resoutput.get_edgerobustness(graph, weightflag=True)
                    
                sf.append(tempsf)
                sfwt.append(tempsfwt)
                efr.append(tempefr)
                efrwt.append(tempefrwt)
                
                ####### critical values when graph gets disconnected into 2 components first time
                if len(tempcc)>1 and ncc2flag==0:
                    ncc2flag =1
                    crncc = len(tempcc)
                    criticaln = len(lcc)
                    criticallcc = max(tempcc)/110
                    sf2 = tempsfwt
                    efr2 = tempefrwt
                    cridr = tempidr
                    # cregr = tempegr
                    # crncr = tempncr
                countstages = countstages +1

            lcc[:] = [x / 110 for x in lcc]
            
            return lcc, ncc,sf,sfwt,efr, efrwt,criticaln, criticallcc ,sf2, efr2, crncc,cridr, idr
        
        Lcc=[]
        Ncc=[]
        Sf= []
        Sfwt=[]
        Efr=[]
        Efrwt=[]
        Expcrn=[] # critical n for disconnected average over multiple times
        Expcrlcc=[] # critical size of LCC when  disconnected average over multiple times
        Expcrsf=[] # critical sf when disconnected, average over multiple times
        Expcrefr=[] # critical efr
        Expn=[]
        Crncc = []
        Idr = []
        Cridr = []
        # Egr =[]
        # Cregr = []
        # Ncr=[]
        # Crncr=[]
        
        for countrun in range(n):
            print("Iteration No", countrun)
            lcc,ncc,sf,sfwt,efr,efrwt, criticaln, criticallcc,sf2,efr2, crncc, cridr, idr = get_completenodeattack(graph_org, acceptable_ratio)
            
            Lcc.append(lcc)
            Ncc.append(ncc)
            Sf.append(sf)
            Sfwt.append(sfwt)
            Efr.append(efr)
            Efrwt.append(efrwt)
            Expcrn.append(criticaln)
            Expcrlcc.append(criticallcc)
            Expcrsf.append(sf2)
            Expcrefr.append(efr2)
            Expn.append(len(ncc))
            Crncc.append(crncc)
            Idr.append(idr)
            Cridr.append(cridr)
            # Egr.append(egr)
            # Cregr.append(cregr)
            # Crncr.append(crncr)
            # Ncr.append(ncr)

        pad = len(max(Lcc, key=len)) 
                
        Lcc = np.mean(np.array([i + [0]*(pad-len(i)) for i in Lcc]), axis=0)
        Ncc = np.mean(np.array([i + [0]*(pad-len(i)) for i in Ncc]), axis=0)
        Sf = np.mean(np.array([i + [0]*(pad-len(i)) for i in Sf]), axis=0)
        Sfwt = np.mean(np.array([i + [0]*(pad-len(i)) for i in Sfwt]), axis=0)
        Efr = np.mean(np.array([i + [0]*(pad-len(i)) for i in Efr]), axis=0)
        Efrwt = np.mean(np.array([i + [0]*(pad-len(i)) for i in Efrwt]), axis=0)
        Idr = np.mean(np.array([i + [0]*(pad-len(i)) for i in Idr]), axis=0)
        # Egr = np.max(np.array([i + [0]*(pad-len(i)) for i in Egr]), axis=0)
        # Ncr= np.max(np.array([i + [0]*(pad-len(i)) for i in Ncr]), axis=0)
        Expcrn = np.mean(np.array(Expcrn))
        Expcrlcc = np.mean(np.array(Expcrlcc))
        Expcrsf = np.mean(np.array(Expcrsf))
        Expcrefr = np.mean(np.array(Expcrefr))
        Expn = np.mean(Expn)
        Crncc = np.mean(Crncc)
        Cridr = np.mean(Cridr)

        # Cregr = np.mean(Cregr)
        # Crncr = np.mean(Crncr)

        return Lcc,Ncc,Sf,Sfwt, Efr, Efrwt,Expcrn, Expcrlcc, Expcrsf, Expcrefr, Expn , Crncc, Cridr, Idr


    def nodeatkpartial_sim(self, graph_org, degradation_amount, n, acceptable_ratio):
        
        def get_partialnodeattack(graph_org, degradation_amount, acceptable_ratio):
            
            criticaln =0
            criticallcc=0
            sf2 = 0
            efr2 = 0
            ncc2flag = 0
            graph = graph_org.copy()

            tempcc= [len(xind) for xind in nx.weakly_connected_components(graph)]
            lcc = [max(tempcc)] # largest connected component
            ncc = [len(tempcc)] # no. of connected comp.
            
            sf  = [self.resoutput.get_servicefactor(graph,  weightflag=None)] # service factor
            sfwt  = [self.resoutput.get_servicefactor(graph,  weightflag="weight")] # service factor
            efr = [self.resoutput.get_edgerobustness(graph, weightflag=None)] # edge flow robust
            efrwt = [self.resoutput.get_edgerobustness(graph, weightflag=True)] # edge flow robust
            idr = [self.resoutput.indegree_robust(graph)]
            # ecr = [self.resoutput.get_weightedeff(graph)] # component robustness
            countstages=0
            while(len(graph.nodes)>2):
#                print("new while loop")
                nodeselected = np.random.choice(graph.nodes())
                recchildnodes = list(nx.dfs_preorder_nodes(graph, nodeselected)) # childnodes upto the end point 
                recchildnodes.remove(nodeselected)
                
                ##### node removal at primary level
                for (nodest, nodeed) in graph.edges(nodeselected):
                    # graph[nodest][nodeed]['newweight'] = graph[nodest][nodeed]['newweight'] - (np.round(random.uniform(0.1,0.5),2))
                    graph[nodest][nodeed]['newweight'] = graph[nodest][nodeed]['newweight'] - min(degradation_amount, graph[nodest][nodeed]['newweight'])

                if graph.out_degree(nodeselected, weight="newweight") <= acceptable_ratio*graph.out_degree(nodeselected, weight="weight") :
                    # print("noderemoval", countstages, nodeselected)
                    graph.remove_node(nodeselected) 
                    
                                 
                #### effect of input on output- transitive property
                
                for childnode in recchildnodes:
                    indeg = graph.in_degree(childnode, weight="newweight")
                    # outdeg = graph.out_degree(childnode, weight="newweight")
                    
                    ## partial attack on the childnodes (find degradation ratio)         
                    try:
                        degradation_ratio= indeg/graph.nodes[childnode]['indegwt']
                    except:
                        continue # considering those nodes which have no indegree, i.e., they are not dependent on any other process.
                    
                    ## reduce the weight of all out edges of childnode if input weight varies
                    for (nodest, nodeed) in graph.edges(childnode):
                        graph[nodest][nodeed]['newweight'] =  degradation_ratio*(graph[nodest][nodeed]['newweight'])
                   
                    ##### node removal at secondary level
                    outdeg = graph.out_degree(childnode, weight="newweight")
                    if (indeg ==0) or (outdeg <= acceptable_ratio*graph.out_degree(childnode, weight="weight")):
                        graph.remove_node(childnode)
                        # print("childnoderemove",countstages, childnode)
                    
                #### collecting metrics   
                tempcc = [len(xind) for xind in nx.weakly_connected_components(graph)]
                try:
                    lcc.append(max(tempcc))
                except :
                    break
                                    
                ncc.append(len(tempcc))
                tempidr = self.resoutput.indegree_robust(graph)
                # tempecr = self.resoutput.get_weightedeff(graph)
                idr.append(tempidr)
                # ecr.append(tempecr)

                if len(graph.edges())>1:
                    tempsf = self.resoutput.get_servicefactor(graph, weightflag=None)
                    tempsfwt = self.resoutput.get_servicefactor(graph, weightflag="weight")
                    tempefr = self.resoutput.get_edgerobustness(graph, weightflag=None)
                    tempefrwt = self.resoutput.get_edgerobustness(graph, weightflag=True)
                    
                sf.append(tempsf)
                sfwt.append(tempsfwt)
                efr.append(tempefr)
                efrwt.append(tempefrwt)
                
                ####### critical values when graph gets disconnected into 2 components first time
                if len(tempcc)>1 and ncc2flag==0:
                    ncc2flag =1
                    crncc = len(tempcc)
                    criticaln = len(lcc)
                    criticallcc = max(tempcc)/110
                    sf2 = tempsfwt
                    efr2 = tempefrwt
                    cridr = tempidr
                countstages = countstages +1

            lcc[:] = [x / 110 for x in lcc]   
            
            return lcc,ncc,sf,sfwt,efr, efrwt,criticaln, criticallcc , sf2, efr2, crncc, cridr, idr
        
        Lcc=[]
        Ncc=[]
        Sf= []
        Sfwt=[]
        Efr=[]
        Efrwt=[]
        Expcrn=[] # critical n for disconnected average over multiple times
        Expcrlcc=[] # critical size of LCC when  disconnected average over multiple times
        Expcrsf=[] # critical sf when disconnected, average over multiple times
        Expcrefr=[] # critical efr
        Expn=[]
        Crncc=[]
        Cridr=[]
        Idr=[]


        for countrun in range(n):
            print("### Iteration No $$$$", countrun)
            lcc,ncc,sf,sfwt,efr,efrwt, criticaln, criticallcc,sf2,efr2, crncc, cridr, idr = \
                get_partialnodeattack(graph_org, degradation_amount, acceptable_ratio)
            
            Lcc.append(lcc)
            print("countsim", countrun, lcc)
            Ncc.append(ncc)
            Sf.append(sf)
            Sfwt.append(sfwt)
            Efr.append(efr)
            Efrwt.append(efrwt)
            Expcrn.append(criticaln)
            Expcrlcc.append(criticallcc)
            Expcrsf.append(sf2)
            Expcrefr.append(efr2)
            Expn.append(len(ncc))
            Crncc.append(crncc)
            Idr.append(idr)
            Cridr.append(cridr)

            
        pad = len(max(Lcc, key=len)) 
                
        Lcc = np.mean(np.array([i + [0]*(pad-len(i)) for i in Lcc]), axis=0)
        Ncc = np.mean(np.array([i + [0]*(pad-len(i)) for i in Ncc]), axis=0)
        Sf = np.mean(np.array([i + [0]*(pad-len(i)) for i in Sf]), axis=0)
        Sfwt = np.mean(np.array([i + [0]*(pad-len(i)) for i in Sfwt]), axis=0)
        Efr = np.mean(np.array([i + [0]*(pad-len(i)) for i in Efr]), axis=0)
        Efrwt = np.mean(np.array([i + [0]*(pad-len(i)) for i in Efrwt]), axis=0)
        Idr = np.mean(np.array([i + [0]*(pad-len(i)) for i in Idr]), axis=0)

        Expcrn = np.mean(np.array(Expcrn))
        Expcrlcc = np.mean(np.array(Expcrlcc))
        Expcrsf = np.mean(np.array(Expcrsf))
        Expcrefr = np.mean(np.array(Expcrefr))
        Expn = np.mean(Expn)
        Crncc = np.mean(Crncc)
        Cridr = np.mean(Cridr)

        return Lcc,Ncc,Sf,Sfwt,Efr, Efrwt, Expcrn, Expcrlcc, Expcrsf, Expcrefr, Expn, Crncc, Cridr, Idr
    
##%% Target attack simulations
        
class targetattack():
    def __init__(self):
        print("target class is invoked")
        from src.models.Robustness_HFGT_Main import new_resmetric
        self.resoutput = new_resmetric()       
        
#%%###### Node attack sim which captures the whole trajectory of percolation    
    def completenodeatk(self, graph_org, centrality, wt, acceptable_ratio, frames = None):
        graph = graph_org.copy()
        ncc2flag =0
        criticaln =0
        criticallcc=0

        sf2 = 0
        efr2 = 0     
        
        ###### initilization
        tempcc= [len(xind) for xind in nx.weakly_connected_components(graph)]
        lcc = [max(tempcc)] # largest connected component
        ncc = [len(tempcc)] # no. of connected comp.
        acc = tempcc # all connected com
        
        sf  = [self.resoutput.get_servicefactor(graph, weightflag="weight")] # service factor
        efr = [self.resoutput.get_edgerobustness(graph, weightflag=True)] # edge flow robust
        idr = [self.resoutput.indegree_robust(graph)] # edge flow robust
        countstages=0
        while(len(graph.nodes)>2):
            
            ########### plot for gif
            if frames==True:
                plt.figure(len(lcc))
                pos = nx.circular_layout(graph)
                d = dict(graph.degree)
                nodesplt = nx.draw_networkx_nodes(graph, pos, with_labels=True,  node_size=[v * 100 for v in d.values()], node_color="lightgreen")
                labels = nx.draw_networkx_labels(graph, pos)

                nx.draw_networkx_edges(graph, pos, edge_color='black', width=1, arrowstyle='->', arrowsize=15)
                labels = nx.get_edge_attributes(graph,'weight')
                nx.draw_networkx_edge_labels(graph ,pos,edge_labels=labels)
                countfig = len(lcc)
                plt.savefig(r'C:\Users\saimunikoti\Manifestation\Scn_Interdependency\Draft\GIF_Comp_Partial\CompTarget_'+ str(countfig)+'.eps', format='eps')  # save the figure to file
                plt.close('all')
            ###########################

            if centrality =="out degree":
                nodeimpscore = dict(graph.out_degree(weight=wt))
            else:
                nodeimpscore = nx.betweenness_centrality(graph, weight=wt)
            
            nodeselected = max(nodeimpscore, key=nodeimpscore.get)
            recchildnodes = list(nx.dfs_preorder_nodes(graph, nodeselected))
            recchildnodes.remove(nodeselected)
            
            ##### node removal at primary level
            graph.remove_node(nodeselected) 
            # print("noderemoval",countstages, nodeselected)

            #### effect of input on output- transitive property
            
            for childnode in recchildnodes:
                indeg = graph.in_degree(childnode, weight="newweight")
                # outdeg = graph.out_degree(childnode, weight="newweight")
                
                ## partial attack on the childnodes (find degradation ratio)         
                try:
                    degradation_ratio= indeg/graph.nodes[childnode]['indegwt']
                except:
                    continue # considering those nodes which have no indegree, i.e., they are not dependent on any other process.
                
                ## reduce the weight of all out edges of childnode if input weight varies
                for (nodest, nodeed) in graph.edges(childnode):
                    graph[nodest][nodeed]['newweight'] =  degradation_ratio*(graph[nodest][nodeed]['newweight'])

                outdeg = graph.out_degree(childnode, weight="newweight")
                ##### node removal at secondary level
                if (indeg ==0) or (outdeg <= acceptable_ratio*graph.out_degree(childnode, weight="weight")):
                    graph.remove_node(childnode)
                    # print("childnoderem", countstages, childnode)

            ### collect metrics
            tempcc = [len(xind) for xind in nx.weakly_connected_components(graph)]
            lcc.append(max(tempcc))
            ncc.append(len(tempcc))
            
            acc.append(tempcc)
                      
            if len(graph.edges())>1:           
                tempsf = self.resoutput.get_servicefactor(graph, weightflag="weight")

            tempefr = self.resoutput.get_edgerobustness(graph, weightflag=True)    
            tempidr = self.resoutput.indegree_robust(graph)

            sf.append(tempsf)    
            efr.append(tempefr)
            idr.append(tempidr)

            ####### critical values when graph gets disconnected into 2 components first time
            if len(tempcc)>1 and ncc2flag==0:
                ncc2flag =1
                crnoofcc = len(tempcc)
                criticaln = len(lcc)
                criticallcc = max(tempcc)/110
                sf2 = tempsf
                efr2 = tempefr
                cridr = tempidr

            countstages = countstages+1

        lcc[:] = [x / 110 for x in lcc]

        return lcc,ncc,sf,efr ,criticaln, criticallcc, sf2, efr2, crnoofcc, cridr, idr, acc
    
    def partialnodeattack(self, graph_org, degradation_amount, centrality, wt, acceptable_ratio, allframes=None):
        graph = graph_org.copy()
        ncc2flag =0
        criticaln =0
        criticallcc=0
        sf2 = 0
        efr2 = 0

        ###### initilization
        tempcc= [len(xind) for xind in nx.weakly_connected_components(graph)]
        lcc = [max(tempcc)] # largest connected component
        ncc = [len(tempcc)] # no. of connected comp.
        acc = tempcc  # all connected com
        sf  = [self.resoutput.get_servicefactor(graph, weightflag="weight")] # service factor
        efr = [self.resoutput.get_edgerobustness(graph, weightflag=True)] # edge flow robust
        idr = [self.resoutput.indegree_robust(graph)]
        countstages = 0

        while(len(graph.nodes)>2):

            if allframes==True:
                ########## plot for gif
                plt.figure(len(lcc))
                pos = nx.circular_layout(graph)
                d = dict(graph.degree)
                nodesplt = nx.draw_networkx_nodes(graph, pos, with_labels=True, node_size=[v * 100 for v in d.values()],
                                                  node_color ="lightgreen")
                labels = nx.draw_networkx_labels(graph, pos)
                nx.draw_networkx_edges(graph, pos, edge_color='black', width=1, arrowstyle='->', arrowsize=15)
                labels = nx.get_edge_attributes(graph, 'weight')
                nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
                countfig = len(lcc)
                # print("countfignodes",countfig, graph.nodes())
                plt.savefig(r'C:\Users\saimunikoti\Manifestation\Scn_Interdependency\Draft\GIF_Comp_Partial\PartialTarget_' + str(countfig)+'.eps', format='eps')  # save the figure to file
                plt.close('all')

            if centrality =="out degree":
                nodeimpscore = dict(graph.out_degree(weight='weight'))
            else:
                nodeimpscore = nx.betweenness_centrality(graph, weight='weight')

            nodeselected = max(nodeimpscore, key=nodeimpscore.get)
#                print(nodeselected)
            recchildnodes = list(nx.dfs_preorder_nodes(graph, nodeselected))
            recchildnodes.remove(nodeselected)

            ##### node removal at primary level
            for (nodest, nodeed) in graph.edges(nodeselected):
                # graph[nodest][nodeed]['newweight'] = graph[nodest][nodeed]['newweight'] - (np.round(random.uniform(0.1,0.4),2))
                graph[nodest][nodeed]['newweight'] = graph[nodest][nodeed]['newweight'] - min(degradation_amount, graph[nodest][nodeed]['newweight'])

            if graph.out_degree(nodeselected, weight="newweight") <= acceptable_ratio*graph.out_degree(nodeselected, weight="weight") :
                graph.remove_node(nodeselected)
                # print('noderemoval', countstages, nodeselected)

            #### effect of input on output- transitive property

            for childnode in recchildnodes:
                indeg = graph.in_degree(childnode, weight="newweight")
                # outdeg = graph.out_degree(childnode, weight="newweight")

                ## partial attack on the childnodes (find degradation ratio)
                try:
                    degradation_ratio= indeg/graph.nodes[childnode]['indegwt']
                except:
                    continue # considering those nodes which have no indegree, i.e., they are not dependent on any other process.

                ## reduce the weight of all out edges of childnode if input weight varies
                for (nodest, nodeed) in graph.edges(childnode):
                    graph[nodest][nodeed]['newweight'] =  degradation_ratio*(graph[nodest][nodeed]['newweight'])

                outdeg = graph.out_degree(childnode, weight="newweight")
                ##### node removal at secondary level
                if (indeg ==0) or (outdeg <= acceptable_ratio*graph.out_degree(childnode, weight="weight")):
                    graph.remove_node(childnode)
                    # print("childnoderem", countstages, childnode)

            ### collect metrics
            tempcc = [len(xind) for xind in nx.weakly_connected_components(graph)]
            lcc.append(max(tempcc))
            ncc.append(len(tempcc))
            acc.append(tempcc)  # all connected com

            if len(graph.edges())>1:
                tempefr = self.resoutput.get_edgerobustness(graph, weightflag=True)
                tempsf = self.resoutput.get_servicefactor(graph, weightflag="weight")
                tempidr = self.resoutput.indegree_robust(graph)

            sf.append(tempsf)
            efr.append(tempefr)
            idr.append(tempidr)

            ####### critical values when graph gets disconnected into 2 components first time
            if len(tempcc)>1 and ncc2flag==0:
                ncc2flag =1
                crnoofcc = len(tempcc)
                criticaln = len(lcc)
                criticallcc = max(tempcc)/110
                sf2 = tempsf
                efr2 = tempefr
                cridr = tempidr
            countstages = countstages+1

        lcc[:] = [x / 110 for x in lcc]

        return lcc,ncc,sf,efr ,criticaln, criticallcc, sf2, efr2, crnoofcc, cridr, idr, acc
        
    ####################### get optimal weight according to targetmetric efr and lcc #########
    """ 
    targetmetric = lcc or efr
    nodevar = particular node whose incoming weioghts need top be optimize
    nsim =  no of weiight combination to search
    """
    
    def get_optimalnodeweights(self, G, nsim, nodevar,targetmetric):
        weightlist=[]
        lcclist=[]
        idrlist=[]
        efrlist=[]
        # Indegree = G.in_degree(nodevar)
        # if Indegree ==2:
        #     tempwtlist2 = [(i / 10, (10 - i) / 10) for i in range(1, 10)]
        # elif Indegree ==3:
        #     tempwtlist3 = [(i / 10, (10 - i - k) / 10, k / 10) for i in range(1, 9) for k in range(1, 10 - i)]

        # for wtcomb in tempwtlist:
        #     # collection = np.random.choice(A, nweights, replace=True)
        #     # countweight = collection/(np.sum(collection))
        #     gcopy = G.copy()
        #     for nodevar in [6,9,12,16,20,28]:
        #         for countedge,(u,v) in enumerate(gcopy.in_edges(nodevar)):
        #             gcopy[u][v]['weight'] = wtcomb[countedge]

        tempwtlist2 = [(i / 10, (10 - i) / 10) for i in range(1, 10)]
        tempwtlist3 = [(i / 10, (10 - i - k) / 10, k / 10) for i in range(1, 9) for k in range(1, 10 - i)]
        nodedict2=[]
        nodedict3=[]
        for node in G.nodes:
            if G.in_degree(nbunch=node, weight=None) == 2:
                nodedict2.append(node)
            elif G.in_degree(nbunch=node, weight=None) == 3:
                nodedict3.append(node)

        for countmcsim in range(nsim):
            gcopy = G.copy()
            wtdict={}
            for countnode in nodedict2:
                temprand = np.random.choice(np.arange(0, len(tempwtlist2)), 1)[0]
                wtcomb2 = tempwtlist2[temprand]
                for countedge,(u,v) in enumerate(gcopy.in_edges(countnode)):
                    gcopy[u][v]['weight'] = wtcomb2[countedge]
                    wtdict[str(u)] = (v, wtcomb2[countedge])

            for countnode in nodedict3:
                temprand = np.random.choice(np.arange(0, len(tempwtlist3)), 1)[0]
                wtcomb3 = tempwtlist3[temprand]
                for countedge, (u, v) in enumerate(gcopy.in_edges(countnode)):
                    gcopy[u][v]['weight'] = wtcomb3[countedge]
                    wtdict[str(u)] = (v, wtcomb3[countedge])

            Lcc_pat, Ncc_pat, Sf_pat, Efr_pat, crn_pat, crlcc_pat, crsf_pat, crefr_pat, crncc_pat, cridr_pat, Idr_pat, acc_pat = self.partialnodeattack(gcopy,0.2, "weight node betweenness" ,"weight",0.5)
            # Lcc_cot, Ncc_cot, Sf_cot, Efr_cot, crn_cot, crlcc_cot, crsf_cot, crefr_cot, crncc_cot, cridr_cot, Idr_cot = \
            #                             self.completenodeatk(gcopy, "weight node betweenness", "weight", 0.5, frames=None)

            Table_wt_complete = self.resoutput.get_countstages(Lcc_pat, Efr_pat, Idr_pat)
            # Table_wt_complete = self.resoutput.get_percentfall_resmetric(Lcc_pat, Efr_pat, Idr_pat)

            weightlist.append(wtdict)
            lcclist.append(Table_wt_complete[0,2])
            efrlist.append(Table_wt_complete[1,2])
            idrlist.append(Table_wt_complete[2,2])
            print(countmcsim)
            # print(Table_wt_complete)

        efrlist = np.array(efrlist)
        weightlist = np.array(weightlist)
        idrlist = np.array(idrlist)
        lcclist = np.array(lcclist)
            
        # if targetmetric=="efr":
        #     maxindex = np.where(efrlist==np.max(efrlist))[0]
        #     optimalval = np.max(efrlist)
        # elif targetmetric =="lcc":
        #     maxindex = np.where(lcclist==np.max(lcclist))[0]
        #     optimalval = np.max(lcclist)
        # elif targetmetric =="idr":
        #     maxindex = np.where(idrlist==np.max(idrlist))[0]
        #     optimalval = np.max(idrlist)

        maxindex1 = set(np.where(efrlist==np.max(efrlist))[0])

        maxindex2 = set(np.where(lcclist==np.max(lcclist))[0])
        maxindex2 = maxindex1.intersection(maxindex2)

        maxindex3 = set(np.where(idrlist==np.max(idrlist))[0])
        maxindex3 = list(maxindex2.intersection(maxindex3))

        optimalweight = weightlist[maxindex3]
        # optimalweight = np.round(optimalweight,1)
        # optimalweight = np.unique(optimalweight, axis=0)
        
        return optimalweight
        

## Plot robustness metric
class new_resmetric():
    
    def __init__(self):
        print("metric class is invoked")
                
#    def get_servicefactor(self, G): 
#        sumq = 0
#        
#        for (node, val) in G.degree(weight='weight'):
#            tempval = val/(G.nodes[node]['qo'])
#            G.nodes[node]['qcurrent'] = tempval
#            
#        for countn in G.nodes():
#            sumq = sumq + (G.nodes[countn]['qcurrent'])/(G.nodes[countn]['crf'])
#        
#        servicefac = sumq/(len(G.nodes))
#        
#        return servicefac
    
    def get_egr_resistancedist(self, G):
        N = len(G.nodes)
        Rg = 0
        Gund = G.to_undirected()
        nodelist = list(Gund.nodes)
        for i in range(N):
            for j in range(i+1,N):
                rab = nx.resistance_distance(Gund, nodelist[i], nodelist[j], weight='weight', invert_weight=True)
                Rg = Rg+ rab

        return (N-1)/Rg
    def get_weff(self, G):
        N = len(G)
        for u,v in G.edges:
            G[u][v]['newweight']= 1/(G[u][v]['weight'])
        sparray = np.zeros((N,N))
        nodelist = list(G.nodes)
        for i in range(N):
            for j in range(N):
                if i!=j:
                    try:
                        sparray[i][j] = nx.shortest_path_length(G,nodelist[i], nodelist[j], weight='newweight')
                    except:
                        continue
        return np.mean(sparray)

    def get_weightedeff(self, G):
        tempsum=[]
        tempsize=[]
        splength = dict(nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path_length(G, weight='weight'))
        for key in splength.keys():
            tempsum.append(sum(splength[key].values()))
            tempsize.append(len(splength[key].values()) - 1)
        try:
            weff = sum(tempsum)/sum(tempsize)
        except:
            weff = 0
        return weff

    def network_criticality(self,G):
        n = len(G.nodes)
        Gcopy = G.copy()
        Gcopy = Gcopy.to_undirected()
        # eig = np.linalg.pinv(nx.directed_laplacian_matrix(G, weight='weight'))

        eig = np.linalg.pinv(nx.laplacian_matrix(Gcopy, weight='weight'))

        ncr = np.trace(eig)
        return (2/(n-1))*ncr

    ###  normalized egr  
    def get_egr(self, G):
        Gcopy = G.copy()
        Gcopy = Gcopy.to_undirected()

        eig = nx.linalg.spectrum.laplacian_spectrum(Gcopy, weight='weight')
        n = len(G.nodes)

        # Laplacian = nx.directed_laplacian_matrix(G, weight='weight') # diercted weighted laplacian
        # eig, v = la.eig(np.squeeze(np.asarray(Laplacian)))

        try:
            eig = [(1/num) for num in eig[1:] if num != 0]
            egr = np.round(sum(np.abs(eig)), 3)
        except:
            print("zero encountered in Laplacian eigen values")

        Rg = (n-1)/(n*egr)
        
        return np.round(Rg, 3)       

    #### indegree robustness of tjhe graph - new designed metric 
    def indegree_robust(self, G):
        
        for (node, val) in G.in_degree(weight = 'weight'):  
            G.nodes[node]['indegcurnt'] = val
           
        sumindegree = 0
        for countn in G.nodes():
            try:
                sumindegree = sumindegree + (G.nodes[countn]['indegcurnt'])/(G.nodes[countn]['indegwt'])
            except:
                continue
            
        for node in G.nodes:
            orgindegreewtcount = G.nodes[node]['orgindgwtcount'] 
            break
        
        return sumindegree/orgindegreewtcount

    def component_robust(self, G, weightflag=True):
        for u in G.nodes():
            orgedgeweights= G.nodes[u]['orgedgewtcount']
            break
        try:
            tempcomp = [xind for xind in nx.weakly_connected_components(G)]
        except:
            tempcomp = [xind for xind in nx.connected_components(G)]
        sumedgecount = 0
        for countcomp in range(len(tempcomp)):
            H = G.subgraph(tempcomp[countcomp])
            if weightflag == True:
                edgecount = H.size(weight='weight')
            else:
                edgecount = H.size(weight=None)

            sumedgecount = sumedgecount + edgecount

        comprobustness = sumedgecount / orgedgeweights

        return comprobustness

    def get_servicefactor(self, G, weightflag="weight"):
        for (node, val) in G.degree(weight = weightflag):
            
            G.nodes[node]['qcurrent'] = val
        sumqnew=0 
        sumqorg=0
        if weightflag=="weight":
            
            for countn in G.nodes():
                sumqnew = sumqnew + (G.nodes[countn]['qcurrent'])*(G.nodes[countn]['crf'])
                sumqorg = sumqorg + (G.nodes[countn]['qo'])*(G.nodes[countn]['crf'])
        else:
            for countn in G.nodes():
                sumqnew = sumqnew + (G.nodes[countn]['qcurrent'])*(G.nodes[countn]['crf'])
                sumqorg = sumqorg + (G.nodes[countn]['orgunwtdegree'])*(G.nodes[countn]['crf'])
            
        return (sumqnew/sumqorg)
     
    def get_edgerobustness(self,G, weightflag=True):
        
        for u in G.nodes():
            n = G.nodes[u]['orgnodecount']
            break
        # n= len(G.nodes)
        try:
            tempcomp = [xind for xind in nx.weakly_connected_components(G)]
        except:
            tempcomp = [xind for xind in nx.connected_components(G)]
        
        sumedgecount = 0
        for countcomp in range(len(tempcomp)):
                
            H = G.subgraph(tempcomp[countcomp])
            if weightflag == True:
                # edgecount = H.size(weight='weight')
               edgecount = len(H.nodes)
            else:
                # edgecount = H.size(weight=None)
               edgecount = len(H.nodes)
            sumedgecount = sumedgecount + edgecount*(edgecount-1)

        edgerobustness = sumedgecount/(n*(n-1))
        
        return edgerobustness

    def get_countstages(self, lcc,fr,idr):
        output=np.zeros((3,3))
        for countmetric,metric in enumerate([lcc,fr,idr]):
            for countcol,residualratio in enumerate([0.8,0.5,0.2]):
                output[countmetric, countcol] = np.where(np.array(metric) < residualratio)[0][0]

        return output

    def get_percentfall_resmetric(self, lcc, efr, sf):
        
        restable = np.zeros((3,3))
        pb = [int(0.2*len(lcc)), int(0.6*len(lcc)), int(1*len(lcc))]
        startindex = [0,pb[0], pb[1]]
        endindex= [pb[0]-1, pb[1]-1, pb[2]-1]
        
        for countmetric,metric in enumerate([lcc,efr,sf]):
            
            for countcol,(startind,endind) in enumerate(zip(startindex,endindex)):
                try:
                    restable[countmetric, countcol] = metric[startind]- metric[endind] #
                    # restable[countmetric, countcol] = ((metric[startind]- metric[endind])/metric[startind])*100 # difference in percentage
                except:
                    restable[countmetric,countcol] = 0.00001
        return restable
    
    def plot_resmetric(self,lcc,ncc,sf,efr):
        
        xind = np.arange(0,len(lcc))
        
        def plot_base(tempax, y, ylabel, Colour):
            tempax.plot(xind, y, marker='o', color=Colour)
            tempax.xaxis.set_tick_params(labelsize=24)
            tempax.yaxis.set_tick_params(labelsize=24)
                    
            tempax.set_ylabel(ylabel, fontsize=24)
            tempax.grid(True)
            
        fig1, ax = plt.subplots(nrows=2, ncols=1, sharex=False,sharey=False,figsize=(8,6))
        plot_base(ax[0], lcc, "LCC","limegreen")
        plot_base(ax[1], ncc, "NCC","slateblue")
        ax[1].set_xlabel("Percolation stage", fontsize=24)
        
        fig2, ax = plt.subplots(nrows=2, ncols=1, sharex=False,sharey=False,figsize=(8,6))
        plot_base(ax[0], sf, "Service factor","coral")
        plot_base(ax[1], efr, "Edge flow robustness","dodgerblue")
        ax[1].set_xlabel("Percolation stage", fontsize=24)
    
    def plot2d_resmetric(self,lcc,ncc,sf,efr, titlename):
        
        xind = np.arange(0,len(lcc))
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True,sharey=False,figsize=(8,6))
                
        lns1 = ax[0].plot(xind, lcc, marker='o',markersize=9, color="limegreen", label="LCC", linewidth=3)
        
        ax02 = ax[0].twinx()
        
        lns2= ax02.plot(xind, ncc , marker='^', markersize=10, color="slateblue", label="NCC",linewidth=3)
        
        ax[0].set_ylabel("LCC", fontsize=26, color="limegreen", fontweight="bold")
        ax[0].tick_params(labelsize=22)
        
        ax02.set_ylabel("NCC", fontsize=26, color="slateblue", fontweight="bold")
        ax02.tick_params(labelsize=22)
        
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc=0, fontsize=20)
        ax[0].grid(True)
        ################ subplo
        lns1 = ax[1].plot(xind, sf, marker='o', markersize=9,color="coral", label="SF",linewidth=3)
                
        ax12 = ax[1].twinx()
        
        lns2 = ax12.plot(xind, efr , marker='^', markersize=10, color="dodgerblue", label="EFR", linewidth=3)
        
        ax[1].set_ylabel("SF", fontsize=26, color="coral", fontweight="bold")
        ax[1].tick_params( labelsize=22)
        ax12.set_ylabel("EFR", fontsize=26, color="dodgerblue", fontweight="bold")
        ax12.tick_params(labelsize=22)
        ax12.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[1].set_xlabel('Percolation stage', fontsize=24)
                
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax[1].legend(lns, labs, loc=0, fontsize=20)
        ax[1].grid(True)
        
        fig.tight_layout()
        plt.show()

    def plot4d_2compresmetric(self, lcccomp,lccpartial,ncccomp,nccpartial ,efrcomp,efrpartial, idrcomp, idrpartial) :
        xind = np.arange(0, len(lcccomp))

        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(8, 6))

        lns1 = ax[0,0].plot(lcccomp, marker='o', markersize=8, color="coral", label="LCC", linewidth=3)
        lns1 = ax[0,0].plot(lccpartial, marker='^', markersize=8, color="dodgerblue", label="LCC", linewidth=3)
        ax[0,0].set_ylabel("LCC", fontsize=26)
        ax[0,0].tick_params(labelsize=22)
        ax[0,0].set_xlabel('Percolation stage', fontsize=24)
        ax[0,0].legend(['random','target'], fontsize=20)

        lns2 = ax[0,1].plot(ncccomp, marker='o', markersize=8, color="coral", label="NCC", linewidth=3)
        lns2 = ax[0,1].plot(nccpartial, marker='^', markersize=8, color="dodgerblue", label="NCC", linewidth=3)
        ax[0,1].set_ylabel("NCC", fontsize=26)
        ax[0,1].tick_params(labelsize=22)
        ax[0,1].set_xlabel('Percolation stage', fontsize=24)
        ax[0,1].legend(['random','target'], fontsize=20)

        lns3 = ax[1,0].plot(efrcomp, marker='o', markersize=8, color="coral", label="egr", linewidth=3)
        lns3 = ax[1,0].plot(efrpartial, marker='^', markersize=8, color="dodgerblue", label="egr", linewidth=3)
        ax[1,0].set_ylabel("FR", fontsize=26)
        ax[1,0].tick_params(labelsize=22)
        ax[1,0].set_xlabel('Percolation stage', fontsize=24)
        ax[1,0].legend(['random','target'], fontsize=20)

        lns4 = ax[1,1].plot(idrcomp, marker='o', markersize=8, color="coral", label="idr", linewidth=3)
        lns4 = ax[1,1].plot(idrpartial, marker='^', markersize=8, color="dodgerblue", label="idr", linewidth=3)
        ax[1,1].set_ylabel("SR", fontsize=26)
        ax[1,1].tick_params(labelsize=22)
        ax[1,1].set_xlabel('Percolation stage', fontsize=24)
        ax[1,1].legend(['random','target'], fontsize=20)

        ax[0,0].grid(True)
        ax[0,1].grid(True)
        ax[1,0].grid(True)
        ax[1,1].grid(True)
    def plot4d_resmetric(self, lcccomp,lccpartial,lccoutd,ncccomp,nccpartial , nccoutd, efrcomp,efrpartial, efroutd, idrcomp, idrpartial, idroutd) :
        xind = np.arange(0, len(lcccomp))

        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(8, 6))

        lns1 = ax[0,0].plot(lcccomp, marker='o', markersize=8, color="coral", label="LCC", linewidth=3)
        lns1 = ax[0,0].plot(lccpartial, marker='^', markersize=8, color="dodgerblue", label="LCC", linewidth=3)
        lns1 = ax[0,0].plot(lccoutd, marker='P', markersize=8, color="limegreen", label="LCC", linewidth=3)
        ax[0,0].set_ylabel("LCC", fontsize=26)
        ax[0,0].tick_params(labelsize=22)
        ax[0,0].set_xlabel('Percolation stage', fontsize=24)
        ax[0,0].legend(['random','betweenness','outdegree'], fontsize=20)

        lns2 = ax[0,1].plot(ncccomp, marker='o', markersize=8, color="coral", label="NCC", linewidth=3)
        lns2 = ax[0,1].plot(nccpartial, marker='^', markersize=8, color="dodgerblue", label="NCC", linewidth=3)
        lns2 = ax[0,1].plot(nccoutd, marker='P', markersize=8, color="limegreen", label="NCC", linewidth=3)
        ax[0,1].set_ylabel("NCC", fontsize=26)
        ax[0,1].tick_params(labelsize=22)
        ax[0,1].set_xlabel('Percolation stage', fontsize=24)
        ax[0,1].legend(['random','betweenness','outdegree'], fontsize=20)

        lns3 = ax[1,0].plot(efrcomp, marker='o', markersize=8, color="coral", label="egr", linewidth=3)
        lns3 = ax[1,0].plot(efrpartial, marker='^', markersize=8, color="dodgerblue", label="egr", linewidth=3)
        lns3 = ax[1,0].plot(efroutd, marker='P', markersize=8, color="limegreen", label="egr", linewidth=3)
        ax[1,0].set_ylabel("FR", fontsize=26)
        ax[1,0].tick_params(labelsize=22)
        ax[1,0].set_xlabel('Percolation stage', fontsize=24)
        ax[1,0].legend(['random','betweenness','outdegree'], fontsize=20)

        lns4 = ax[1,1].plot(idrcomp, marker='o', markersize=8, color="coral", label="idr", linewidth=3)
        lns4 = ax[1,1].plot(idrpartial, marker='^', markersize=8, color="dodgerblue", label="idr", linewidth=3)
        lns4 = ax[1,1].plot(idroutd, marker='P', markersize=8, color="limegreen", label="idr", linewidth=3)
        ax[1,1].set_ylabel("SR", fontsize=26)
        ax[1,1].tick_params(labelsize=22)
        ax[1,1].set_xlabel('Percolation stage', fontsize=24)
        ax[1,1].legend(['random','betweenness','outdegree'], fontsize=20)

        ax[0,0].grid(True)
        ax[0,1].grid(True)
        ax[1,0].grid(True)
        ax[1,1].grid(True)

    def plot2d_resmetric_3plots(self,lcc,ncc,sf,sfwt,efr,efrwt,titlename, nplots=3):
        
        xind = np.arange(0,len(lcc))
        fig, ax = plt.subplots(nrows=nplots, ncols=1, sharex=True,sharey=False,figsize=(8,6))
                
        lns1 = ax[0].plot(xind, lcc, marker='o',markersize=9, color="limegreen", label="LCC")
        
        ax02 = ax[0].twinx()
        
        lns2= ax02.plot(xind, ncc , marker='^', markersize=9, color="slateblue", label="NCC")
        
        ax[0].set_ylabel("LCC", fontsize=26, color="limegreen", fontweight='bold' )
        ax[0].tick_params(labelsize=22)
        
        ax02.set_ylabel("NCC", fontsize=24, color="slateblue", fontweight='bold' )
        ax02.tick_params(labelsize=22)
        
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc=0, fontsize=20)
        ax[0].grid(True)
        
#        ax[0].set_title(titlename, fontsize=20)
        
        lns1 = ax[1].plot(xind, sf, marker='o', markersize=9,color="coral", label="SF")
                
        ax12 = ax[1].twinx()
        
        lns2 = ax12.plot(xind, efr , marker='^', markersize=9, color="dodgerblue", label="EFR")
        
        ax[1].set_ylabel("SF", fontsize=24, color="coral", fontweight='bold')
        ax[1].tick_params( labelsize=22)
        ax12.set_ylabel("EFR", fontsize=24, color="dodgerblue", fontweight='bold')
        ax12.tick_params(labelsize=22)
        ax12.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#        ax[1].set_xlabel('Percolation stage', fontsize=24)
                
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax[1].legend(lns, labs, loc=0, fontsize=20)
        ax[1].grid(True)
        
        ##################
        if nplots==3:
            lns1 = ax[2].plot(xind, sfwt, marker='o', markersize=9,color="coral", label="SF")
                    
            ax12 = ax[2].twinx()
            
            lns2 = ax12.plot(xind, efrwt , marker='^', markersize=9, color="dodgerblue", label="EFR")
            
            ax[2].set_ylabel("SFWt", fontsize=24, color="coral", fontweight='bold')
            ax[2].tick_params( labelsize=22)
            ax12.set_ylabel("EFRWt", fontsize=24, color="dodgerblue", fontweight='bold')
            ax12.tick_params(labelsize=22)
            ax12.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
                    
            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax[2].legend(lns, labs, loc=0, fontsize=20)
            ax[2].grid(True)
            
        ax[-1].set_xlabel('Percolation stage', fontsize=24)
        fig.tight_layout()
        plt.show()
    

        
            
        
        
        
        
        
        
        
        
        
        
        
        
    
        
    
