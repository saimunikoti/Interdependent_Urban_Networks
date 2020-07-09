# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:13:25 2019

@author: saimunikoti
"""
# import networkx as nx
# from src.models.Robustness_HFGT_Main import randomattack, targetattack, new_resmetric
# import numpy as np
# import matplotlib.pyplot as plt
#
# ## different process
# # deifne graph
# G = nx.DiGraph()
# #15 process
# #G.add_weighted_edges_from([(7,1,0.2),(8,1,0.8),(5,2,1),(6,4,0.9),(9,4,0.1),(12,4,0.2),(13,4,0.8), (10,5,1),(1,6,1),(3,7,1)
# #                            ,(5,8,1),(2,9,1),(2,10,1),(5,11,1),(3,12,1),(5,13,1),(12,14,0.2),(13,14,0.8),(14,15,1)])
#
# # convention - Target node depends on start nodeand arrow is from start-target
#
# #43 process network
# #G.add_weighted_edges_from([(1,11,1),(1,12,1),(2,16,1),(2,17,1),(2,18,1),(2,19,1),(3,8,0.5),(4,21,1),(4,22,1),
# #                           (4,24,1),(4,28,1),(6,10,1),(6,20,1),(7,8,0.5),(9,34,1),(10,4,0.2),(10,38,0.2),
# #                           (10,13,1),(11,4,0.8),(11,38,0.3),(12,3,0.7),(12,5,0.2),(15,29,1),(16,1,0.7),(16,35,0.25),
# #                           (18,9,0.6),(18,39,0.2),(19,5,0.5),(20,3,0.3),(20,5,0.2),(21,1,0.3),(21,35,0.25),(23,33,1),(24,5,0.4),(25,31,1),
# #                           (26,30,1),(27,32,1),(28,9,0.4),(28,39,0.3),(29,38,0.5),(30,36,1),(31,37,1),(32,39,0.5),(33,35,0.5),
# #                           (34,5,0.1) ,(42,40,0.5) ,(17,40,0.5),(40,41,1),(4,42,1),(32,43,0.5),(17,43,0.25),(42,43,0.25)])
#
# #121 process old
# #G.add_weighted_edges_from([(1,35,1),(1,59,1),(1,80,1),(2,60,1),(2,101,1),(3,36,1),
# #                           (4,37,0.5),(4,55,0.5),(4,61,1),(4,81,0.5),(4,97,0.5),(5,62,1),(5,102,1),
# #                           (6,38,0.5),(6,57,0.5),(6,63,1),(6,82,0.5),(6,99,0.5),(7,39,1),(7,64,1),
# #                           (8,40,1),(8,65,1),(8,83,1),(9,41,0.5),(9,54,0.5),(9,66,1),(9,84,0.5),(9,96,0.5),
# #                           (10,67,1),(10,103,1),(11,42,1),(12,43,0.5),(12,56,0.5),(12,68,1),
# #                           (12,85,0.5),(12,98,0.5),(13,44,0.5),(13,53,0.5),(13,69,1),(13,86,0.5),(13,95,0.5),(13,106,1),(14,45,1),
# #                           (14,70,1),(14,87,1),(15,71,1),(16,46,0.5),(16,58,0.5),
# #                           (16,72,1),(16,88,0.5),(16,100,0.5),(17,47,1),(17,73,1),(17,89,1),(17,107,1),(18,74,1),(18,104,1),
# #                           (19,48,1),(19,75,1),(19,90,1),(19,108,1),(20,49,1),(20,76,1),(20,91,1),(20,109,1),
# #                           (21,50,1),(21,77,1),(22,51,1),(22,78,1),(22,93,1),(22,110,1),
# #                           (23,105,1),(24,52,1),(24,79,1),(24,94,1),(24,111,1),(25,112,1),(26,113,1),(27,114,1),
# #                           (28,115,1),(29,116,1),(30,117,1),(31,118,1),(32,119,1),(33,120,1),(34,121,1),
# #                           (35,5,1),(36,5,1),(37,5,1),(38,5,1),(39,5,1),(40,5,1),(41,15,1),(42,15,1),(43,15,1),
# #                           (44,15,1),(45,15,1),(46,15,1),(47,15,1),(48,15,1),(49,15,1),(50,15,1),(51,15,1),
# #                           (52,15,1),(53,23,1),(54,23,1),(55,23,1),(56,23,1),(57,23,1),(58,23,1),(59,3,1),
# #                           (60,3,1),(61,3,1),(62,3,1),(63,3,1),(64,3,1),(65,3,1),(66,3,1),(67,3,1),(68,3,1),
# #                           (69,11,1),(70,11,1),(71,3,1),(72,3,1),(73,3,1),(74,11,1),(75,11,1),(76,3,1),
# #                           (77,3,1),(78,11,1),(79,11,1),(80,2,1),(81,2,1),(82,2,1),(83,2,1),(84,2,1),(85,10,1),
# #                           (86,10,1),(87,10,1),(88,10,1),(89,10,1),(90,18,1),(91,18,1),(92,18,1),(93,18,1),(94,18,1),
# #                           (95,23,1),(96,23,1),(97,23,1),(98,23,1),(99,23,1),(100,23,1),(101,7,1),(102,7,1),(103,7,1),
# #                           (104,21,1),(105,21,1),(106,7,1),(107,7,1),(108,7,1),(109,21,1),(110,21,1),(111,21,1),
# #                           (112,14,1),(113,1,1),(114,9,1),(115,14,1),(116,14,1),(117,1,1),(118,8,1),(119,14,1),
# #                           (120,14,1),(121,14,1),(35,36,1),
# #                           (35,37,1),(36,37,1),(39,38,1),(40,38,1),(40,39,1),(41,42,1),(41,43,1),(41,44,1),(41,45,1),
# #                           (42,43,1),(42,44,1),(42,45,1),(43,44,1),(43,45,1),(44,45,1),(47,46,1),(48,36,1),(48,37,1),
# #                           (48,38,1),(48,39,1),(48,40,1),(48,41,1),(48,42,1),(48,43,1),(48,44,1),(48,45,1),(49,36,1),
# #                           (49,37,1),(49,38,1),(49,39,1),(49,40,1),(49,41,1),(49,42,1),(49,43,1),(49,44,1),(49,45,1),
# #                           (49,48,1),(50,36,1),(50,37,1),(50,38,1),(50,39,1),(50,40,1),(50,41,1),(50,42,1),(50,43,1),
# #                           (50,44,1),(50,45,1),(50,48,1),(50,49,1),(51,36,1),(51,37,1),(51,38,1),(51,39,1),(51,40,1),
# #                           (51,41,1),(51,42,1),(51,43,1),(51,44,1),(51,45,1),(51,48,1),(51,49,1),(51,50,1),(52,36,1),
# #                           (52,37,1),(52,38,1),(52,39,1),(52,40,1),(52,41,1),(52,42,1),(52,43,1),(52,44,1),(52,45,1),
# #                           (52,53,1),(52,54,1),(52,55,1),(52,56,1),(52,57,1),(53,36,1),(53,37,1),(53,38,1),(53,39,1),
# #                           (53,40,1),(54,36,1),(54,37,1),(54,38,1),(54,39,1),(54,40,1),(55,36,1),(56,36,1),(56,37,1),
# #                           (56,38,1),(56,39,1),(56,40,1),(57,36,1),(57,37,1),(57,38,1),(58,36,1),(58,37,1),(58,38,1),
# #                           (58,39,1),(58,40,1),(58,41,1),(58,42,1),(58,43,1),(58,44,1),(58,45,1),(58,46,1),(61,60,1),
# #                           (62,60,1),(62,61,1),(63,60,1),(63,61,1),(63,62,1),(64,60,1),(64,61,1),(64,62,1),(64,63,1),
# #                           (65,60,1),(65,61,1),(65,62,1),(65,63,1),(66,60,1),(66,61,1),(66,62,1),(66,63,1),(66,65,1),
# #                           (68,60,1),(68,61,1),(68,62,1),(68,65,1),(70,69,1),(71,67,1),(72,67,1),(72,71,1),(75,74,1),
# #                           (76,60,1),(76,61,1),(76,77,1),(77,60,1),(77,61,1),(78,69,1),(79,74,1),(102,101,1),(103,101,1),
# #                           (106,101,1),(106,102,1),(107,101,1),(107,102,1),(107,106,1),(108,101,1),(108,102,1),
# #                           (108,106,1),(108,107,1),(109,104,1),(110,105,1),(111,105,1),(111,110,1)])
#
# #%%## 111 process new
# G.add_weighted_edges_from([(95,1,0.5),(70,1,0.5),(60,2,0.5),(101,2,0.5),(1,25,0.5),(2,81,1),
#                            (2,82,1),(2,84,1),(3,60,1),(3,61,1),(3,62,1),(3,63,1),(3,64,1),(3,66,1),(3,67,1),(3,68,1),
#                            (5,36,1),(5,37,1),(5,38,1),(5,39,1),(5,41,1),(7,101,1),(7,102,1),(7,103,1),(7,106,1),(7,107,1),(7,108,1),
#                            (8,26,0.5),(10,85,1),(10,86,1),(10,88,1),(10,89,1),(11,69,1),(11,74,1),(11,75,1),(11,78,1),(11,79,1),(14,27,0.5),
#                            (15,35,1),(15,42,1),(15,43,1),(15,44,1),(15,46,1),(15,47,1),(15,48,1),(15,49,1),(15,50,1),(15,51,1),(15,52,1),(15,53,1),(15,55,1),
#                            (18,90,1),(18,91,1),(18,92,1),(18,93,1),(18,94,1),
#                            (21,100,1),(21,104,1),(21,105,1),(21,109,1),(21,110,1),(21,111,1),(23,28,0.5),
#                            (34,31,0.5),(35,3,0.2),(35,26,0.1),(36,3,0.8),(36,26,0.4),(37,4,0.3),(38,6,0.5),(39,7,0.7),(39,28,0.35),
#                            (40,34,1),(41,9,0.5),(43,12,0.6),(44,13,0.3),(46,16,0.5),(47,17,0.3),
#                            (48,19,0.3),(49,20,0.4),(50,21,0.8),(50,33,0.4),(51,22,0.4),(52,24,0.4),(53,11,1),(53,30,0.5),
#                            (54,29,0.5),(55,4,0.2),(56,30,0.5),(57,33,0.5),(58,32,0.5),(59,14,0.5),(60,2,0.5),(60,25,0.25),
#                            (61,4,0.3),(62,5,0.5),(62,27,0.25),(63,6,0.3),(64,7,0.3),(64,28,0.15),(65,56,0.5),(66,9,0.3),
#                            (67,10,0.5),(67,29,0.25),(68,12,0.2),(69,13,0.3),(70,1,0.5),(71,15,1),(71,31,0.5),
#                            (72,16,0.3),(73,17,0.3),(74,18,0.5),(74,32,0.25),(75,19,0.3),(76,20,0.3),(77,21,0.20),(77,33,0.1),
#                            (78,22,0.3),(79,24,0.2),(80,57,0.5),(81,4,0.2),(82,6,0.2),(83,57,0.5),(84,9,0.2),(85,12,0.2),
#                            (86,13,0.1),(87,58,1),(88,16,0.2),(89,17,0.1),(90,19,0.1),(91,20,0.3),(93,22,0.30),(94,24,0.3),
#                            (95,1,0.5),(96,54,1),(97,8,1),(98,56,0.5),(99,14,0.5),(100,24,0.1),
#                            (101,2,0.5),(101,25,0.25),(102,5,0.5),(102,27,0.25),(103,10,0.5),(103,29,0.25),(104,18,0.5),(104,32,0.25),
#                            (106,13,0.3),(107,17,0.3),(108,19,0.3)])
#
# # assign uniform binary weight to all nodes
# Gunwt = G.copy()
# for u,v in Gunwt.edges():
#     Gunwt[u][v]['weight']=1
#
# ## plot
# pos = nx.circular_layout(G)
# d = dict(G.degree)
# nodes = nx.draw_networkx_nodes(G, pos, with_labels=True,  node_size=[v * 100 for v in d.values()], node_color="lightgreen")
# labels = nx.draw_networkx_labels(G, pos)
#
# nx.draw_networkx_edges(G, pos, edge_color='black', width=1, arrowstyle='->', arrowsize=15)
# labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
#
# plt.xticks(ticks=[])
# plt.yticks([])
#
# #%% new plot
#
# #fig, ax = plt.subplots()
# #nodes = nx.draw_networkx_nodes(G, pos, with_labels=False,  node_size=[v * 100 for v in d.values()], node_color="lightsalmon")
# #
# #nodes.set_edgecolor('orangered')
# #nx.draw_networkx_edges(G, pos, edge_color='peachpuff', width=2)
# #
# #ax.set_facecolor('black')
# #ax.axis('off')
# #
# #fig.set_facecolor('black')
# #plt.show()
#
# ##%%############################## initialize q and crf attributes for each nodes
#
# def get_initialattr(G):
#
#     m=G.size(weight='weight')
#     for (node, val) in G.degree(weight='weight'):
#         G.nodes[node]['qo'] = val
#         G.nodes[node]['crf'] = 1
#         G.nodes[node]['orgedgewtcount'] = m
#
#     count = 0
#     for (node, val) in G.in_degree(weight='weight'):
#         G.nodes[node]['indegwt'] = val
#         if val !=0:
#             count = count+1
#
#     for (node, val) in G.degree(weight=None):
#         G.nodes[node]['orgunwtdegree'] = val
#         G.nodes[node]['orgindgwtcount'] = count
#
#     for u,v in G.edges():
#
#         G[u][v]['newweight'] = G[u][v]['weight']
#
# get_initialattr(G)
# get_initialattr(Gunwt)

#%%############################### random node complete attack ########################
### new method  of full trajectory of percolation

## ############################# Random node complete attack

Lcc_cor, Ncc_cor, Sf_cor,Sfwt_cor, Efr_cor, Efrwt_cor, crn_cor, crlcc_cor, crsf_cor, crefr_cor, Expn_cor, crncc_cor, \
        cridr_cor, Idr_cor = randout.nodeatkcomplete_sim(G, 1, 0.5)

Table_wt_complete = metric.get_percentfall_resmetric(Lcc_cor, Efrwt_cor, Idr_cor)

Lcc_cor, Ncc_cor, Sf_cor, Sfwt_cor, Efr_cor, Efrwt_cor, crn_cor, crlcc_cor, crsf_cor, crefr_cor, Expn_cor, crncc_cor, \
cridr_cor, Idr_cor = randout.nodeatkcomplete_sim(Gunwt, 4000,0.7)

Table_unwt_complete = metric.get_percentfall_resmetric(Lcc_cor, Efrwt_cor, Idr_cor)

with open("Completerandom_LCC.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(Lcc_cor)
#
stages_wt_completerandom = metric.get_countstages(Lcc_cor, Efrwt_cor, Idr_cor)
##%%############################# random node partial node attack #################

Lcc_par, Ncc_par, Sf_par,Sfwt_par, Efr_par,Efrwt_par, crn_par, crlcc_par, crsf_par, crefr_par, Expn_par, crncc_par,\
        cridr_par, Idr_par  = randout.nodeatkpartial_sim(G, 0.1, 100, 0.5)

Table_wt_partial = metric.get_percentfall_resmetric(Lcc_par, Efrwt_par, Idr_par)

Lcc_par, Ncc_par, Sf_par,Sfwt_par, Efr_par,Efrwt_par, crn_par, crlcc_par, crsf_par, crefr_par, Expn_par, crncc_par,\
        cridr_par, Idr_par  = randout.nodeatkpartial_sim(Gunwt, 4000, 0.7)

Table_unwt_partial = metric.get_percentfall_resmetric(Lcc_par, Efrwt_par, Idr_par)

metric.plot4d_resmetric(Lcc, Lccp,Ncc, Nccp, Efrwt,Efrwtp ,Idr, Idrp)

metric.plot4d_resmetric(Lcc_cor, Lcc_par, Lcc_par, Ncc_cor, Ncc_par, Ncc_par, Efrwt_cor, Efr_par , Efr_par, Idr_cor, Idr_par, Idr_par)

metric.plot4d_2compresmetric(Lcc_cor, Lcc_par, Ncc_cor, Ncc_par, Efrwt_cor, Efrwt_par, Idr_cor, Idr_par)

with open("Partialrandom_LCC.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(Lcc_par)

# track stages for degradation
stages_wt_partialrandom = metric.get_countstages(Lcc_par, Efrwt_par, Idr_par)

##%%################################### Target complete node attack ############################

# Lcc_cot, Ncc_cot, Sf_cot, Efr_cot, crn_cot, crlcc_cot, crsf_cot, crefr_cot, crncc_cot, acc_cot, idr_cot  = tarout.completenodeatk(G,"out degree" ,"weight",0.7)

Lcc_cot, Ncc_cot, Sf_cot, Efr_cot, crn_cot, crlcc_cot, crsf_cot, crefr_cot, crncc_cot, cridr_cot, Idr_cot, acc_cot  = \
    tarout.completenodeatk(G,"weight node betweenness" ,"weight",0.5, frames=None)

Lcc_cot_outd, Ncc_cot_outd, Sf_cot, Efr_cot_outd, crn_cot, crlcc_cot, crsf_cot, crefr_cot, crncc_cot, cridr_cot, \
    Idr_cot_outd, acc_cot  =  tarout.completenodeatk(G,"out degree" ,"weight",0.5)

Table_wt_complete = metric.get_percentfall_resmetric(Lcc_cot, Efr_cot, Idr_cot)
Table_wt_complete = metric.get_percentfall_resmetric(Lcc_cot_outd, Efr_cot_outd, Idr_cot_outd)

Lcc_cot, Ncc_cot, Sf_cot, Efr_cot, crn_cot, crlcc_cot, crsf_cot, crefr_cot, crncc_cot, acc_cot, Idr_cot = \
    tarout.completenodeatk(Gunwt,"weight node betweenness" ,"weight",0.5)

Table_unwt_complete = metric.get_percentfall_resmetric( Lcc_cot, Efr_cot, Idr_cot)
Table_Unwt_complete = metric.get_percentfall_resmetric(Lcc_cot_outd, Efr_cot_outd, Idr_cot_outd)

# metric.plot2d_resmetric(Lcc, Ncc, Sf, Efr,"target complete")
metric.plot4d_resmetric(Lcc_cor, Lcc_cot, Lcc_cot_outd, Ncc_cor, Ncc_cot, Ncc_cot_outd, Efrwt_cor, Efr_cot , Efr_cot_outd, Idr_cor, Idr_cot, Idr_cot_outd)

stages_wt_completetargetbtw = metric.get_countstages(Lcc_cot, Efr_cot, Idr_cot)
stages_wt_completetargetoutd = metric.get_countstages(Lcc_cot_outd, Efr_cot_outd, Idr_cot_outd)
##%% Target partial node attack

Lcc_pat, Ncc_pat, Sf_pat, Efr_pat, crn_pat, crlcc_pat, crsf_pat, crefr_pat, crncc_pat, cridr_pat, Idr_pat, acc_pat  = \
        tarout.partialnodeattack(G, 0.1,"weight node betweenness" ,"weight",0.5, allframes=None)

Lcc_pat_outd, Ncc_pat_outd, Sf_pat_outd, Efr_pat_outd, crn_pat_outd, crlcc_pat_outd, crsf_pat_outd, crefr_pat_outd, \
        crncc_pat_outd, cridr_pat_outd, Idr_pat_outd = tarout.partialnodeattack(G,0.1, "out degree" ,"weight",0.5)

Table_wt_partial = metric.get_percentfall_resmetric(Lcc_pat, Efr_pat, Idr_pat)
Table_wt_partial = metric.get_percentfall_resmetric(Lcc_pat_outd, Efr_pat_outd, Idr_pat_outd)

stages_wt_partialtargetbtw = metric.get_countstages(Lcc_pat, Efr_pat, Idr_pat)
stages_wt_partialtargetoutd = metric.get_countstages(Lcc_pat_outd, Efr_pat_outd, Idr_pat_outd)

Lcc_pat, Ncc_pat, Sf_pat, Efr_pat, crn_pat, crlcc_pat, crsf_pat, crefr_pat, crncc_pat, cridr_pat, Idr_pat  = \
        tarout.partialnodeattack(Gunwt ,0.2, "weight node betweenness" ,"weight", 0.5)

Lcc_pat_outd, Ncc_pat_outd, Sf_pat_outd, Efr_pat_outd, crn_pat_outd, crlcc_pat_outd, crsf_pat_outd, crefr_pat_outd, \
        crncc_pat_outd, cridr_pat_outd, Idr_pat_outd  = tarout.partialnodeattack(Gunwt,1000, "out degree" ,"weight",0.7)

Table_unwt_partial = metric.get_percentfall_resmetric(Lcc_pat, Efr_pat, Idr_pat)

Table_unwt_partial = metric.get_percentfall_resmetric(Lcc_pat_outd, Efr_pat_outd, Idr_pat_outd)

# compaison betweeen partial random and partial target
metric.plot4d_resmetric(Lcc_par, Lcc_pat, Lcc_pat_outd, Ncc_par, Ncc_pat,Ncc_pat_outd, Efrwt_par, Efr_pat,
                        Efr_pat_outd, Idr_par, Idr_pat,Idr_pat_outd)

# between complete target and partial target
metric.plot4d_2compresmetric(Lcc_cot, Lcc_pat, Ncc_cot, Ncc_pat, Efr_cot, Efr_pat, Idr_cot, Idr_pat)

metric.plot4d_2compresmetric(Lcc_cot_outd, Lcc_pat_outd, Ncc_cot_outd, Ncc_pat_outd, Efr_cot_outd, Efr_pat_outd, Idr_cot_outd, Idr_pat_outd)

with open("Partialtarget_btw_LCC.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(Lcc_pat)

Table_wt_partial = metric.get_countstages(Lcc_pat, Efr_pat, Idr_pat)

##%% optimal weight
optimalnodeweight = tarout.get_optimalnodeweights(G, 100000, 3,'idr')

 # def get_optimalnodeweights(nsim, nodevar, targetmetric):
#    weightlist=[]
#    lcclist=[]
#    sflist=[]
#    efrlist=[]
#    ncclist= []
#    A = np.arange(1,10)
#    for countsim in range(nsim):
#        collection = np.random.choice(A, G.in_degree(nodevar), replace=True)
#        countweight = collection/(np.sum(collection))
#        gcopy = G.copy()
#        for countedge,(u,v) in enumerate(gcopy.in_edges(nodevar)):
#            gcopy[u][v]['weight'] = countweight[countedge]
#        
#        Lcc, Ncc, Sf, Efr, criticaln, criticallcc, crsf, crefr, crncc  = tarout.partialnodeatk(gcopy,"weight node betweenness" ,"weight",0.7)
#        weightlist.append(countweight)
#        lcclist.append(criticallcc)
#        sflist.append(crsf)
#        efrlist.append(crefr)
#        ncclist.append(crncc)
#    
#    efrlist = np.array(efrlist)
#    weightlist = np.array(weightlist)
#    sflist = np.array(sflist)
#    lcclist = np.array(lcclist)
#        
#    if targetmetric=="efr":        
#        maxindex = np.where(efrlist==np.max(efrlist))[0]
#    elif targetmetric =="lcc":
#        maxindex = np.where(lcclist==np.max(lcclist))[0]
#
#    optimalweight = weightlist[maxindex,:]
#    optimalweight = np.round(optimalweight,1)
#    optimalweight = np.unique(optimalweight, axis=0)
#    
#    return optimalweight

#%% plot of matrics for  various Qos values

## plot of metrics for

dfactor = [1.0, 0.9, 0.8, 0.7 ,0.6]

Colour = ["limegreen","dodgerblue","coral","lightseagreen","burlywood"]

def plot_base(tempax, y, ylabel, Colour):
    tempax.plot(xind, y, marker='o', color=Colour)
    tempax.xaxis.set_tick_params(labelsize=16)
    tempax.yaxis.set_tick_params(labelsize=16)
            
    tempax.set_ylabel(ylabel, fontsize=20)
    tempax.grid(True)
    
fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=False,sharey=False,figsize=(8,6))   
fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=False,sharey=False,figsize=(8,6))  
 
for countdf,df in enumerate(dfactor):
    
#    Lcc, Ncc, Sf, Sfwt,Efr,Efrwt, crn, crlcc, crsf, crefr, Expn, crncc = randout.nodeatkpartial_sim(G, 500, df) # dfact = weight degradation 
    Lcc, Ncc, Sf, Efr, criticaln, criticallcc, crsf, crefr, crncc  = tarout.partialnodeatk(G,"weight node betweenness" ,"weight" , df)

    xind = np.arange(0,len(Lcc))
            
    plot_base(ax1[0], Lcc, "LCC", Colour[countdf])
    
    plot_base(ax1[1], Ncc, "NCC", Colour[countdf])
    
    plot_base(ax2[0], Sf, "Service Factor", Colour[countdf])
    plot_base(ax2[1], Efr, "Edge flow robustness", Colour[countdf])
    
ax1[0].legend(("1","0.9","0.8","0.7","0.6"))    
ax1[1].legend(("1","0.9","0.8","0.7","0.6"))    
ax1[1].set_xlabel("Attack event", fontsize=20)
ax2[0].legend(("1","0.9","0.8","0.7","0.6"))    
ax2[1].legend(("1","0.9","0.8","0.7","0.6"))    
ax2[1].set_xlabel("Attack event", fontsize=20)

##%% Histogram of number of nodes in each sub clusters of a graph

st1 = acc_pat[0]
st5 = acc_pat[1]
st6 = acc_pat[10]
st15 = acc_pat[22]
st16 = acc_pat[23]

Bins=10
fig, axs = plt.subplots(nrows=1, ncols=5, sharex=False,sharey=True,figsize=(10,4))

axs[0].hist(st1, density=False, bins=Bins,color="mediumslateblue")

axs[0].xaxis.set_tick_params(labelsize=23)
axs[0].set_title("Stage 0", fontsize=24)
axs[0].grid(True)

axs[1].hist(st5, density=False, bins=Bins, color="mediumslateblue")
#axs[1].set_xlabel("Stage 5", fontsize=24)
axs[1].xaxis.set_tick_params(labelsize=23)
axs[1].set_title("Stage 1", fontsize=24)
axs[1].grid(True)

axs[2].hist(st6, density=False, bins=Bins, color="mediumslateblue")
axs[2].set_xlabel("Nodes per sub-component", fontsize=26)
axs[2].xaxis.set_tick_params(labelsize=23)
axs[2].set_title("Stage 10", fontsize=24)
axs[2].grid(True)

axs[3].hist(st15, density=False, bins=Bins, color="mediumslateblue")
#axs[3].set_xlabel("Stage 15",fontsize=24)
axs[3].xaxis.set_tick_params(labelsize=23)
axs[3].set_title("Stage 22", fontsize=24)
axs[3].grid(True)

axs[4].hist(st16, density=False, bins=Bins, color="mediumslateblue")
#axs[4].set_xlabel("Stage 16", fontsize=24)
axs[4].xaxis.set_tick_params(labelsize=23)
axs[4].set_title("Stage 23", fontsize=24)
axs[4].grid(True)

axs[0].set_ylabel("Frequency", fontsize=26)
axs[0].yaxis.set_tick_params(labelsize=23)


##

