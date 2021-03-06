import pm4py
import copy
import numpy as np
from tqdm import tqdm
from random import choice
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.petri_net.data_petri_nets import semantics as dpn_semantics
from pm4py.objects.petri_net import properties as petri_properties
from pm4py.objects.petri_net.data_petri_nets.data_marking import DataMarking

# create empty petri net
net_name = "running-example-Will-BPM-silent-loops-silent-loopB"
net = PetriNet(net_name)

# create and add places
source = PetriNet.Place("source")
sink = PetriNet.Place("sink")
p_0 = PetriNet.Place("p_0")
p_1 = PetriNet.Place("p_1")
p_2 = PetriNet.Place("p_2")
p_3 = PetriNet.Place("p_3")
p_4 = PetriNet.Place("p_4")
p_5 = PetriNet.Place("p_5")
p_6 = PetriNet.Place("p_6")
p_7 = PetriNet.Place("p_7")
p_8 = PetriNet.Place("p_8")
p_9 = PetriNet.Place("p_9")
p_10 = PetriNet.Place("p_10")
p_11 = PetriNet.Place("p_11")
p_12 = PetriNet.Place("p_12")
p_13 = PetriNet.Place("p_13")
net.places.add(source)
net.places.add(sink)
net.places.add(p_0)
net.places.add(p_1)
net.places.add(p_2)
net.places.add(p_3)
net.places.add(p_4)
net.places.add(p_5)
net.places.add(p_6)
net.places.add(p_7)
net.places.add(p_8)
net.places.add(p_9)
net.places.add(p_10)
net.places.add(p_11)
net.places.add(p_12)
net.places.add(p_13)

# create and add trasitions
t_A = PetriNet.Transition("trans_A", "Register claim")
t_B = PetriNet.Transition("trans_B", "Check all")
t_C = PetriNet.Transition("trans_C", "Check payment only")
t_D = PetriNet.Transition("trans_D", "Evaluate possible appeal")
t_E = PetriNet.Transition("trans_E", None)
t_F = PetriNet.Transition("trans_F", "Evaluate Claim")
t_G = PetriNet.Transition("trans_G", "Issue Payment")
t_H = PetriNet.Transition("trans_H", None)
t_I = PetriNet.Transition("trans_I", None)
t_L = PetriNet.Transition("trans_L", "Send rejection letter")
t_M = PetriNet.Transition("trans_M", "Send rejection email")
t_N = PetriNet.Transition("trans_N", "Send approval letter")
t_O = PetriNet.Transition("trans_O", "Send approval email")
t_P = PetriNet.Transition("trans_P", None)
t_Q = PetriNet.Transition("trans_Q", None)
t_R = PetriNet.Transition("trans_R", "Evaluate Appeal")
t_S = PetriNet.Transition("trans_S", None)
t_T = PetriNet.Transition("trans_T", "Final check")
t_U = PetriNet.Transition("trans_U", "Claim discarded")
t_V = PetriNet.Transition("trans_V", "Archive")
net.transitions.add(t_A)
net.transitions.add(t_B)
net.transitions.add(t_C)
net.transitions.add(t_D)
net.transitions.add(t_E)
net.transitions.add(t_F)
net.transitions.add(t_G)
net.transitions.add(t_H)
net.transitions.add(t_I)
net.transitions.add(t_L)
net.transitions.add(t_M)
net.transitions.add(t_N)
net.transitions.add(t_O)
net.transitions.add(t_P)
net.transitions.add(t_Q)
net.transitions.add(t_R)
net.transitions.add(t_S)
net.transitions.add(t_T)
net.transitions.add(t_U)
net.transitions.add(t_V)

# add arcs
petri_utils.add_arc_from_to(source, t_A, net)
petri_utils.add_arc_from_to(t_A, p_0, net)
petri_utils.add_arc_from_to(p_0, t_B, net)
petri_utils.add_arc_from_to(p_0, t_C, net)
petri_utils.add_arc_from_to(t_B, p_1, net)
petri_utils.add_arc_from_to(t_C, p_1, net)
petri_utils.add_arc_from_to(p_1, t_D, net)
petri_utils.add_arc_from_to(t_D, p_2, net)
petri_utils.add_arc_from_to(p_2, t_E, net)
petri_utils.add_arc_from_to(p_2, t_U, net)
petri_utils.add_arc_from_to(t_U, p_13, net)
petri_utils.add_arc_from_to(t_E, p_3, net)
petri_utils.add_arc_from_to(p_3, t_F, net)
petri_utils.add_arc_from_to(t_F, p_4, net)
petri_utils.add_arc_from_to(t_F, p_5, net)
petri_utils.add_arc_from_to(p_4, t_G, net)
petri_utils.add_arc_from_to(p_4, t_H, net)
petri_utils.add_arc_from_to(p_5, t_H, net)
petri_utils.add_arc_from_to(p_5, t_I, net)
petri_utils.add_arc_from_to(t_G, p_11, net)
petri_utils.add_arc_from_to(t_H, p_6, net)
petri_utils.add_arc_from_to(t_I, p_7, net)
petri_utils.add_arc_from_to(p_6, t_L, net)
petri_utils.add_arc_from_to(p_6, t_M, net)
petri_utils.add_arc_from_to(p_7, t_N, net)
petri_utils.add_arc_from_to(p_7, t_O, net)
petri_utils.add_arc_from_to(t_L, p_8, net)
petri_utils.add_arc_from_to(t_M, p_8, net)
petri_utils.add_arc_from_to(t_N, p_9, net)
petri_utils.add_arc_from_to(t_O, p_9, net)
petri_utils.add_arc_from_to(p_8, t_P, net)
petri_utils.add_arc_from_to(p_9, t_Q, net)
petri_utils.add_arc_from_to(t_P, p_11, net)
petri_utils.add_arc_from_to(t_P, p_10, net)
petri_utils.add_arc_from_to(t_Q, p_10, net)
petri_utils.add_arc_from_to(p_11, t_R, net)
petri_utils.add_arc_from_to(p_10, t_R, net)
petri_utils.add_arc_from_to(t_R, p_12, net)
petri_utils.add_arc_from_to(p_12, t_T, net)
petri_utils.add_arc_from_to(p_12, t_S, net)
petri_utils.add_arc_from_to(t_S, p_3, net)
petri_utils.add_arc_from_to(t_T, p_13, net)
petri_utils.add_arc_from_to(p_13, t_V, net)
petri_utils.add_arc_from_to(t_V, sink, net)

# transitions properties
t_B.properties[petri_properties.TRANS_GUARD] = 'amount > 500 && policyType == "normal"'
t_B.properties[petri_properties.READ_VARIABLE] = ['amount', 'policyType']
t_B.properties[petri_properties.WRITE_VARIABLE] = []

t_C.properties[petri_properties.TRANS_GUARD] = 'amount <= 500 || policyType == "premium"'
t_C.properties[petri_properties.READ_VARIABLE] = ['amount', 'policyType']
t_C.properties[petri_properties.WRITE_VARIABLE] = []

t_E.properties[petri_properties.TRANS_GUARD] = 'discarded == False'
t_E.properties[petri_properties.READ_VARIABLE] = ['discarded']
t_E.properties[petri_properties.WRITE_VARIABLE] = []

t_U.properties[petri_properties.TRANS_GUARD] = 'discarded == True'
t_U.properties[petri_properties.READ_VARIABLE] = ['discarded']
t_U.properties[petri_properties.WRITE_VARIABLE] = []

t_G.properties[petri_properties.TRANS_GUARD] = 'status == "approved"'
t_G.properties[petri_properties.READ_VARIABLE] = ['status']
t_G.properties[petri_properties.WRITE_VARIABLE] = []

t_H.properties[petri_properties.TRANS_GUARD] = 'status == "rejected"'
t_H.properties[petri_properties.READ_VARIABLE] = ['status']
t_H.properties[petri_properties.WRITE_VARIABLE] = []

t_I.properties[petri_properties.TRANS_GUARD] = 'status == "approved"'
t_I.properties[petri_properties.READ_VARIABLE] = ['status']
t_I.properties[petri_properties.WRITE_VARIABLE] = []

t_L.properties[petri_properties.TRANS_GUARD] = 'communication == "letter"'
t_L.properties[petri_properties.READ_VARIABLE] = ['communication']
t_L.properties[petri_properties.WRITE_VARIABLE] = []

t_M.properties[petri_properties.TRANS_GUARD] = 'communication == "email"'
t_M.properties[petri_properties.READ_VARIABLE] = ['communication']
t_M.properties[petri_properties.WRITE_VARIABLE] = []

t_N.properties[petri_properties.TRANS_GUARD] = 'communication == "letter"'
t_N.properties[petri_properties.READ_VARIABLE] = ['communication']
t_N.properties[petri_properties.WRITE_VARIABLE] = []

t_O.properties[petri_properties.TRANS_GUARD] = 'communication == "email"'
t_O.properties[petri_properties.READ_VARIABLE] = ['communication']
t_O.properties[petri_properties.WRITE_VARIABLE] = []

t_T.properties[petri_properties.TRANS_GUARD] = 'appeal == False'
t_T.properties[petri_properties.READ_VARIABLE] = ['appeal']
t_T.properties[petri_properties.WRITE_VARIABLE] = []

t_S.properties[petri_properties.TRANS_GUARD] = 'appeal == True'
t_S.properties[petri_properties.READ_VARIABLE] = ['appeal']
t_S.properties[petri_properties.WRITE_VARIABLE] = []

# initial and final marking
initial_marking = DataMarking()
initial_marking[source] = 1
final_marking = DataMarking()
final_marking[sink] = 1

#breakpoint()
pnml_exporter.apply(net, initial_marking, "models/{}.pnml".format(net_name), final_marking=final_marking)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)
pn_visualizer.save(gviz, "{}.svg".format(net_name))
