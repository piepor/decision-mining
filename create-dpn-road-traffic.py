from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
# create empty petri net
net_name = "Road_Traffic_Fine_Management_Process"
net = PetriNet(net_name)

# create and add places
source = PetriNet.Place("source")
sink = PetriNet.Place("sink")
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
p_14 = PetriNet.Place("p_14")
p_15 = PetriNet.Place("p_15")
p_16 = PetriNet.Place("p_16")
p_17 = PetriNet.Place("p_17")
p_18 = PetriNet.Place("p_18")
p_19 = PetriNet.Place("p_19")
p_20 = PetriNet.Place("p_20")
p_21 = PetriNet.Place("p_21")
p_22 = PetriNet.Place("p_22")
p_23 = PetriNet.Place("p_23")
p_24 = PetriNet.Place("p_24")
p_25 = PetriNet.Place("p_25")
p_26 = PetriNet.Place("p_26")
p_27 = PetriNet.Place("p_27")
net.places.add(source)
net.places.add(sink)
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
net.places.add(p_14)
net.places.add(p_15)
net.places.add(p_16)
net.places.add(p_17)
net.places.add(p_18)
net.places.add(p_19)
net.places.add(p_20)
net.places.add(p_21)
net.places.add(p_22)
net.places.add(p_23)
net.places.add(p_24)
net.places.add(p_25)
net.places.add(p_26)
net.places.add(p_27)

# create and add transitions
t_A = PetriNet.Transition("trans_A", "Create Fine")
t_B = PetriNet.Transition("trans_B", "Send Appeal to Prefecture")
t_C = PetriNet.Transition("trans_C", "Insert Fine Notification")
t_D = PetriNet.Transition("trans_D", "Send Fine")
t_E = PetriNet.Transition("trans_E", "Insert Date Appeal to Prefecture")
t_F = PetriNet.Transition("trans_F", "Payment")
t_G = PetriNet.Transition("trans_G", "Appeal to Judge")
t_H = PetriNet.Transition("trans_H", "Receive Result Appeal from Prefecture")
t_I = PetriNet.Transition("trans_I", "Notify Result Appeal to Offender")
t_L = PetriNet.Transition("trans_L", "Add penalty")
t_M = PetriNet.Transition("trans_M", "Send for Credit Collection")
skip_1 = PetriNet.Transition("skip_1", None)
skip_2 = PetriNet.Transition("skip_2", None)
skip_3 = PetriNet.Transition("skip_3", None)
skip_4 = PetriNet.Transition("skip_4", None)
skip_5 = PetriNet.Transition("skip_5", None)
skip_6 = PetriNet.Transition("skip_6", None)
skip_7 = PetriNet.Transition("skip_7", None)
skip_8 = PetriNet.Transition("skip_8", None)
skip_9 = PetriNet.Transition("skip_9", None)
skip_10 = PetriNet.Transition("skip_10", None)
skip_11 = PetriNet.Transition("skip_11", None)
skip_12 = PetriNet.Transition("skip_12", None)
skip_13 = PetriNet.Transition("skip_13", None)
tauSplit_1 = PetriNet.Transition("tauSplit_1", None)
tauSplit_2 = PetriNet.Transition("tauSplit_2", None)
tauSplit_3 = PetriNet.Transition("tauSplit_3", None)
tauJoin_1 = PetriNet.Transition("tauJoin_1", None)
tauJoin_2 = PetriNet.Transition("tauJoin_2", None)
tauJoin_3 = PetriNet.Transition("tauJoin_3", None)
tauJoin_4 = PetriNet.Transition("tauJoin_4", None)
tauJoin_5 = PetriNet.Transition("tauJoin_5", None)
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
net.transitions.add(skip_1)
net.transitions.add(skip_2)
net.transitions.add(skip_3)
net.transitions.add(skip_4)
net.transitions.add(skip_5)
net.transitions.add(skip_6)
net.transitions.add(skip_7)
net.transitions.add(skip_8)
net.transitions.add(skip_9)
net.transitions.add(skip_10)
net.transitions.add(skip_11)
net.transitions.add(skip_12)
net.transitions.add(skip_13)
net.transitions.add(tauSplit_1)
net.transitions.add(tauSplit_2)
net.transitions.add(tauSplit_3)
net.transitions.add(tauJoin_1)
net.transitions.add(tauJoin_2)
net.transitions.add(tauJoin_3)
net.transitions.add(tauJoin_4)
net.transitions.add(tauJoin_5)

# add arcs
petri_utils.add_arc_from_to(source, t_A, net)
petri_utils.add_arc_from_to(t_A, p_1, net)
petri_utils.add_arc_from_to(t_A, p_2, net)
petri_utils.add_arc_from_to(t_A, p_3, net)
petri_utils.add_arc_from_to(p_1, t_B, net)
petri_utils.add_arc_from_to(p_1, skip_1, net)
petri_utils.add_arc_from_to(p_2, skip_2, net)
petri_utils.add_arc_from_to(p_2, tauSplit_1, net)
petri_utils.add_arc_from_to(p_3, skip_3, net)
petri_utils.add_arc_from_to(p_3, skip_4, net)
petri_utils.add_arc_from_to(t_B, p_4, net)
petri_utils.add_arc_from_to(skip_1, p_4, net)
petri_utils.add_arc_from_to(skip_2, p_25, net)
petri_utils.add_arc_from_to(tauSplit_1, p_5, net)
petri_utils.add_arc_from_to(tauSplit_1, p_6, net)
petri_utils.add_arc_from_to(tauSplit_1, p_7, net)
petri_utils.add_arc_from_to(skip_3, p_8, net)
petri_utils.add_arc_from_to(skip_4, p_16, net)
petri_utils.add_arc_from_to(p_4, tauJoin_5, net)
petri_utils.add_arc_from_to(p_5, t_C, net)
petri_utils.add_arc_from_to(p_5, skip_5, net)
petri_utils.add_arc_from_to(p_6, t_D, net)
petri_utils.add_arc_from_to(p_7, t_E, net)
petri_utils.add_arc_from_to(p_7, skip_6, net)
petri_utils.add_arc_from_to(p_8, t_F, net)
petri_utils.add_arc_from_to(t_C, p_9, net)
petri_utils.add_arc_from_to(skip_5, p_9, net)
petri_utils.add_arc_from_to(t_D, p_10, net)
petri_utils.add_arc_from_to(t_E, p_11, net)
petri_utils.add_arc_from_to(skip_6, p_11, net)
petri_utils.add_arc_from_to(t_F, p_12, net)
petri_utils.add_arc_from_to(p_9, tauSplit_2, net)
petri_utils.add_arc_from_to(p_10, tauJoin_3, net)
petri_utils.add_arc_from_to(p_11, tauJoin_3, net)
petri_utils.add_arc_from_to(p_12, skip_7, net)
petri_utils.add_arc_from_to(p_12, skip_8, net)
petri_utils.add_arc_from_to(skip_7, p_8, net)
petri_utils.add_arc_from_to(tauSplit_2, p_13, net)
petri_utils.add_arc_from_to(tauSplit_2, p_14, net)
petri_utils.add_arc_from_to(tauSplit_2, p_15, net)
petri_utils.add_arc_from_to(skip_8, p_16, net)
petri_utils.add_arc_from_to(p_16, tauJoin_4, net)
petri_utils.add_arc_from_to(p_13, skip_9, net)
petri_utils.add_arc_from_to(p_13, t_G, net)
petri_utils.add_arc_from_to(p_14, t_H, net)
petri_utils.add_arc_from_to(p_14, skip_10, net)
petri_utils.add_arc_from_to(p_15, skip_11, net)
petri_utils.add_arc_from_to(p_15, tauSplit_3, net)
petri_utils.add_arc_from_to(skip_9, p_17, net)
petri_utils.add_arc_from_to(t_G, p_17, net)
petri_utils.add_arc_from_to(t_H, p_18, net)
petri_utils.add_arc_from_to(skip_10, p_18, net)
petri_utils.add_arc_from_to(skip_11, p_23, net)
petri_utils.add_arc_from_to(tauSplit_3, p_19, net)
petri_utils.add_arc_from_to(tauSplit_3, p_20, net)
petri_utils.add_arc_from_to(p_17, tauJoin_2, net)
petri_utils.add_arc_from_to(p_18, tauJoin_2, net)
petri_utils.add_arc_from_to(p_19, t_I, net)
petri_utils.add_arc_from_to(p_19, skip_12, net)
petri_utils.add_arc_from_to(p_20, t_L, net)
petri_utils.add_arc_from_to(t_I, p_21, net)
petri_utils.add_arc_from_to(skip_12, p_21, net)
petri_utils.add_arc_from_to(t_L, p_22, net)
petri_utils.add_arc_from_to(p_21, tauJoin_1, net)
petri_utils.add_arc_from_to(p_22, tauJoin_1, net)
petri_utils.add_arc_from_to(tauJoin_1, p_23, net)
petri_utils.add_arc_from_to(p_23, tauJoin_2, net)
petri_utils.add_arc_from_to(tauJoin_2, p_24, net)
petri_utils.add_arc_from_to(p_24, tauJoin_3, net)
petri_utils.add_arc_from_to(tauJoin_3, p_25, net)
petri_utils.add_arc_from_to(p_25, tauJoin_4, net)
petri_utils.add_arc_from_to(tauJoin_4, p_26, net)
petri_utils.add_arc_from_to(p_26, t_M, net)
petri_utils.add_arc_from_to(p_26, skip_13, net)
petri_utils.add_arc_from_to(t_M, p_27, net)
petri_utils.add_arc_from_to(skip_13, p_27, net)
petri_utils.add_arc_from_to(p_27, tauJoin_5, net)
petri_utils.add_arc_from_to(tauJoin_5, sink, net)

# initial and final marking
initial_marking = Marking()
initial_marking[source] = 1
final_marking = Marking()
final_marking[sink] = 1

#breakpoint()
pnml_exporter.apply(net, initial_marking, "models/{}.pnml".format(net_name), final_marking=final_marking)
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)
pn_visualizer.save(gviz, "{}.svg".format(net_name))
