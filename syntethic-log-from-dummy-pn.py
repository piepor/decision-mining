import pm4py
import copy
from random import choice
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.decision_mining import algorithm as decision_mining
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.data_petri_nets import semantics as dpn_semantics
from pm4py.objects.petri_net import properties as petri_properties
from pm4py.objects.petri_net.data_petri_nets.data_marking import DataMarking

# create empty petri net
net = PetriNet("petri_net")

# create and add fundamental places
source = PetriNet.Place("source")
sink = PetriNet.Place("sink")
p_1 = PetriNet.Place("p_1")
p_2 = PetriNet.Place("p_2")
net.places.add(source)
net.places.add(sink)
net.places.add(p_1)
net.places.add(p_2)

# create and add trasitions
t_1 = PetriNet.Transition("name_1", "label_1")
t_2 = PetriNet.Transition("name_2", "label_2")
t_2.properties[petri_properties.TRANS_GUARD] = 'A >= 5'
t_2.properties[petri_properties.READ_VARIABLE] = 'A'
t_2.properties[petri_properties.WRITE_VARIABLE] = []
t_3 = PetriNet.Transition("name_3", "label_3")
t_3.properties[petri_properties.TRANS_GUARD] = 'A < 5'
t_3.properties[petri_properties.READ_VARIABLE] = 'A'
t_3.properties[petri_properties.WRITE_VARIABLE] = []
t_4 = PetriNet.Transition("name_4", "label_4")
net.transitions.add(t_1)
net.transitions.add(t_2)
net.transitions.add(t_3)
net.transitions.add(t_4)

# add arcs
petri_utils.add_arc_from_to(source, t_1, net)
petri_utils.add_arc_from_to(t_1, p_1, net)
petri_utils.add_arc_from_to(p_1, t_2, net)
petri_utils.add_arc_from_to(p_1, t_3, net)
petri_utils.add_arc_from_to(t_2, p_2, net)
petri_utils.add_arc_from_to(t_3, p_2, net)
petri_utils.add_arc_from_to(p_2, t_4, net)
petri_utils.add_arc_from_to(t_4, sink, net)

# initial and final marking
initial_marking = DataMarking()
initial_marking[source] = 1
final_marking = DataMarking()
final_marking[sink] = 1
dm = DataMarking()
dm[list(initial_marking.keys())[0]] = initial_marking.get(list(initial_marking.keys())[0])

# execution context
ex_cont = {"A": 5}

# playout
max_trace_length = 100
visited_elements = []
all_enabled_trans = [0]
# playout until you are in final marking or exceeded max length or you are in a deadlock
while dm != final_marking and len(visited_elements) < max_trace_length and len(all_enabled_trans) > 0:
    all_enabled_trans = dpn_semantics.enabled_transitions(net, dm, ex_cont)
    for enabled in list(all_enabled_trans):
        if "guard" in enabled.properties:
            if not dpn_semantics.evaluate_guard(enabled.properties["guard"], enabled.properties["readVariable"], dm.data_dict):
                all_enabled_trans.discard(enabled)
    trans = choice(list(all_enabled_trans))
    dm = dpn_semantics.execute(trans, net, dm, ex_cont)
    visited_elements.append(trans)

if dm == final_marking:
    print("Final marking reached!")
elif len(all_enabled_trans) == 0:
    print("Block in deadlock!")
else:
    print("Max length of traces permitted")

print("Visited activities: {}".format(visited_elements))
