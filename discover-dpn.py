import pm4py
import copy
import pandas as pd
from pm4py.algo.decision_mining import algorithm as decision_mining
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer

def get_place_from_transition(net, transition):
    for place in net.places:
        for arc in place.out_arcs:
            if arc.target.label == transition:
                return place
    return None

def get_attributes_from_event(event):
    attributes = dict()
    for attribute in event.keys():
        try:
            attributes[attribute] = [float(event[attribute])]
        except:
            attributes[attribute] = [event[attribute]]
    return attributes


net_name =  'one-split-PetriNet'
try:
    net, initial_marking, final_marking = pnml_importer.apply("{}.pnml".format(net_name))
except:
    raise Exception("File not found")
log = pm4py.read_xes('log-one-split-PetriNet.xes')

# discover dpn using pm4py library
dpnet, im, fm = decision_mining.create_data_petri_nets_with_decisions(log, net, initial_marking, final_marking)
for trans in dpnet.transitions:
    if "guard" in trans.properties:
        print(trans.properties["guard"])
gviz = pn_visualizer.apply(dpnet, im, fm)
pn_visualizer.view(gviz)

# get a dict of data for every decision point
decision_points_data = dict()
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points_data[place.name] = pd.DataFrame()

# fill the data
for trace in log:
    for event in trace:
        place_from_event = get_place_from_transition(net, event['concept:name'])  
        if place_from_event is None:
            raise Exception("Transition not found")
        elif place_from_event.name in decision_points_data.keys():
            event_attr = get_attributes_from_event(event)
            event_attr.pop('time:timestamp')
            old_df = copy.copy(decision_points_data[place_from_event.name])
            new_row = pd.DataFrame.from_dict(event_attr)
            decision_points_data[place_from_event.name] = pd.concat([old_df, new_row], ignore_index=True)
