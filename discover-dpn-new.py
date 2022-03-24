import pm4py
import os
from time import time
import numpy as np
import copy
import pandas as pd
import argcomplete, argparse
#from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import metrics
from pm4py.algo.decision_mining import algorithm as decision_mining
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer
from utils import detect_loops, get_in_place_loop, get_next_not_silent
from utils import get_out_place_loop, get_map_place_to_events
from utils import get_place_from_transition
from utils import get_attributes_from_event, get_feature_names
from utils import extract_rules, get_map_transitions_events
from DecisionTree import DecisionTree

def ModelCompleter(**kwargs):
    return [name.split('.')[0] for name in os.listdir('models')]
# Argument (verbose and net_name)
parser = argparse.ArgumentParser()
parser.add_argument("net_name", help="name of the petri net file (without extension)", type=str).completer = ModelCompleter
argcomplete.autocomplete(parser)
# parse arguments
args = parser.parse_args()
net_name = args.net_name
k = 2

try:
    net, initial_marking, final_marking = pnml_importer.apply("models/{}.pnml".format(net_name))
except:
    raise Exception("File not found")
log = xes_importer.apply('data/log-{}.xes'.format(net_name))
for trace in log:
    for event in trace:
        for attr in event.keys():
           # if attr == 'appeal':
           #     breakpoint()
            if not isinstance(event[attr], bool):
                try:
                    event[attr] = float(event[attr])
                except:
                    pass

loop_vertex = detect_loops(net)
in_places_loops = get_in_place_loop(net, loop_vertex)
in_transitions_loop = list()        
for place_name in in_places_loops:
    place = [place_net for place_net in net.places if place_net.name == place_name]
    place = place[0]
    for out_arc in place.out_arcs:
        out_transitions = get_next_not_silent(place, [], loop_vertex) 
        in_transitions_loop.extend(out_transitions)
out_places_loops = get_out_place_loop(net, loop_vertex)
#breakpoint()
# get the map of places and events
places_events_map = get_map_place_to_events(net, loop_vertex)
# get the map of transitions and events
trans_events_map = get_map_transitions_events(net)
# get a dict of data for every decision point
decision_points_data = dict()
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points_data[place.name] = pd.DataFrame()

# fill the data
tic = time()
attributes_map = {'amount': 'continuous', 'policyType': 'categorical', 'appeal': 'boolean', 'status': 'categorical', 'communication': 'categorical'}
for trace in log:
    #breakpoint()
    if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
        trace_attr_row = trace.attributes
        #trace_attr_row.pop('concept:name')
    last_k_events = list()
    for event in trace:
        #tic = time()
        trans_from_event = trans_events_map[event["concept:name"]]
        if len(last_k_events) == 0:
            last_event_name = 'None'
        else:
            last_event_name = last_k_events[-1]['concept:name']
        last_event_trans = trans_events_map[last_event_name]
        places_from_event = get_place_from_transition(places_events_map, event['concept:name'], loop_vertex, last_event_trans, 
                in_transitions_loop, out_places_loops, in_places_loops, trans_from_event)  
        if len(places_from_event) == 0:
            last_k_events.append(event)
            if len(last_k_events) > k:
                last_k_events = last_k_events[-k:]
            continue
        #toc = time()
        #print("first part of first part: {}".format(toc-tic))
        #breakpoint()
        #tic = time()
        for place_from_event in places_from_event:
            last_k_event_dict = dict()
            #tic_1 = time()
            for last_event in last_k_events:
                event_attr = get_attributes_from_event(last_event)
                event_attr.pop('time:timestamp')
                last_k_event_dict.update(event_attr)
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                last_k_event_dict.update(trace_attr_row)
            last_k_event_dict.pop("concept:name")
            #toc_1 = time()
            #print("first part of second part of first part: {}".format(toc_1-tic_1))
            #tic_1 = time()
            #old_df = copy.copy(decision_points_data[place_from_event[0]])
            #tic_2 = time()
            new_row = pd.DataFrame.from_dict(last_k_event_dict)
            #tic_3 = time()
            #new_row = pd.get_dummies(new_row)
            #toc_3 = time()
            #print("pandas get dummies: {}".format(toc_3-tic_3))
            new_row["target"] = place_from_event[1]
#            if new_row["target"][0] == 'trans_G':
#                breakpoint()
            decision_points_data[place_from_event[0]] = pd.concat([decision_points_data[place_from_event[0]], new_row], ignore_index=True)
            #toc_2 = time()
            #print("pandas create row: {}".format(toc_2-tic_2))
            #toc_1 = time()
            #print("second part of second part of first part: {}".format(toc_1-tic_1))
        #breakpoint()
        last_k_events.append(event)
        if len(last_k_events) > k:
            last_k_events = last_k_events[-k:]
        #toc = time()
        #print("second part of first part: {}".format(toc-tic))
#toc = time()
#print("first part: {}".format(toc-tic))

#tic = time()
for decision_point in decision_points_data.keys():
    print("")
    print(decision_point)
    dataset = decision_points_data[decision_point]
    #breakpoint()
    feature_names = get_feature_names(dataset)
    dt = DecisionTree(attributes_map)
    dt.fit(dataset)
    y_pred = dt.predict(dataset.drop(columns=['target']))
    print("Train accuracy: {}".format(metrics.accuracy_score(dataset['target'], y_pred)))
    print(dt.extract_rules())
toc = time()
print("Total time: {}".format(toc-tic))
