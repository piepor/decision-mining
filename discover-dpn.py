import pm4py
import copy
import pandas as pd
import argparse
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import metrics
from pm4py.algo.decision_mining import algorithm as decision_mining
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer

def get_map_place_to_events(net):
    places = dict()
    for place in net.places:
        if len(place.out_arcs) >= 2:
            # dictionary containing for every decision point target categories 
            places[place.name] = dict()
            # loop for out arcs
            for arc in place.out_arcs:
                if not arc.target.label is None:
                    places[place.name][arc.target.name] = arc.target.label
                else:
                    silent_out_arcs = arc.target.out_arcs
                    next_not_silent = []
                    for silent_out_arc in silent_out_arcs:
                        next_place_silent = silent_out_arc.target
                        next_not_silent = get_next_not_silent(next_place_silent, next_not_silent)
                    places[place.name][arc.target.name] = next_not_silent
    return places

def get_place_from_transition(places_map, transition):
    places = list() 
    for place in places_map.keys():
        for trans in places_map[place].keys():
            if transition in places_map[place][trans]:
                places.append((place, trans))
    return places

def get_attributes_from_event(event):
    attributes = dict()
    for attribute in event.keys():
        try:
            attributes[attribute] = [float(event[attribute])]
        except:
            attributes[attribute] = [event[attribute]]
    return attributes

def get_next_not_silent(place, not_silent):
    if len(place.in_arcs) > 1:
        return not_silent
    out_arcs_label = [arc.target.label for arc in place.out_arcs]
    if not None in out_arcs_label:
        not_silent.extend(out_arcs_label)
        return not_silent
    for out_arc in place.out_arcs:
        if not out_arc.target.label is None:
            not_silent.extend(out_arc.target.label)
        else:
            for out_arc_inn in out_arc.target.out_arcs:
                not_silent = get_next_not_silent(out_arc_inn.target, not_silent)
    return not_silent

# Argument (verbose and net_name)
parser = argparse.ArgumentParser()
parser.add_argument("net_name", help="name of the petri net file (without extension)", type=str)

# parse arguments
args = parser.parse_args()
net_name = args.net_name
k = 1

try:
    net, initial_marking, final_marking = pnml_importer.apply("models/{}.pnml".format(net_name))
except:
    raise Exception("File not found")
log = xes_importer.apply('data/log-{}.xes'.format(net_name))
for trace in log:
    for event in trace:
        for attr in event.keys():
            try:
                event[attr] = float(event[attr])
            except:
                pass

def extract_rules(dt, feature_names):
    #breakpoint()
    text_rules = export_text(dt)
    for feature_name in feature_names.keys():
            text_rules = text_rules.replace(feature_names[feature_name], feature_name) 
    text_rules = text_rules.split('\n')[:-1]
    extracted_rules = dict()
    one_complete_pass = ""
    #breakpoint()
    tree_level = 0
    for text_rule in text_rules:
        single_rule = text_rule.split('|')[1:]
        if '---' in single_rule[0]:
            one_complete_pass = single_rule[0].split('--- ')[1]
        else:
            if 'class' in text_rule:
                label_name = text_rule.split(': ')[-1]
                if label_name in extracted_rules.keys():
                    extracted_rules[label_name].append(one_complete_pass)
                else:
                    extracted_rules[label_name] = list()
                    extracted_rules[label_name].append(one_complete_pass)
                reset_level_rule = one_complete_pass.split(' & ')
                if len(reset_level_rule) > 1:
                    one_complete_pass = reset_level_rule[:-1][0]
            else:
                #breakpoint()
                single_rule = text_rule.split('|--- ')[1]
                one_complete_pass = "{} & {}".format(one_complete_pass, single_rule)
                tree_level += 1
    return extracted_rules

def get_feature_names(dataset):
    if not isinstance(dataset, pd.DataFrame):
        raise Exception("Not a dataset object")
    features = dict()
    for index, feature in enumerate(dataset.drop(columns=['target']).columns):
        if not feature == 'target':
            features[feature] = "feature_{}".format(index)
    return features
# discover dpn using pm4py library
#breakpoint()
#dpnet, im, fm = decision_mining.create_data_petri_nets_with_decisions(log, net, initial_marking, final_marking)
#for trans in dpnet.transitions:
#    if "guard" in trans.properties:
#        print(trans.properties["guard"])
#gviz = pn_visualizer.apply(dpnet, im, fm)
#pn_visualizer.view(gviz)

# get the map of place and events
places_events_map = get_map_place_to_events(net)
# get a dict of data for every decision point
decision_points_data = dict()
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points_data[place.name] = pd.DataFrame()

# fill the data
last_k_events = list()
for trace in log:
    #breakpoint()
    if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
        trace_attr_row = trace.attributes
        #trace_attr_row.pop('concept:name')
    for event in trace:
        places_from_event = get_place_from_transition(places_events_map, event['concept:name'])  
        if len(places_from_event) == 0:
            last_k_events.append(event)
            if len(last_k_events) > k:
                last_k_events = last_k_events[-k:]
            continue
        #breakpoint()
        for place_from_event in places_from_event:
            last_k_event_dict = dict()
            for last_event in last_k_events:
                event_attr = get_attributes_from_event(last_event)
                event_attr.pop('time:timestamp')
                last_k_event_dict.update(event_attr)
            if len(trace.attributes.keys()) > 1 and 'concept:name' in trace.attributes.keys():
                last_k_event_dict.update(trace_attr_row)
            last_k_event_dict.pop("concept:name")
            old_df = copy.copy(decision_points_data[place_from_event[0]])
            new_row = pd.DataFrame.from_dict(last_k_event_dict)
            new_row = pd.get_dummies(new_row)
            new_row["target"] = place_from_event[1]
            #breakpoint()
            decision_points_data[place_from_event[0]] = pd.concat([old_df, new_row], ignore_index=True)
        last_k_events.append(event)
        if len(last_k_events) > k:
            last_k_events = last_k_events[-k:]

for decision_point in decision_points_data.keys():
    print("")
    print(decision_point)
    #breakpoint()
    dataset = decision_points_data[decision_point]
    feature_names = get_feature_names(dataset)
    X = copy.copy(dataset).drop(columns=['target'])
    if net_name == 'one-split-PetriNet-categorical':
        X.fillna(value={"A": -1, "cat_cat_1": 0, "cat_cat_2": 0}, inplace=True)
    elif net_name == 'running-example-Will-BPM':
        X.fillna(value={"policyType_premium": 0, "policyType_normal": 0, "status_approved": 0, "status_rejected": 0}, inplace=True)
    elif net_name == 'running-example-Will-BPM-silent':
        X.fillna(value={"policyType_premium": 0, "policyType_normal": 0, "status_approved": 0, 
            "status_rejected": 0, "communication_email": 0, "communication_letter": 0}, inplace=True)
    elif net_name == 'running-example-Will-BPM-silent-trace-attr':
        X.fillna(value={"policyType_premium": 0, "policyType_normal": 0, "status_approved": 0, 
            "status_rejected": 0, "communication_email": 0, "communication_letter": 0}, inplace=True)
    y = copy.copy(dataset)['target']
    #breakpoint()
    dt = DecisionTreeClassifier()
    dt = dt.fit(X, y)
    y_pred = dt.predict(X)
    print("Train accuracy: {}".format(metrics.accuracy_score(y, y_pred)))
    print(export_text(dt))
    rule_extr = extract_rules(dt, feature_names)
    for label_class in rule_extr.keys():
        event_name = places_events_map[decision_point]
        if not isinstance(event_name[label_class], list):
            event_name = event_name[label_class]
        else:
            event_name = label_class
        print(event_name)
        print(rule_extr[label_class])
