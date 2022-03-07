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
    #breakpoint()
    places = dict()
    for place in net.places:
        if len(place.out_arcs) >= 2:
            #breakpoint()
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

#def get_place_from_transition(net, transition, not_silent_dict):
#    # THE METHOD HAS PROBLEMS IN CASE OF A DECISION POINT WHICH HAS ONLY INVISIBLE ACTIVITIES AS OUT ARCS:
#    # IN THAT CASE IT WILL NOT BE PROCESSED - DA CAMBIARE
#    places = list()
#    for place in net.places:
#        for arc in place.out_arcs:
#            if arc.target.label is None:
#                if len(not_silent_dict[arc.target.name]["labels"]) > 0:
#                    if transition in not_silent_dict[arc.target.name]["label"]:
#                        places.append(place)
#            if arc.target.label == transition:
#                places.append(place)
#    return places
def get_place_from_transition(places_map, transition):
    #breakpoint()
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
    #breakpoint()
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

def get_previous_not_silent(place, not_silent):
    breakpoint()
    in_arcs_label = [arc.source.label for arc in place.in_arcs]
    if not None in in_arcs_label:
        not_silent.extend(in_arcs_label)
        return not_silent
    for in_arc in place.in_arcs:
        if not in_arc.source.label is None:
            not_silent.extend(in_arc.source.label)
        else:
            for in_arc_inn in in_arc.target.out_arcs:
                not_silent = get_previous_not_silent(in_arc_inn.source, not_silent)
    return not_silent

def get_silent_activities_map(net):
    #breakpoint()
    not_silent_dict = dict()
    for transition in net.transitions:
        if transition.label is None:
            for out_place in transition.out_arcs:
                not_silent_trans = get_next_not_silent(out_place.target, [])
            not_silent_dict[transition.name] = {"labels": not_silent_trans}
            for in_place in transition.in_arcs:
                not_silent_trans = get_previous_not_silent(in_place.source, [])
            not_silent_trans = list(dict.fromkeys(not_silent_trans))
            not_silent_dict[transition.name]["attributes"] = not_silent_trans
    return not_silent_dict

# Argument (verbose and net_name)
parser = argparse.ArgumentParser()
parser.add_argument("net_name", help="name of the petri net file (without extension)", type=str)

# parse arguments
args = parser.parse_args()
net_name = args.net_name
k = 1

try:
    net, initial_marking, final_marking = pnml_importer.apply("{}.pnml".format(net_name))
except:
    raise Exception("File not found")
log = xes_importer.apply('log-{}.xes'.format(net_name))
#breakpoint()
for trace in log:
    for event in trace:
        #if "A" in event.keys():
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
#breakpoint()
# get a dict of data for every decision point
decision_points_data = dict()
for place in net.places:
    #breakpoint()
    if len(place.out_arcs) >= 2:
        decision_points_data[place.name] = pd.DataFrame()

# fill the data
last_k_events = list()
for trace in log:
    #breakpoint()
    for event in trace:
        #breakpoint()
        #print(event)
        places_from_event = get_place_from_transition(places_events_map, event['concept:name'])  
        if len(places_from_event) == 0:
            last_k_events.append(event)
            if len(last_k_events) > k:
                last_k_events = last_k_events[-k:]
            #print(event)
            #raise Exception("Transition not found")
            continue
        for place_from_event in places_from_event:
            #print(event)
            #if place_from_event.name in decision_points_data.keys():
            #breakpoint()
            last_k_event_dict = dict()
            for last_event in last_k_events:
                #for attr in last_event:
                event_attr = get_attributes_from_event(last_event)
                event_attr.pop('time:timestamp')
                event_attr.pop("concept:name")
                last_k_event_dict.update(event_attr)
            #breakpoint()
            old_df = copy.copy(decision_points_data[place_from_event[0]])
            new_row = pd.DataFrame.from_dict(last_k_event_dict)
            new_row = pd.get_dummies(new_row)
            new_row["target"] = place_from_event[1]
            #breakpoint()
            decision_points_data[place_from_event[0]] = pd.concat([old_df, new_row], ignore_index=True)
        last_k_events.append(event)
        if len(last_k_events) > k:
            last_k_events = last_k_events[-k:]
        #if len(last_k_events) > k:
        #    last_k_events = last_k_events[-k:]
# 
#breakpoint()
for decision_point in decision_points_data.keys():
    #breakpoint()
    print("")
    print(decision_point)
    dataset = decision_points_data[decision_point]
    #breakpoint()
    feature_names = get_feature_names(dataset)
    #breakpoint()
    X = copy.copy(dataset).drop(columns=['target'])
    if net_name == 'one-split-PetriNet-categorical':
        X.fillna(value={"A": -1, "cat_cat_1": 0, "cat_cat_2": 0}, inplace=True)
    elif net_name == 'running-example-Will-BPM':
        X.fillna(value={"policyType_premium": 0, "policyType_normal": 0, "status_approved": 0, "status_rejected": 0}, inplace=True)
    elif net_name == 'running-example-Will-BPM-silent':
        X.fillna(value={"policyType_premium": 0, "policyType_normal": 0, "status_approved": 0, 
            "status_rejected": 0, "communication_email": 0, "communication_letter": 0}, inplace=True)
#    X["A"].fillna(-1, inplace=True)
#    X[["cat_cat_1", "cat_cat_2"]].fillna(0, inplace=True)
    y = copy.copy(dataset)['target']
    dt = DecisionTreeClassifier()
    dt = dt.fit(X, y)
    y_pred = dt.predict(X)
    print("Train accuracy: {}".format(metrics.accuracy_score(y, y_pred)))
    print(export_text(dt))
    rule_extr = extract_rules(dt, feature_names)
    #breakpoint()
    for label_class in rule_extr.keys():
        event_name = places_events_map[decision_point]
        if not isinstance(event_name[label_class], list):
            event_name = event_name[label_class]
        else:
            event_name = label_class
        print(event_name)
        print(rule_extr[label_class])
