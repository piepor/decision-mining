import pm4py
import copy
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import metrics
from pm4py.algo.decision_mining import algorithm as decision_mining
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer

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


net_name =  'one-split-PetriNet-categorical'
k = 1

try:
    net, initial_marking, final_marking = pnml_importer.apply("{}.pnml".format(net_name))
except:
    raise Exception("File not found")
log = xes_importer.apply('log-one-split-PetriNet-categorical.xes')
for trace in log:
    for event in trace:
        if "A" in event.keys():
            event["A"] = float(event["A"])

# discover dpn using pm4py library
#breakpoint()
#dpnet, im, fm = decision_mining.create_data_petri_nets_with_decisions(log, net, initial_marking, final_marking)
#for trans in dpnet.transitions:
#    if "guard" in trans.properties:
#        print(trans.properties["guard"])
#gviz = pn_visualizer.apply(dpnet, im, fm)
#pn_visualizer.view(gviz)

# get a dict of data for every decision point
decision_points_data = dict()
for place in net.places:
    if len(place.out_arcs) >= 2:
        decision_points_data[place.name] = pd.DataFrame()

# fill the data
last_k_events = list()
for trace in log:
    for event in trace:
        place_from_event = get_place_from_transition(net, event['concept:name'])  
        if place_from_event is None:
            raise Exception("Transition not found")
        elif place_from_event.name in decision_points_data.keys():
            last_k_event_dict = dict()
            for last_event in last_k_events:
                for attr in last_event:
                    event_attr = get_attributes_from_event(last_event)
                    event_attr.pop('time:timestamp')
                    event_attr.pop("concept:name")
                    last_k_event_dict.update(event_attr)
            old_df = copy.copy(decision_points_data[place_from_event.name])
            new_row = pd.DataFrame.from_dict(last_k_event_dict)
            #new_row = pd.get_dummies(new_row)
            #breakpoint()
            new_row["target"] = event["concept:name"]
            decision_points_data[place_from_event.name] = pd.concat([old_df, new_row], ignore_index=True)
        last_k_events.append(event)
        if len(last_k_events) > k:
            last_k_events = last_k_events[-k:]
# 
#for decision_point in decision_points_data.keys():
#    dataset = decision_points_data[decision_point]
#    X = copy.copy(dataset).drop(columns=['target'])
#    X.fillna(value={"A": -1, "cat_cat_1": 0, "cat_cat_2": 0}, inplace=True)
##    X["A"].fillna(-1, inplace=True)
##    X[["cat_cat_1", "cat_cat_2"]].fillna(0, inplace=True)
#    y = copy.copy(dataset)['target']
#    dt = DecisionTreeClassifier()
#    dt = dt.fit(X, y)
#    y_pred = dt.predict(X)
#    print("Train accuracy: {}".format(metrics.accuracy_score(y, y_pred)))

def extract_rules(dt):
    text_rules = export_text(dt).split('\n')[:-1]
    extracted_rules = dict()
    one_complete_pass = ""
    #breakpoint()
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
            else:
                #breakpoint()
                single_rule = text_rule.split('|--- ')[1]
                one_complete_pass = "{} & {}".format(one_complete_pass, single_rule)
    return extracted_rules
            
rule_extr = extract_rules(dt)

