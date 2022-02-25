import pm4py
import copy
import numpy as np
import datetime
from tqdm import tqdm
from random import choice
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.data_petri_nets import semantics as dpn_semantics
from pm4py.objects.petri_net import properties as petri_properties
from pm4py.objects.petri_net.data_petri_nets.data_marking import DataMarking
from pm4py.objects.log import obj as log_instance
from pm4py.util import xes_constants


verbose = False
verboseprint = print if verbose else lambda *args, **kwargs: None
net_name =  'one-split-PetriNet'
try:
    net, initial_marking, final_marking = pnml_importer.apply("{}.pnml".format(net_name))
except:
    raise Exception("File not found")

# playout
max_trace_length = 100
NO_TRACES = 100
case_id_key = xes_constants.DEFAULT_TRACEID_KEY
activity_key = xes_constants.DEFAULT_NAME_KEY
timestamp_key = xes_constants.DEFAULT_TIMESTAMP_KEY
curr_timestamp = datetime.datetime.now()

# playout until you are in final marking or exceeded max length or you are in a deadlock
all_visited = []
all_context = {"A": []}
for i in tqdm(range(NO_TRACES)):
    #breakpoint()
    # reset marking to initial
    dm = DataMarking()
    dm[list(initial_marking.keys())[0]] = initial_marking.get(list(initial_marking.keys())[0])
    visited_elements = []
    all_enabled_trans = [0]
    # execution context
    a = np.random.uniform(0, 10, 1)
    ex_cont = {"A": a[0]}
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
        verboseprint("Final marking reached!")
    elif len(all_enabled_trans) == 0:
        verboseprint("Block in deadlock!")
    else:
        verboseprint("Max length of traces permitted")

    verboseprint("Visited activities: {}".format(visited_elements))
    all_visited.append(tuple(visited_elements))
    all_context["A"].append(ex_cont["A"])

log = log_instance.EventLog()
for index, element_sequence in tqdm(enumerate(all_visited)):
    trace = log_instance.Trace()
    trace.attributes[case_id_key] = str(index)
    ex_cont = all_context['A'][index]
    for element in element_sequence:
        if type(element) is PetriNet.Transition:
            event = log_instance.Event()
            event[activity_key] = element.label
            event[timestamp_key] = curr_timestamp
            event['A'] = copy.copy(ex_cont)
            trace.append(event)
            # increase 5 minutes
            curr_timestamp = curr_timestamp + datetime.timedelta(minutes=5)
    log.append(trace)

xes_exporter.apply(log, 'log-{}.xes'.format(net_name))
