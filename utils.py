from pm4py.objects.petri_net.importer import importer as pnml_importer

net, initial_marking, final_marking = pnml_importer.apply("running-example-Will-BPM-silent.pnml")
places = list(net.places)

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
    breakpoint()
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

not_silent_dict = get_silent_activities_map(net)
