from pm4py.objects.petri_net.importer import importer as pnml_importer

net, initial_marking, final_marking = pnml_importer.apply("running-example-Will-BPM-silent.pnml")
places = list(net.places)

def get_next_not_silent(place, not_silent):
    #breakpoint()
    if len(place.in_arcs) > 1:
        return not_silent
    out_arcs_label = [arc.target.label for arc in place.out_arcs]
    if not None in out_arcs_label:
        return not_silent.extend(out_arcs_label)
    for out_arc in place.out_arcs:
        if not out_arc.target.label is None:
            not_silent.append(out_arc.target.label)
        else:
            for out_arc_inn in out_arc.target.out_arcs:
                get_next_not_silent(out_arc_inn.target, not_silent)
    return not_silent

def get_silent_activities_map(net):
    breakpoint()
    not_silent = dict()
    for transition in net.transitions:
        if transition.label is None:
            for out_place in transition.out_arcs:
                not_silent_trans = get_next_not_silent(out_place.target, [])
            not_silent[transition.name] = not_silent_trans
    return not_silent
