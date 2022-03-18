import numpy as np
import operator


class DecisionNode:
    def __init__(self, parent_attribute_value, attribute_type, attribute):
        self._label = parent_attribute_value
        self._attribute = attribute
        self._attribute_type = attribute_type
        self._test_function = get_test_function(attribute_type)
        self._childs = list()
    
    def get_label(self):
        return self._label

    def get_attribute(self):
        return self._attribute

    def get_childs(self):
        return self._childs

    def add_child(self, child):
        self._childs.append(child)

    def run_test(self, attr_value):
        attr_name = self._attribute.split()[0]
        if self._attribute_type == 'categorical':
            result_test = "{} = {}".format(attr_name, attr_value)
        elif self._attribute_type == 'continuous':
            threshold = float(self._attribute.split(':')[1])
            if self.run_test(threshold, attr_value):
                result_test = "{} < {}".format(attr_name, threshold)
            else:
                result_test = "{} >= {}".format(attr_name, threshold)
        elif self._attribute_type == 'boolean':
            result_test = "{} = {}".format(attr_name, attr_value)
        else:
            raise Exception("Attribute type not understood")
        return result_test

    def get_child(self, attr_value):
        return next((child for child in self._childs if child.get_label() == self.run_test(attr_value)), None)


class LeafNode:
    def __init__(self, classes, parent_attribute_value):
        self._classes = classes
        self._label = parent_attribute_value
        self._label_class = max(self._classes.items(), key=operator.itemgetter(1))[0]

    def get_class_names(self):
        return list(self._classes.keys())

    def get_class_examples(self, class_name):
        return self._classes[class_name]

    def get_label(self):
        return self._label

    def predict_class(self):
        return self._label_class

#def categorical_test_function(choices, attr_value):
#    return next((choice for choice in choices if choice.get_label() == attr_value), None)

#def continuous_test_function_higher_equal(threshold, attr_value):
#    return attr_value >= threshold

def continuous_test_function_lower(threshold, attr_value):
    return attr_value < threshold

def boolean_test_function(attr_value):
    return attr_value == True

def get_test_function(attribute_type):
    if attribute_type == 'continuous':
        return continuous_test_function
    elif attribute_type == 'boolean' or attribute_type == 'categorical':
        return None
    else:
        raise Exception("Attribute value not supported")

leaf_data1 = {'cip':32}
leaf_data2 = {'ciop':22}

decision_point = DecisionNode('root', 'categorical', 'color')
leaf_node1 = LeafNode(leaf_data1, 'color = brown')
leaf_node2 = LeafNode(leaf_data2, 'color = black')
decision_point.add_child(leaf_node1)
decision_point.add_child(leaf_node2)
