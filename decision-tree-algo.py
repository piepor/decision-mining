import numpy as np
import operator
import pandas as pd


class DecisionNode:
    def __init__(self, parent_attribute_value, attribute_type, attribute):
        self._label = parent_attribute_value
        self._attribute = attribute
        self._attribute_type = attribute_type
        if self._attribute_type == 'continuous':
            try:
                self._threshold = float(self._attribute.split(':')[1])
            except:
                raise Exception("Threshold must be a numerica value")
        elif not self._attribute_type in ['categorical', 'boolean']:
            raise Exception("Attribute value not supported")
        self._childs = set()
    
    def get_label(self):
        return self._label

    def get_attribute(self):
        return self._attribute

    def get_childs(self):
        return self._childs

    def add_child(self, child):
        self._childs.add(child)

    def run_test(self, attr_value):
        attr_name = self._attribute.split(':')[0]
        if self._attribute_type == 'categorical':
            result_test = "{} = {}".format(attr_name, attr_value)
        elif self._attribute_type == 'continuous':
            if self.continuous_test_function_lower(attr_value):
                result_test = "{} < {}".format(attr_name, self._threshold)
            else:
                result_test = "{} >= {}".format(attr_name, self._threshold)
        elif self._attribute_type == 'boolean':
            result_test = "{} = {}".format(attr_name, attr_value)
        else:
            raise Exception("Attribute type not understood")
        return result_test

    def get_child(self, attr_value):
        return next((child for child in self._childs if child.get_label() == self.run_test(attr_value)), None)

    def continuous_test_function_lower(self, attr_value):
        return attr_value < self._threshold


class LeafNode(object):
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


class DecisionTree:
    def __init__(self):
        self._nodes = set()
        self._root_node = None

    def add_node(self, node, parent_node):
        #parent_node = next(( node for node in self._nodes if node.get_label() == parent_label), None)
        #breakpoint()
        self._nodes.add(node)
        if not parent_node is None:
            parent_node.add_child(node)
        elif node.get_label() == 'root':
            self._root_node = node
        else:
            raise Exception('Parent label not present in the tree')

    def _predict(self, row_in, node):
        #breakpoint()
        attribute = node.get_attribute().split(':')[0]
        child = node.get_child(row_in[attribute])
        if isinstance(child, LeafNode):
            return child.predict_class()
        else:
            return self._predict(row_in, child)


    def predict(self, data_in):
        #breakpoint()
        attribute = self._root_node.get_attribute()
        preds = list()
        for index, row in data_in.iterrows():
            child = self._root_node.get_child(row[attribute])
            if isinstance(child, LeafNode):
                preds.append(child.predict_class())
            else:
                preds.append(self._predict(row, child))
        return preds



leaf_data1 = {'ciop':32}
leaf_data2 = {'cip':22}
leaf_data3 = {'cip':32}
leaf_data4 = {'ciop':22}
leaf_data5 = {'cip':32}
leaf_data6 = {'ciop':22}
leaf_data7 = {'ciop':32}
leaf_data8 = {'cip':22}

dt = DecisionTree()
#breakpoint()
decision_point_root = DecisionNode('root', 'categorical', 'color')
decision_point_1 = DecisionNode('color = brown', 'continuous', 'amount:200')
decision_point_2 = DecisionNode('color = black', 'continuous', 'amount:500')
decision_point_3 = DecisionNode('amount < 200.0', 'boolean', 'isStupid')
decision_point_4 = DecisionNode('amount >= 200.0', 'boolean', 'isStupid')
decision_point_5 = DecisionNode('amount < 500.0', 'boolean', 'isStupid')
decision_point_6 = DecisionNode('amount >= 500.0', 'boolean', 'isStupid')
leaf_node1 = LeafNode(leaf_data1, 'isStupid = True')
leaf_node2 = LeafNode(leaf_data2, 'isStupid = False')
leaf_node3 = LeafNode(leaf_data3, 'isStupid = True')
leaf_node4 = LeafNode(leaf_data4, 'isStupid = False')
leaf_node5 = LeafNode(leaf_data5, 'isStupid = True')
leaf_node6 = LeafNode(leaf_data6, 'isStupid = False')
leaf_node7 = LeafNode(leaf_data7, 'isStupid = True')
leaf_node8 = LeafNode(leaf_data8, 'isStupid = False')
dt.add_node(decision_point_root, None)
dt.add_node(decision_point_1, decision_point_root)
dt.add_node(decision_point_2, decision_point_root)
dt.add_node(decision_point_3, decision_point_1)
dt.add_node(decision_point_4, decision_point_1)
dt.add_node(decision_point_5, decision_point_2)
dt.add_node(decision_point_6, decision_point_2)
dt.add_node(leaf_node1, decision_point_3)
dt.add_node(leaf_node2, decision_point_3)
dt.add_node(leaf_node3, decision_point_4)
dt.add_node(leaf_node4, decision_point_4)
dt.add_node(leaf_node5, decision_point_5)
dt.add_node(leaf_node6, decision_point_5)
dt.add_node(leaf_node7, decision_point_6)
dt.add_node(leaf_node8, decision_point_6)
df = pd.DataFrame({'isStupid': [True, False, True, False, True, False, True, False], 
    'color': ['brown', 'brown', 'brown', 'brown', 'black', 'black','black', 'black'], 
    'amount': [100, 50, 300, 400, 250, 450, 550, 1000]})
out_pred = dt.predict(df)
print(out_pred)
