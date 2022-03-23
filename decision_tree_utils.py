import numpy as np

def get_total_threshold(data, local_threshold):
    return data[data.le(local_threshold)].max()

def class_entropy(data):
    ops = data.value_counts() / len(data)
    return - np.sum(ops * np.log2(ops))

def get_split_gain(data_in, attr_type):
    attr_name = [col for col in data_in.columns if col != 'target'][0]
    split_gain = class_entropy(data_in['target'])
    split_info = 0
    local_threshold = None
    if attr_type in ['categorical', 'boolean']:
        data_counts = data_in[attr_name].value_counts()
        total_count = len(data_in)
        for attr_value in data_in[attr_name].unique():
            #breakpoint()
            freq_attr = data_counts[attr_value] / total_count
            split_gain -= freq_attr * class_entropy(data_in[data_in[attr_name] == attr_value]['target'])
            split_info += - freq_attr * np.log2(freq_attr)
        #breakpoint()
        gain_ratio = split_gain / split_info
    elif attr_type == 'continuous':
        data_in_sorted = data_in[attr_name].sort_values()
        thresholds = data_in_sorted - (data_in_sorted.diff() / 2)
        max_gain = 0
        for threshold in thresholds[1:]:
            #breakpoint()
            freq_attr = data_in[data_in[attr_name] <= threshold][attr_name].count() / len(data_in)
            class_entropy_low = class_entropy(data_in[data_in[attr_name] <= threshold]['target'])
            class_entropy_high = class_entropy(data_in[data_in[attr_name] > threshold]['target'])
            split_gain_threshold = split_gain - freq_attr * class_entropy_low - (1 - freq_attr) * class_entropy_high 
            split_info = - freq_attr * np.log2(freq_attr) - (1 - freq_attr) * np.log2(1 - freq_attr) 
            gain_ratio_temp = split_gain_threshold / split_info
            if gain_ratio_temp > max_gain:
                local_threshold = threshold
                max_gain = gain_ratio_temp
            gain_ratio = max_gain
            
    return gain_ratio, local_threshold
