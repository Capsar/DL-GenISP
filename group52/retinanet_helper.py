import csv

import numpy as np
import torch as th

np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(4, 3))


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def create_label_dictionary(class_list):
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))
    labels = {}
    for key, value in classes.items():
        labels[value] = (key, COLORS[value])
    return labels


def process_model_output(outputs, detection_threshold, labels):
    pred_scores = outputs[0]['scores'].detach().cpu()
    pred_bboxes = outputs[0]['boxes'].detach().cpu()
    pred_labels = outputs[0]['labels'].detach().cpu()
    output_indices = (pred_scores > detection_threshold).nonzero()
    print(output_indices)
    output_probs = pred_scores[output_indices].flatten()
    output_bboxes = pred_bboxes[output_indices]
    output_labels = pred_labels[output_indices].flatten()
    return output_probs, output_bboxes, output_labels
