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
    """
    Create a dictionary with the labels and their corresponding color.
    :param class_list: list of classes.
    :return: Dictionary with the labels, later used in process_model_output.
    """
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))
    labels = {}
    for key, value in classes.items():
        labels[value] = (key, COLORS[value])
    return labels


def process_model_output(outputs, detection_threshold, labels):
    """
    Process the output of the model to get the bounding boxes, labels and probabilities. Given the detection threshold and labels dictionary.

    :param outputs: Outputs of the model in eval model.
    :param detection_threshold: value between 0 and 1.
    :param labels: Dictionary with the labels.
    :return:
    """

    pred_scores = outputs[0]['scores'].detach().cpu()
    pred_bboxes = outputs[0]['boxes'].detach().cpu()
    pred_labels = outputs[0]['labels'].detach().cpu()
    # Convert labels list to a tensor
    labels_tensor = th.tensor(list(labels.keys()), dtype=th.long)
    mask = (pred_scores > detection_threshold) & pred_labels.unsqueeze(-1).eq(labels_tensor).any(dim=1)
    output_probs = pred_scores[mask].flatten()
    output_bboxes = pred_bboxes[mask]
    output_labels = pred_labels[mask].flatten()
    output_classes = [labels[label.item()] for label in output_labels]
    return output_probs, output_bboxes, output_labels, output_classes
