import csv

import numpy as np

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
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy().astype(np.int32)
    pred_labels = outputs[0]['labels'].detach().cpu().numpy()

    output_boxes = []
    output_categories = []
    output_classes = []
    for score, bbox, label in zip(pred_scores, pred_bboxes, pred_labels):
        if score > detection_threshold and label in labels:
            output_boxes.append(bbox)
            output_categories.append(label)
            output_classes.append(labels[label])

    return np.array(output_boxes), np.array(output_categories), output_classes
