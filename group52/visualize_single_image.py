import csv
import os

import cv2
import numpy as np
import torch

from group52.image_helper import draw_boxes
from group52.retinanet_helper import load_classes, process_model_output, COLORS


def visualize_images_in_folder(image_path, model_path, class_list, detection_threshold=0.6):
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))
    labels = {}
    for key, value in classes.items():
        labels[value] = (key, COLORS[value])

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.training = False
    model.eval()

    for img_name in os.listdir(image_path):
        image_array = cv2.imread(os.path.join(image_path, img_name))
        if image_array is None:
            print("Image not found: " + img_name)
            continue
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).permute(0, 3, 1, 2).div(255.0)

        with torch.no_grad():
            outputs = model(image_tensor)
            output_boxes, output_categories, output_classes = process_model_output(outputs, detection_threshold, labels)

        image = draw_boxes(output_boxes, output_classes, image_array)
        cv2.namedWindow('detections', cv2.WINDOW_NORMAL)
        cv2.imshow('detections', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    visualize_images_in_folder('../data/our_sony/processed_images/', '../data/retinanetv2_object_detector.pickle', '../data/class_list.csv')
