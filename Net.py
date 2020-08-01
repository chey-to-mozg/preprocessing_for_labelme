from tensorflow.keras.models import load_model
import segmentation_models as sm
sm.set_framework('tf.keras')
import cv2
import numpy as np
import os.path as osp
import os

import json



class Net:


    def __init__(self, path_to_model):
        self.path_to_model = path_to_model

        self.load_and_compile()

    def load_and_compile(self):
        self.model = load_model(self.path_to_model, compile=False)


    def find_objects(self, path_to_file):
        image = cv2.imread(path_to_file, -1)

        self.height = image.shape[0]
        self.width = image.shape[1]
        self.file_name = osp.split(path_to_file)[-1]
        self.path_to_file = osp.split(path_to_file)[0]

        image = np.expand_dims(image, axis=2)
        mask = self.model.predict(np.array([image]))[0, ..., 0]

        return mask.astype(np.float32)


    def process_mask(self, mask):
        mask[mask>0.6] = 1
        mask[mask <= 0.6] = 0
        mask = np.round(mask).astype(np.float32)

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(mask, kernel, iterations=2)
        erosion = cv2.erode(dilation, kernel, iterations=2)
        erosion = erosion * 255
        erosion = np.asarray(erosion, 'uint8')
        contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def make_json(self, contours):
        new_json = dict()
        new_json["version"] = "4.4.0"
        new_json["flags"] = {}

        leafs = []

        for leaf in contours[1:]:
            leaf_dict = dict()
            leaf_dict['label'] = "Leaf"
            points = []
            for point in leaf:
                points.append(point[0].tolist())
            leaf_dict["points"] = points
            leaf_dict["group_id"] = None
            leaf_dict["shape_type"] = "polygon"
            leaf_dict["glags"] = {}

            leafs.append(leaf_dict)

        new_json["shapes"] = leafs
        new_json["imagePath"] = self.file_name
        new_json["imageData"] = None
        new_json["imageHeight"] = self.height
        new_json["imageWidth"] = self.width


        with open(f'{self.path_to_file}/{self.file_name.split(".")[0]}.json', 'w') as json_file:
            json.dump(new_json, json_file)


    def process_file(self, filename):
        if self.check_for_json(filename):
            print(f'file {filename} already processed')
            return
        mask = self.find_objects(filename)
        contours = self.process_mask(mask)
        self.make_json(contours)
        print(f'file {filename} processed')


    def check_for_json(self, filename):
        json_name = filename.split('.')[0] + '.json'
        if osp.exists(json_name):
            return True
        return False


    def process_directory(self, dir_name):
        files = os.listdir(dir_name)

        for file in files:
            if file.endswith(".json"):
                continue

            self.process_file(f'{dir_name}/{file}')