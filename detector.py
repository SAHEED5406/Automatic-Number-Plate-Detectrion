from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import easyocr

import cv2
import numpy as np

class Detector:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file('config.yml')
        self.cfg.MODEL.WEIGHTS = 'model_final.pth'

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = 'cpu'

        self.predictor = DefaultPredictor(self.cfg)

        self.reader = easyocr.Reader(['en'])

    def onImage(self,imagePath):
        self.image = cv2.imread(imagePath)

        predictions = self.predictor(self.image)

        self.boxes = predictions['instances'].pred_boxes.tensor
        self.num_of_boxes = len(self.boxes)

        viz = Visualizer(self.image[:,:,::-1],metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),scale=0.2)

        output = viz.draw_instance_predictions(predictions['instances'].to("cpu"))

        cv2.imshow('Result',output.get_image()[:,:,::-1])
        cv2.waitKey(0)
        self.cropImage(self.image)
        #self.textReader(cropped_image=cropped_image)

    def cropImage(self,image):
        self.image = image
        for box in range(self.num_of_boxes):

              y1 = int(self.boxes[box,0])
              x1 = int(self.boxes[box,1])
              y2 = int(self.boxes[box,2])
              x2 = int(self.boxes[box,3])

              crop_image = self.image[x1:x2,y1:y2]
              self.textReader(cropped_image=crop_image)

              #return crop_image
       

    def textReader(self,cropped_image):

        self.cropped_image = cropped_image

        extract_text = self.reader.readtext(self.cropped_image)

        for text in range(len(extract_text)):
            print(extract_text[text][1])
            
        cv2.imshow('crop',cropped_image)
        cv2.waitKey(0)
