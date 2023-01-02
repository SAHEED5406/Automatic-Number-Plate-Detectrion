from detector import *
import os

path = "D:\DATA SCIENCE\LEARNING\Deep learning\Automatic Number Plate Detection\Own Data\Inference data"

for file in os.listdir(path):
    if 'jpg' in file :
        detector = Detector()
        detector.onImage(os.path.join(path,file))