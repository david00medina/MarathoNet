import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from random import randint
from openpose import pyopenpose as op

PATH = '../videos'
video_files = list()
for root, dirs, files in os.walk(os.path.relpath(PATH)):
    for file in files:
        video_files.append(os.path.join(root, file))

no_videos = len(video_files)
print(video_files)

params = dict()
params["model_folder"] = "../models/"
params["write_json"] = "../out/openpose"

print(params)

params["video"] = video_files[0]

print(params)

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
opWrapper.execute()
