import sys
import cv2
import math
import numpy as np
import logging
import tensorflow as tf
from keras import models

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s: %(message)s')
LOGGER = logging.getLogger("ClosingCredits")
LIMIT_POINT = 0.8
ACCEPTANCE_THRESHOLD = 0.9
AREAS = 3
FRAME_PER_SECOND = 5

if len(sys.argv) < 3:
    LOGGER.error("Missing arguments! You should provide to arguments to the script. First, path to the video and then path to the model.")
    exit()

video_path = sys.argv[1]
model_path = sys.argv[2]

model = models.load_model(model_path) # loading the pretrained model
metadata = [] # Contains the timestamp (in milliseconds) and frame ID of all frames fed into the model

capture = cv2.VideoCapture(video_path)

width = capture.get(3)
height = capture.get(4)
cutoff = int((width - height)/(AREAS-1))
frame_rate = capture.get(5)
frame_step = math.ceil(frame_rate/FRAME_PER_SECOND)
total_frames = capture.get(7)


LOGGER.info(f"Movie metadata - width: {width}, height: {height}, framerate: {frame_rate}, total_frames: {total_frames}")

def get_starting_index(estimates, window_size=25):
    window = np.zeros((window_size,))
    count = 0
    for i in range(estimates.shape[0]-window_size):
        if count == 10:
            return index
        ratio_in_window = np.sum(estimates[i:(i+window_size)] != window)/window_size
        if ratio_in_window > ACCEPTANCE_THRESHOLD:
            if count == 0:
                index = i
            count += 1
        else:
            count = 0
            index = None
    return estimates.shape[0]-1

frame_set = []
for i in range(AREAS):
    frames = []
    current_frame = total_frames - 1
    while current_frame > total_frames * LIMIT_POINT:
        capture.set(1, int(current_frame))
        frame_info = {"time_progress": capture.get(0),
                      "frame_id": capture.get(1)}
        ret, frame = capture.read()
        if ret == True:
            metadata.append(frame_info)
            step = i*cutoff
            frame = frame[:, step:int(height)+step, :]
            frame = cv2.resize(frame, (224, 224))/255.0
            frames.append(frame)
        current_frame -= frame_step
    frame_set.append(frames)

LOGGER.info("finished preprocessing frames.")
capture.release()

estimates_set = []
for i in range(AREAS):
    frames = np.array(frame_set[i])
    prediction_classes = model.predict_classes(frames)
    estimates = np.array([x[0] for x in prediction_classes])
    estimates_set.append(estimates)

LOGGER.info("prediction stage is done.")

def zero_pad(variable):
    if int(variable) < 10:
        return f'0{int(variable)}'
    else:
        return str(int(variable))

credits_info = metadata[max([get_starting_index(estimates) for estimates in estimates_set])]
hour = credits_info['time_progress']//3600000
minute = (credits_info['time_progress'] - hour * 3600000)//60000
second = (credits_info['time_progress'] - hour * 3600000 - minute * 60000)//1000
hour, minute, second = zero_pad(hour), zero_pad(minute), zero_pad(second)
ms = int(math.floor(credits_info['time_progress']) % 1000)

LOGGER.info(f"Credits started rolling at {hour}:{minute}:{second}.{ms}, at {credits_info['frame_id']} frame.")
