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

if len(sys.argv) < 2:
    LOGGER.error("Missing arguments! You should provide to arguments to the script. First, path to the video and then path to the model.")
    exit()

video_path = sys.argv[1]
model_path = sys.argv[2]

model = models.load_model(model_path) # loading the pretrained model
metadata = [] # Contains the timestamp (in milliseconds) and frame ID of all frames fed into the model
frames = [] # Contains the frames themselves

capture = cv2.VideoCapture(video_path)

width = capture.get(3)
height = capture.get(4)
cutoff = int((width - height)/2)
frame_rate = capture.get(5)
total_frames = capture.get(7)

LOGGER.info(f"Movie metadata - width: {width}, height: {height}, framerate: {frame_rate}"
            f"total_frames: {total_frames} currentframe: {capture.get(1)}")

def get_starting_index(estimates, window_size=50):
    window = np.zeros((window_size,))
    count = 0
    for i in range(estimates.shape[0]-window_size):
        if count == 10:
            return index
        if np.sum(estimates[i:(i+window_size)] == window)/window_size > 0.95:
            if count == 0:
                index = i
            count += 1
        else:
            count = 0
            index = None
    return None


while(capture.isOpened()):
    frame_info = {"time_progress": capture.get(0),
                  "frame_id": capture.get(1)}
    ret, frame = capture.read()
    if ret != True:
        break
    if frame_info['frame_id']/total_frames > 0.75 and frame_info['frame_id'] % math.floor(frame_rate/10) == 0:
        metadata.append(frame_info)
        frame = frame[:, cutoff:-cutoff, :]
        frame = cv2.resize(frame, (224, 224))/255.0
        frames.append(frame)

LOGGER.info("finished preprocessing frames.")

frames = np.array(frames)
capture.release()

prediction_classes = model.predict_classes(frames)
estimates = np.array([x[0] for x in prediction_classes])
LOGGER.info("prediction stage is done.")

def zero_pad(variable):
    if int(variable) < 10:
        return f'0{int(variable)}'
    else:
        return str(int(variable))

credits_info = metadata[get_starting_index(estimates)]
hour = credits_info['time_progress']//3600000
minute = (credits_info['time_progress'] - hour * 3600000)//60000
second = (credits_info['time_progress'] - hour * 3600000 - minute * 60000)//1000
hour, minute, second = zero_pad(hour), zero_pad(minute), zero_pad(second)
ms = int(math.floor(credits_info['time_progress']) % 1000)

LOGGER.info(f"Credits started rolling at {hour}:{minute}:{second}.{ms}, at {credits_info['frame_id']} frame.")
