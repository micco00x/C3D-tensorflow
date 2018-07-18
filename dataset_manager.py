# TODO: open the json file created in the first part of the elective and create
# mappings from label to ids and from filename and path
#                              +
# functions needed to create the training batches

import json
import os
import cv2
import numpy as np
import random
import pprint
import sys

import PoseEstimation

# There are also "G2-1", "G2-3", "G2-4" (torso:bend forward, torso:bend left, torso:bend right) in dataset_testing.json
DS_CLASSES = ['head:Lean forward', 'head:Raise the head', 'head:Roll left',
              'head:Roll right', 'head:Turn bottom-left', 'head:Turn bottom-right',
              'head:Turn left', 'head:Turn right', 'head:Turn top-left',
              'head:Turn top-right', 'l-arm:Elbow extension', 'l-arm:Elbow flexion',
              'l-arm:Roll the wrist', 'l-arm:Shoulder abduction',
              'l-arm:Shoulder adduction', 'l-arm:Shoulder extension',
              'l-arm:Shoulder external rotation',
              'l-arm:Shoulder external rotation and elbow extension',
              'l-arm:Shoulder external rotation and elbow flexion',
              'l-arm:Shoulder flexion', 'l-arm:Shoulder internal rotation',
              'l-arm:Shoulder internal rotation and elbow extension',
              'l-arm:Shoulder internal rotation and elbow flexion',
              'l-leg:Hip abduction', 'l-leg:Hip adduction', 'l-leg:Hip extension',
              'l-leg:Hip external rotation', 'l-leg:Hip flexion', 'l-leg:Hip internal rotation',
              'l-leg:Knee extension', 'l-leg:Knee flexion', 'r-arm:Elbow extension',
              'r-arm:Elbow flexion', 'r-arm:Roll the wrist', 'r-arm:Shoulder abduction',
              'r-arm:Shoulder adduction', 'r-arm:Shoulder extension',
              'r-arm:Shoulder external rotation',
              'r-arm:Shoulder external rotation and elbow extension',
              'r-arm:Shoulder external rotation and elbow flexion',
              'r-arm:Shoulder flexion', 'r-arm:Shoulder internal rotation',
              'r-arm:Shoulder internal rotation and elbow extension',
              'r-arm:Shoulder internal rotation and elbow flexion', 'r-leg:Hip abduction',
              'r-leg:Hip adduction', 'r-leg:Hip extension', 'r-leg:Hip external rotation',
              'r-leg:Hip flexion', 'r-leg:Hip internal rotation', 'r-leg:Knee extension',
              'r-leg:Knee flexion']

def search_path(rootdir, filename):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file == filename:
                return os.path.join(subdir, file)

# Extract the actual frames from the video and compute the poses:
def get_frames(video_path, frames_per_step, segment, im_size, sess):
    # load video and acquire its parameters using opencv:
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    max_len = video.get(cv2.CAP_PROP_POS_MSEC)

    # check segment consistency:
    if max_len < segment[1]:
        segment[1] = max_len

    # define starting frame:
    centrale_frame = (np.linspace(segment[0], segment[1], num=3)) / 1000 * fps
    start_frame = centrale_frame[1] - frames_per_step / 2

    # for every frame in the clip extract frame, compute pose
    # and insert result in the matrix:
    frames = np.zeros(shape=(frames_per_step, im_size, im_size, 3), dtype=float)
    for z in range(frames_per_step):
        frame = start_frame + z
        video.set(1, frame)
        ret, im = video.read()
        pose_frame = PoseEstimation.compute_pose_frame(im, sess)
        res = cv2.resize(pose_frame, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC)
        frames[z,:,:,:] = res

    return frames

# Return two np matrices, one for the input and one for the labels
def read_clip_and_label(json_filename, batch_size, frames_per_step, im_size, sess):

    # TODO: manage the .json in a better way:
    #JSON_DS_FILENAME = "datasets/Dataset_PatternRecognition/json/dataset_training.json"
    #with open(JSON_DS_FILENAME) as json_file:
    with open(json_filename) as json_file:
        json_dataset = json.load(json_file)

    batch = np.zeros(shape=(batch_size, frames_per_step, im_size, im_size, 3), dtype=float)
    labels = np.zeros(shape=(batch_size), dtype=int)

    #for s in range(batch_size):
    s = 0
    while s < batch_size:
        entry_name = random.choice(list(json_dataset.keys()))
        training_entry = random.choice(json_dataset[entry_name])

        # Check that the label is contained in DS_CLASSES
        # (to manage missing items, ds is noisy here):
        if not training_entry["label"] in DS_CLASSES:
            continue

        # Search correct folder for the path:
        path = search_path("datasets/Dataset_PatternRecognition/H3.6M", entry_name)

        segment = training_entry["milliseconds"]

        clip = get_frames(path, frames_per_step, segment, im_size, sess)
        batch[s,:,:,:,:] = clip
        labels[s] = DS_CLASSES.index(training_entry["label"]) #label_to_id[training_entry["label"]]

        # Increase s at the end of the cycle:
        s += 1

    return batch, labels
