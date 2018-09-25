# TODO: create the functions needed to use the OpenPose network

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.client import timeline


import sys
sys.path.append("third_party/tf-openpose")
from common import estimate_pose, draw_humans
from networks import get_network

model = "mobilenet"
input_width = 368
input_height = 368
stage_level = 6

input_node = tf.placeholder(tf.float32, shape=(1, input_height, input_width, 3), name="image")

first_time = True
net, _, last_layer = get_network(model, input_node, None)

def compute_pose_frame(input_image, sess):
    # Build the network to produce the poses:
    image = cv2.resize(input_image, dsize=(input_height, input_width), interpolation=cv2.INTER_CUBIC)
    global first_time
    if first_time:
        # load pretrained weights
        #print(first_time)
        s = "%dx%d" % (input_node.shape[2], input_node.shape[1])
        ckpts = "./third_party/tf-openpose/models/trained/mobilenet_" + s + "/model-release"
        vars_in_checkpoint = tf.train.list_variables(ckpts)
        var_rest = []
        for el in vars_in_checkpoint:
            var_rest.append(el[0])
        variables = tf.contrib.slim.get_variables_to_restore()
        var_list = [v for v in variables if v.name.split(":")[0] in var_rest]

        loader = tf.train.Saver(var_list=var_list)

        loader.restore(sess, ckpts)

        first_time = False

    # Compute and draw the pose:
    vec = sess.run(net.get_output(name="concat_stage7"), feed_dict={"image:0": [image]})
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    pafMat, heatMat = sess.run([net.get_output(name=last_layer.format(stage=stage_level, aux=1)),
                                net.get_output(name=last_layer.format(stage=stage_level, aux=2))],
                               feed_dict={"image:0": [image]},
                               options=run_options)
    heatMat, pafMat = heatMat[0], pafMat[0]
    humans = estimate_pose(heatMat, pafMat)
    pose_image = np.zeros(tuple(image.shape), dtype=np.uint8)
    pose_image = draw_humans(pose_image, humans)

    # Extract argmax along axis 2:
    heatMat = np.amax(heatMat, axis=2)
    pafMat  = np.amax(pafMat,  axis=2)

    # Resize input_image, heatMat, pafMat to match pose_image:
    heatMat     = cv2.resize(heatMat,     dsize=(input_height, input_width), interpolation=cv2.INTER_CUBIC)
    pafMat      = cv2.resize(pafMat,      dsize=(input_height, input_width), interpolation=cv2.INTER_CUBIC)

    # Make sure there is a third dimension:
    heatMat = np.expand_dims(heatMat, axis=2)
    pafMat  = np.expand_dims(pafMat,  axis=2)

    # Concatenate input_image, pose_image, heatMat and pafMat:
    pose_image = np.concatenate((image, pose_image, heatMat, pafMat), axis=2)

    return pose_image
