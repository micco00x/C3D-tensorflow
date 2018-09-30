# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import dataset_manager as input_data
import c3d_model
import math
import numpy as np

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
#flags.DEFINE_float('learning_rate', 0.0, 'Initial learning rate.')
#flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
flags.DEFINE_integer("epochs", 5, "Total number of epochs.")
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models'

#def placeholder_inputs(batch_size):
def placeholder_inputs():
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
	batch_size: The batch size will be baked into both placeholders.

    Returns:
        images_placeholder: Images placeholder.
	       labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           c3d_model.NUM_FRAMES_PER_CLIP,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.uint8, shape=(None, len(input_data.DS_CLASSES)))
    return images_placeholder, labels_placeholder

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads

def tower_loss(name_scope, logit, labels):
    #cross_entropy_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logit))
    cross_entropy_mean = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels,logits=logit, pos_weight=2.5))
    tf.summary.scalar(name_scope + '_cross_entropy', cross_entropy_mean)
    weight_decay_loss = tf.get_collection('weightdecay_losses')
    tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss) )

    # Calculate the total loss for the current tower.
    total_loss = cross_entropy_mean + weight_decay_loss
    tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss) )
    return total_loss

#def tower_acc(logit, labels):
#    #correct_pred = tf.equal(tf.argmax(logit, 1), labels)
#    correct_pred = tf.equal(tf.round(tf.sigmoid(logit)), labels)
#    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#    return accuracy

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

# Return a dataset (X, y) given a .npz filename and a .json filename:
def _generate_dataset(npz_filename, sess, videos_folder=None, json_filename=None):
    # Generate a dataset if the .npz file does not exist:
    if os.path.isfile(npz_filename):
        npzfile = np.load(npz_filename)
        X = npzfile["X"]
        y = npzfile["y"]
    else:
        X, y = input_data.generate_dataset(
            videos_folder=videos_folder,
            json_filename=json_filename,
            frames_per_step=c3d_model.NUM_FRAMES_PER_CLIP,
            im_size=c3d_model.CROP_SIZE,
            sess=sess,
            output_filename=npz_filename
        )
    return X, y

# Preprocess the data before feeding it to the neural network:
def _preprocess_data(X):
    # Rescale X from [0,255] to [0,1]:
    return X.astype(np.float32) / 255.0

def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    use_pretrained_model = True
    model_filename = "./sports1m_finetuning_ucf101.model"

    #with tf.Graph().as_default():
    with tf.variable_scope("C3D"):
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        #images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
        images_placeholder, labels_placeholder = placeholder_inputs()
        tower_grads1 = []
        tower_grads2 = []
        logits = []
        opt_stable = tf.train.AdamOptimizer(1e-4)
        opt_finetuning = tf.train.AdamOptimizer(1e-3)
        with tf.variable_scope('var_name') as var_scope:
            weights = {
                'wc1': _variable_with_weight_decay('wc1', [c3d_model.CHANNELS, c3d_model.CHANNELS, c3d_model.CHANNELS, c3d_model.CHANNELS, 64], 0.0005),
                'wc2': _variable_with_weight_decay('wc2', [c3d_model.CHANNELS, c3d_model.CHANNELS, c3d_model.CHANNELS, 64, 128], 0.0005),
                'wc3a': _variable_with_weight_decay('wc3a', [c3d_model.CHANNELS, c3d_model.CHANNELS, c3d_model.CHANNELS, 128, 256], 0.0005),
                'wc3b': _variable_with_weight_decay('wc3b', [c3d_model.CHANNELS, c3d_model.CHANNELS, c3d_model.CHANNELS, 256, 256], 0.0005),
                'wc4a': _variable_with_weight_decay('wc4a', [c3d_model.CHANNELS, c3d_model.CHANNELS, c3d_model.CHANNELS, 256, 512], 0.0005),
                'wc4b': _variable_with_weight_decay('wc4b', [c3d_model.CHANNELS, c3d_model.CHANNELS, c3d_model.CHANNELS, 512, 512], 0.0005),
                'wc5a': _variable_with_weight_decay('wc5a', [c3d_model.CHANNELS, c3d_model.CHANNELS, c3d_model.CHANNELS, 512, 512], 0.0005),
                'wc5b': _variable_with_weight_decay('wc5b', [c3d_model.CHANNELS, c3d_model.CHANNELS, c3d_model.CHANNELS, 512, 512], 0.0005),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                'out': _variable_with_weight_decay('wout', [4096, len(input_data.DS_CLASSES)], 0.0005)
            }
            biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
                'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                'out': _variable_with_weight_decay('bout', [len(input_data.DS_CLASSES)], 0.000),
            }
        for gpu_index in range(0, gpu_num):
            with tf.device('/gpu:%d' % gpu_index):

                varlist2 = [ weights['out'],biases['out'] ]
                varlist1 = list( set(weights.values()) | set(biases.values()) - set(varlist2) )
                logit = c3d_model.inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:],
                                                0.5,
                                                #FLAGS.batch_size,
                                                weights,
                                                biases)
                loss_name_scope = ('gpud_%d_loss' % gpu_index)
                loss = tower_loss(loss_name_scope, logit,
                                  tf.cast(labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size], tf.float32))
                loss_rm = tf.reduce_mean(loss)
                grads1 = opt_stable.compute_gradients(loss, varlist1)
                grads2 = opt_finetuning.compute_gradients(loss, varlist2)
                tower_grads1.append(grads1)
                tower_grads2.append(grads2)
                logits.append(logit)
        logits = tf.concat(logits,0)
        #accuracy = tower_acc(logits, tf.cast(labels_placeholder, tf.float32))
        with tf.variable_scope("metrics"):
            sigm_logits = tf.sigmoid(logits)
            predictions = tf.round((tf.sign(sigm_logits - tf.reduce_mean(sigm_logits, axis=1, keepdims=True) * 1.2) + 1) / 2)
            accuracy, accuracy_update_op = tf.metrics.accuracy(labels_placeholder, predictions)
            precision, precision_update_op = tf.metrics.precision(labels_placeholder, predictions)
            recall, recall_update_op = tf.metrics.recall(labels_placeholder, predictions)
            #accuracy, accuracy_update_op = tf.metrics.accuracy(labels_placeholder, tf.round(tf.sigmoid(logits)))
            #precision, precision_update_op = tf.metrics.precision(labels_placeholder, tf.round(tf.sigmoid(logits)))
            #recall, recall_update_op = tf.metrics.recall(labels_placeholder, tf.round(tf.sigmoid(logits)))
            f1score = 2 * precision * recall / (precision + recall)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("precision", precision)
        tf.summary.scalar("recall", recall)
        tf.summary.scalar("f1score", f1score)

        grads1 = average_gradients(tower_grads1)
        grads2 = average_gradients(tower_grads2)
        apply_gradient_op1 = opt_stable.apply_gradients(grads1)
        apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
        null_op = tf.no_op()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(list(weights.values()) + list(biases.values()))
    init = tf.global_variables_initializer()
    metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="C3D/metrics")
    metrics_vars_init = tf.variables_initializer(var_list=metrics_vars)
    #init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Create a session for running Ops on the Graph.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    if os.path.isfile(model_filename) and use_pretrained_model:
        saver.restore(sess, model_filename)

    # Create summary writter
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./visual_logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/test', sess.graph)

    # TODO: pass these as args:
    videos_folder = "datasets/Dataset_PatternRecognition/H3.6M"
    train_json_filename = "datasets/Dataset_PatternRecognition/json/dataset_training.json"
    train_npz_filename = "datasets/Dataset_PatternRecognition/npz/dataset_training.npz"
    val_json_filename = "datasets/Dataset_PatternRecognition/json/dataset_testing.json"
    val_npz_filename = "datasets/Dataset_PatternRecognition/npz/dataset_testing.npz"

    # Generate datasets:
    train_X, train_y = _generate_dataset(train_npz_filename, sess, videos_folder, train_json_filename)
    val_X, val_y = _generate_dataset(val_npz_filename, sess, videos_folder, val_json_filename)

    # Train the network and compute metrics on train a val sets:
    batch_size = FLAGS.batch_size * gpu_num
    for epoch in range(FLAGS.epochs):
        print("Epoch {}/{}:".format(epoch+1, FLAGS.epochs))

        # Reset metrics:
        sess.run(metrics_vars_init)

        # Iterate through training set:
        rand_indices = np.random.randint(train_X.shape[0], size=train_X.shape[0])
        for idx in range(0, train_X.shape[0], batch_size):
            # Extract the following batch_size indices:
            L = min(idx+batch_size, train_X.shape[0])
            train_images = _preprocess_data(train_X[rand_indices[idx:L]])
            train_labels = train_y[rand_indices[idx:L]]

            # Update metrics and get results:
            sess.run([train_op, accuracy_update_op, precision_update_op, recall_update_op],
                feed_dict={images_placeholder: train_images, labels_placeholder: train_labels})
            summary, train_curr_loss, train_curr_accuracy, train_curr_precision, train_curr_recall, train_curr_f1score = \
                sess.run([merged, loss_rm, accuracy, precision, recall, f1score],
                    feed_dict={images_placeholder: train_images, labels_placeholder: train_labels})

            # Print results:
            print("Progress: {}/{} - train_loss: {:2.3} - train_accuracy: {:2.3} - "
                "train_precision: {:2.3} - train_recall: {:2.3} - train_f1score: {:2.3}"
                .format(L, train_X.shape[0], train_curr_loss, train_curr_accuracy, train_curr_precision,
                    train_curr_recall, train_curr_f1score), end="\r")
        print("")

        # Save metrics to TensorBoard:
        train_writer.add_summary(summary, epoch+1)

        # Reset metrics:
        sess.run(metrics_vars_init)

        # Iterate through validation set:
        for idx in range(0, val_X.shape[0], batch_size):
            # Extract the following batch_size indices:
            L = min(idx+batch_size, val_X.shape[0])
            val_images = _preprocess_data(val_X[idx:L])
            val_labels = val_y[idx:L]

            # Update metrics and get results:
            sess.run([accuracy_update_op, precision_update_op, recall_update_op],
                feed_dict={images_placeholder: val_images, labels_placeholder: val_labels})
            summary, val_curr_loss, val_curr_accuracy, val_curr_precision, val_curr_recall, val_curr_f1score = \
                sess.run([merged, loss_rm, accuracy, precision, recall, f1score],
                    feed_dict={images_placeholder: val_images, labels_placeholder: val_labels})

            # Print results:
            print("Progress: {}/{} - val_loss: {:2.3} - val_accuracy: {:2.3} - "
                "val_precision: {:2.3} - val_recall: {:2.3} - val_f1score: {:2.3}"
                .format(L, val_X.shape[0], val_curr_loss, val_curr_accuracy, val_curr_precision,
                    val_curr_recall, val_curr_f1score), end="\r")
        print("")

        # Save metrics to TensorBoard:
        test_writer.add_summary(summary, epoch+1)

        # Save checkpoint:
        saver.save(sess, os.path.join(model_save_dir, 'c3d_ucf_model'), global_step=epoch+1)
    print("done")

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
