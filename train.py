""" Module to train the model for slant correction

This script trains the convolutional neural network for the specified
parameters. After each epoch, the validation error is computed and checkpoints
are saved after the specified number of epochs.

"""

import os
import math
import tensorflow as tf
from file_io import load_gt, load_data
from model import Model

# Specify files and directories to use
file_train_ids = './gt/train_debug.txt'
file_valid_ids = './gt/valid_debug.txt'
file_gt = './gt/lines_gt.txt'
dir_images = './dataset/'
dir_checkpoints = './checkpoints/'
if not os.path.isdir(dir_checkpoints):
    os.makedirs(dir_checkpoints)

# Specify parameters to use
num_epochs = 60
batch_size = 8
checkpoint_steps = 1
image_height = 32
learning_rate = 0.0005
dropout_rate = 0.3
width_stretch = 1.8

# Get the filenames and corresponding slants for the datasets
train_gt = load_gt(file_train_ids, file_gt)
valid_gt = load_gt(file_valid_ids, file_gt)

# Load training and validation images
batched_train_data = load_data(dir_images, train_gt, batch_size=batch_size,
                               image_height=image_height,
                               width_stretch=width_stretch)
batched_valid_data = load_data(dir_images, valid_gt, batch_size=batch_size,
                               image_height=image_height,
                               width_stretch=width_stretch)

# Create the model object
model = Model(learning_rate=learning_rate, dropout_rate=dropout_rate,
              image_height=image_height)

# Start training
train_costs = []
valid_costs = []
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for curr_epoch in range(num_epochs):
        # Process all training batches and compute training error
        train_cost = 0
        for batch in range(len(batched_train_data)):
            #print('Batch {} / {}'.format(batch, len(batched_train_data)))
            feed = {model.inputs: batched_train_data[batch][0],
                    model.targets: batched_train_data[batch][1],
                    model.img_widths: batched_train_data[batch][2],
                    model.training: True}
            prediction, batch_cost, _ = session.run([model.pred, model.cost, model.optimizer], feed)
            train_cost += batch_cost                
        train_cost /= len(batched_train_data)
        train_costs.append(train_cost)

        # Process all validation batches and compute validation error
        valid_cost = 0
        for batch in range(len(batched_valid_data)):
            val_feed = {model.inputs: batched_valid_data[batch][0],
                        model.targets: batched_valid_data[batch][1],
                        model.img_widths: batched_valid_data[batch][2],
                        model.training: False}
            prediction, val_cost = session.run([model.pred, model.cost], feed_dict=val_feed)
            valid_cost += val_cost                
        valid_cost /= len(batched_valid_data)
        valid_costs.append(valid_cost)

        # Print training and validation errors after last training epoch
        log = 'Epoch {}/{}, train_cost = {:.3f}, valid_cost = {:.3f}'
        print(log.format(curr_epoch, num_epochs, train_cost, valid_cost))

        # Save current model and print predictions of first validation batch
        if (curr_epoch%checkpoint_steps == 0) or (curr_epoch == n_epochs-1):
            model.saver.save(session, dir_checkpoints + 'epoch' + str(curr_epoch) + '/model.ckpt')
            val_feed = {model.inputs: batched_valid_data[0][0],
                        model.targets: batched_valid_data[0][1],
                        model.img_widths: batched_valid_data[0][2],
                        model.training: False}
            prediction = session.run(model.pred, feed_dict=val_feed)
            print([float('%.1f' % elem) for elem in prediction])
            print(batched_valid_data[0][1])
            print()
    
