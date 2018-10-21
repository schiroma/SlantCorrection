""" Module to test the model for slant correction

This script tests a trained model on the test set. It prints the test error
(MSE) and writes a text file containing predicted and true slants per sample.

"""

import os
import tensorflow as tf
from file_io import load_gt, load_data
from model import Model

# Specify files and directories to use
file_test_ids = './gt/train_debug.txt'
file_gt = './gt/lines_gt.txt'
dir_images = './dataset/'
dir_model = './checkpoints/epoch44/'
image_height = 32
batch_size = 1

# Specify ouput file with predictions of test samples
file_predictions = './predictions/test_predictions.txt'
if not os.path.isdir('./predictions/'):
    os.makedirs('./predictions/')

# Get the filenames and corresponding slants for the dataset
test_gt = load_gt(file_test_ids, file_gt)
test_ids = test_gt.keys()

# Load test images
batched_test_data = load_data(dir_images, test_gt, batch_size=batch_size,
                              image_height=image_height)

# Create the model object
model = Model(image_height=image_height)

if not os.path.isdir(dir_model):
    print('Selected model for testing (' + dir_model + ') does not exist')
else:   
    file_model = dir_model + 'model.ckpt'
    num_test_samples = len(batched_test_data)         
    with tf.Session() as session:
        # Load the trained network from file      
        model.saver.restore(session, file_model)

        # Get predictions and error for the test samples
        predictions = []
        test_cost = 0
        for batch in range(num_test_samples):
            test_feed = {model.inputs: batched_test_data[batch][0],
                         model.targets: batched_test_data[batch][1],
                         model.img_widths: batched_test_data[batch][2],
                         model.training: False}
            batch_pred, batch_cost = session.run([model.pred, model.cost], feed_dict=test_feed)
            test_cost += batch_cost   
            predictions.append(batch_pred)

            # Print predictions and true slants
            num_predictions = len(batched_test_data[batch][3])
            for i in range(len(batched_test_data[batch][3])):
                sample_id = batched_test_data[batch][3][i]
                pred = batch_pred
                if num_predictions > 1:
                    pred = batch_pred[i]
                true_slant = test_gt[sample_id]
                print(str(sample_id) + ', ' + str(pred) + ', ' + str(true_slant) + '\n')             

        # Compute and print total error of test set
        test_cost /= num_test_samples
        print('test error: ' + str(test_cost))
    
        # Write error and predicted slants to text file for analysis
        with open(file_predictions, 'w') as output_file:
            output_file.write('Error (MSE): ' + str(test_cost) + '\n')
            for batch in range(num_test_samples):
                num_predictions = len(batched_test_data[batch][3])
                for i in range(len(batched_test_data[batch][3])):
                    sample_id = batched_test_data[batch][3][i]
                    pred = predictions[batch]
                    if num_predictions > 1:
                        pred = predictions[batch][i]
                    true_slant = test_gt[sample_id]
                    output_file.write(str(sample_id) + '\t' + str(pred) + '\t'
                                      + str(true_slant) + '\n')    
