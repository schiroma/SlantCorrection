""" Module containing the definition of the neural network

This module contains the definition of the neural network, its cost function,
optimization method etc.

"""
import tensorflow as tf

        
def conv2d(x, W, b, stride=1, padding='SAME'):
    """ Convolutional layer and relu activation
        Args:
            x (4dTensor): Input tensor (batch_size x h x w x channels)
            W (4dTensor): Weights (k_h x k_w x channels x out_channels)
            b (1dTensor): Bias (out_channels)
            stride (int): Stride in vertical and horizontal direction
            padding (string): Type of padding
        Returns:
            4dTensor: Output tensor after applying convolution
    """
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k_h=2, k_w=2, stride_h=2, stride_w=2, padding='SAME'):
    """ Max pooling layer
        Args:
            x (4dTensor): Input tensor (batch_size x h x w x channels)
            k_h (int): Height of kernel
            k_w (int): Width of kernel
            stride_h (int): Stride in vertical direction
            stride_w (int): Stride in horizontal direction
            padding (string): Type of padding
        Returns:
            4dTensor: Output tensor after applying max pooling
    """
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding)

def fully_connected(x, W, b, relu=True):
    """ Fully connected layer
        Args:
            x (4dTensor): Input tensor (batch_size x 512)
            W (2dTensor): Weights (512 x 1)
            b (1dTensor): Bias (1)
            relu (bool): Apply relu activation if True
        Returns:
            float: Output of layer (scalar since we do regression)
    """
    x = tf.add(tf.matmul(x, W), b)
    if relu:
        x = tf.nn.relu(x)
    return x

def gloabal_avg_pool(x, image_widths, variable_seq_lengths=True):
    """ Apply global average pooling i.e. compute the mean along the width axis
        batch_size x new_seq_length x 512 -> batch_size x 512
        Args:
            x (3dTensor): cnn output tensor (batch_size x new_seq_length x 512)
            image_widths (1dTensor): Actual width of the images in the batch
                                     after the cnn (without the padded part)
            variable_seq_lengths (bool): Apply pooling on entire tensor x if
                                         false, ignore padded part if true                                    
        Returns:
            2dTensor: tensor of size batch_size x 512
    """
    # If simple gap (without ignoring the padded pixels in the images)
    if not variable_seq_lengths:
        return tf.reduce_mean(x, axis=1)

    # If padded pixels in the images should be ignored (experimental)
    # Only compute average over the values corresponding to the original image
    first_sample = tf.slice(x, [0, 0, 0],
                            [1, img_widths[0], x.get_shape().as_list()[2]])
    first_sample = tf.reduce_mean(first_sample, axis=1)

    # Loop over all samples in the batch to build the gap tensor of the batch
    i = tf.constant(1)
    while_condition = lambda i, g: tf.less(i, tf.shape(pool8)[0])
    def loop_body(i, partial_result):
        sample = tf.slice(x, [i, 0, 0], [1, img_widths[i], x.get_shape().as_list()[2]])
        sample = tf.reduce_mean(sample, axis=1)
        next_partial_result = tf.concat([partial_result, sample], axis=0)
        return [tf.add(i, 1), next_partial_result]
    _, gap = tf.while_loop(while_condition, loop_body, [i, first_sample])
    return gap

class Model:
    """ The Model class builds and stores the network architecture, weights,
        and the methods to predict, optimize etc
        Attributes:
            weights (dict): dict containing the network weights
            biases (dict): dict containing the network biases   
            inputs (tf.placeholder): Placeholder for the images of the samples in the batch
            targets (tf.placeholder): Placeholder for the true slants of the samples in the batch
            img_widths (tf.placeholder): Placeholder for the width of each image in the batch 
            training (tf.placeholder): True when training, False when doing inference
            pred (1dTensor): Predictions for the samples in the batch
            cost (float): MSE between predictions and true values            
            optimizer (tf.Operation): Optimization method
            saver (tf.train.Saver):  Saver object to store a trained network to file
    """
    
    def __init__(self, learning_rate=0.0005, dropout_rate=0.5, image_height=32):
        """ Initialize the network and its methods to predict, optimize etc
            Args:
                learning_rate (float): Learning rate in optimization
                dropout_rate (float): Dropout rate after global average pooling
                image_height (int): Required height of the input images  
        """
        self.weights = {
            'wc1': tf.get_variable('W0', shape=(3,3,1,64), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc2': tf.get_variable('W1', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc4': tf.get_variable('W3', shape=(3,3,128,128), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc5': tf.get_variable('W4', shape=(3,3,128,256), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc6': tf.get_variable('W5', shape=(3,3,256,256), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc7': tf.get_variable('W6', shape=(3,3,256,512), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc8': tf.get_variable('W7', shape=(3,3,512,512), initializer=tf.contrib.layers.xavier_initializer()), 
            'fc1': tf.get_variable('W8', shape=(512,1), initializer=tf.contrib.layers.xavier_initializer()), 
        }
        self.biases = {
            'bc1': tf.get_variable('B0', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'bc4': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'bc5': tf.get_variable('B4', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
            'bc6': tf.get_variable('B5', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
            'bc7': tf.get_variable('B6', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
            'bc8': tf.get_variable('B7', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
            'bfc1': tf.get_variable('B8', shape=(1), initializer=tf.contrib.layers.xavier_initializer()),
        }

        # Placeholder for the images to be fed to the network
        self.inputs = tf.placeholder(tf.float32, [None, image_height, None, 1], name = 'inputs')

        # Placeholder for the true slants
        self.targets = tf.placeholder(tf.float32, [None], name='targets')

        # Actual image width for each sample in the batch (before padding)
        self.img_widths = tf.placeholder(tf.int32, [None], name='img_widths')

        # Flag to specify if we are training of doing inference
        self.training = tf.placeholder_with_default(False, shape=(), name='training')

        # Compute prediction and optimize cost 
        self.pred = self.conv_net(self.inputs, self.img_widths, self.weights,
                                  self.biases, dropout_rate, self.training)
        self.cost = tf.reduce_mean(tf.square(self.pred - self.targets))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
        # Saver object to store trained networks in a file
        self.saver = tf.train.Saver(max_to_keep=None)

    def compute_out_img_widths(self, img_widths):
        """ Compute the width of each image in the batch after it went through 
            the cnn. We need these widths to ignore the padded pixels in the
            computation.
            Note: This method needs to be adjusted if network architecture is
            changed
            Args:
                img_widths (1dTensor): The widths of the images in the batch
            Returns:
                1dTensor: The widths of the images in the batch after going
                          through the cnn
        """
        conv1_trim = tf.constant( 2 * (3 // 2), dtype=tf.int32)
        one = tf.constant(1, dtype=tf.int32, name='one')
        two = tf.constant(2, dtype=tf.int32, name='two')
        after_conv1 = tf.subtract(img_widths, conv1_trim)
        after_pool2 = tf.floor_div(after_conv1, two)
        after_pool4 = tf.subtract(after_pool2, one)
        after_pool6 = tf.subtract(after_pool4, one)
        after_pool8 = tf.identity(after_pool6)
        img_widths_after = tf.reshape(after_pool8, [-1])
        return img_widths_after

    def conv_net(self, x, img_widths, weights, biases, dropout_rate, training):
        """ Compute the network output for input x
            Args:
                x (4dTensor): Input tensor (batch_size x 32 x max_width_in_batch x 1)
                img_widths (1dTensor): The widths of the images in the batch
                weights (dict): dict containing the network weights
                biases (dict): dict containing the network biases
                dropout_rate (float): Dropout rate after global average pooling
                training (bool): True when training, False when doing inference
            Returns:
                1DTensor: Predicted slants for the batch (batch_size x 1)
        """  
        conv1 = conv2d(x, weights['wc1'], biases['bc1'], padding='VALID')
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], padding='SAME')
        pool2 = maxpool2d(conv2, k_h=2, k_w=2, stride_h=2, stride_w=2, padding='VALID')
        conv3 = conv2d(pool2, weights['wc3'], biases['bc3'], padding='SAME')
        conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], padding='SAME')
        pool4 = maxpool2d(conv4, k_h=2, k_w=2, stride_h=2, stride_w=1, padding='VALID')
        conv5 = conv2d(pool4, weights['wc5'], biases['bc5'], padding='SAME')
        conv6 = conv2d(conv5, weights['wc6'], biases['bc6'], padding='SAME')
        pool6 = maxpool2d(conv6, k_h=2, k_w=2, stride_h=2, stride_w=1, padding='VALID')
        conv7 = conv2d(pool6, weights['wc7'], biases['bc7'], padding='SAME')
        conv8 = conv2d(conv7, weights['wc8'], biases['bc8'], padding='SAME')
        pool8 = maxpool2d(conv8, k_h=3, k_w=1, stride_h=3, stride_w=1, padding='VALID')      
        pool8 = tf.squeeze(pool8, axis=1, name='features')                
        # dimensions of pool8: batch_size x new_max_seq_len x 512
        
        # Calculate resulting sequence lengths from original image widths
        #img_widths_after = self.compute_out_img_widths(img_widths)

        # global average pooling -> batch_size x 512
        #gap = gloabal_avg_pool(pool8, img_widths_after)
        gap = tf.reduce_mean(pool8, axis=1)
        gap_dropout = tf.layers.dropout(gap, dropout_rate, training=training)

        # regression -> batch_size x 1
        out = fully_connected(gap_dropout, weights['fc1'], biases['bfc1'], relu=False)
        
        return tf.squeeze(out)
