""" Module to read the dataset and other files

This module provides functions to read the datasets and group it into batches.
It also provides functions to load the ground truth file and other files
    
"""
import numpy as np
from skimage import io
from random import shuffle
from os import listdir
from os.path import isfile, join
from image_processing import preprocess, shear_image

def load_gt(file_ids, file_gt):
    """ Read the slants of a list of samples and store them in a dictionary
        Args:
            file_ids (string): Path to file with a list of samples
            file_gt (string): Path to the gt file with the slants of all samples
        Returns:
            dict: Dictionary that maps a sample id to its slant
    """
    # Collect all samples (i.e. their filenames)
    with open(file_ids, 'r') as input_file:
        ids = input_file.read().splitlines()

    # Collect slant values from ground truth
    data = {}
    with open(file_gt, 'r') as input_file:
        lines = input_file.readlines()
        for file_id in ids:
            for line in lines:
                parts = line.rstrip('\n').split()
                id_gt = parts[0]
                if file_id == id_gt:
                    data[file_id] = float(parts[1])
    return data

def sort_images(dir_images, sample_ids):
    """ Sort a list of image filenames by the width of the images. Extracting
        batches from consecutive files of this list reduces the amount of
        padding required which improves the results
        Args:
            dir_images (string): directory containing the image files
            sample_ids (list): list of image filenames
        Returns:
            list: list of image filenames sorted by image width
    """
    # Read images and collect their width (relative to their height)
    widths = {}
    for sample_id in sample_ids:
        img = io.imread(dir_images + sample_id)
        img_width = img.shape[0] / img.shape[1]
        widths[sample_id] = img_width

    # Sort image filenames by width
    sorted_widths = list(sorted(widths, key=lambda x: widths[x]))
    return sorted_widths

def load_data(dir_images, dataset=None, batch_size=1, image_height=32, width_stretch=2, sigma=2):
    """ Read the images of a dataset and group them into batches together 
        with the corresponding ground truth slant, the widths of the image
        and the filenames
        Args:
            dir_data (string): Directory that stores the images
            dataset (dict/list): Dictionary that maps sample ids to their true slant
                                 or list of filenames or None (if None, all files
                                 in dir_data will be loaded)
            batch_size (int): Desired batch size
            image_height (int): Height to which images get scaled
            width_stretch (float): Width gets strechted by this factor
            sigma (float): Sigma parameter of Gaussian blurring
        Returns:
            list: List of padded data-batches, true slants, image widths
                  and their filenames
    """
    # Get the filenames of the files to process (depending on type of 'dataset')
    ids = dataset
    mode = 'unlabeled'
    if isinstance(dataset, dict):
        ids = list(dataset.keys())
        mode = 'labeled'
    elif not isinstance(dataset, list):
        ids = [f for f in listdir(dir_images) if isfile(join(dir_images, f))]
    ids = sort_images(dir_images, ids)
        
    batched_data = []
    num_batches = int(np.ceil(len(ids) / batch_size))

    for n_batch in range(num_batches):
        batch_images = []
        batch_targets = []
        batch_img_width = []
        batch_sample_ids = []
            
        for sample_id in ids[n_batch * batch_size: min((n_batch+1) * batch_size, len(ids))]:
            # Read and preprocess image (binarize, blur and resize)
            img = io.imread(dir_images + sample_id)
            img = preprocess(img, new_height=image_height,
                             width_stretch=width_stretch, sigma=sigma)

            # Add dimensions for batch size and number of channels
            img_width = img.shape[1]
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=3)

            # Add image, width and filename to this batch's collections
            batch_images.append(img)
            batch_img_width.append(img_width)
            batch_sample_ids.append(sample_id)

            # Get true slant and add it to this batch's collections
            if mode == 'labeled':
                target = dataset[sample_id]
                batch_targets.append(target)

        # Pad all samples in the batch to width of widest sample
        max_width = max(batch_img_width)
        padded_images = np.ones(shape=(len(batch_images), batch_images[0].shape[1], max_width, 1), dtype=np.float)  
        for i, feat in enumerate(batch_images):
            padded_images[i, :, :feat.shape[2], :] = feat
            
        # Add batch to collection of batches
        batched_data.append((padded_images, batch_targets, batch_img_width, batch_sample_ids))
    shuffle(batched_data)
    return batched_data

def load_single_sample(file_image, image_height=32, width_stretch=2, sigma=2):
    """ Read and preprocess the image to process and return it s.t. it can be
        fed to the neural network
        Args:
            file_image (string): Image file to process
            image_height (int): Height to which image gets scaled
            width_stretch (float): Width gets strechted by this factor
            sigma (float): Sigma parameter of Gaussian blurring
        Returns:
            list: The image, its width and filename in a one-element list
    """        
    # Read and preprocess image (binarize, blur and resize)
    img = io.imread(file_image)
    img = preprocess(img, new_height=image_height, width_stretch=width_stretch,
                     sigma=sigma)

    # Add dimensions for batch size and number of channels
    img_width = img.shape[1]
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)

    # Add image, slant, width and filename to this batch's collections
    batch_sample_ids = [file_image]
    batch_img_width = [img_width]

    # Add batch to collection of batches
    batched_data = [(img, [], batch_img_width, batch_sample_ids)]
    return batched_data

def write_corrected_image(file_image, file_output, slant):
    img = io.imread(file_image)
    img_corrected = shear_image(img, slant)
    io.imsave(file_output, img_corrected)
    
