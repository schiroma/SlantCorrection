""" Module containing image processing methods

This module contains functions for blurring, binarizing, resizing and shearing
of images, as well as the image preprocessing pipeline

"""

import numpy as np
from skimage import transform
from skimage import filters
from skimage import img_as_float
from skimage.transform import resize
from scipy import ndimage
from skimage import util
import math

def resize_image(image, new_height, width_stretch=1): 
    """ Resize an image to a desired height and stretch its width
        Args:
            image (matrix): The image to resize
            new_height (int): Target height of the image in pixels
            width_stretch (float): Width gets strechted by this factor
        Returns:
            matrix: The resized image
    """
    new_width = max(10, math.ceil(width_stretch * image.shape[1] * new_height/image.shape[0]))
    image_resized = resize(image, (new_height, new_width))
    return image_resized

def binarize_image(image):
    """ Binarize image using Otsu's method
        Args:
            image (matrix): The image to binarize
        Returns:
            matrix: The binarized image
    """
    threshold = filters.threshold_otsu(image)
    image_binarized = image > threshold
    image_binarized = np.uint8(image_binarized)*255
    image_binarized = img_as_float(image_binarized)
    return image_binarized

def blur_image(image, sigma):
    """ Blur an image using Gaussian blurring
        Args:
            image (matrix): The image to blur
            sigma (float): Sigma parameter of Gaussian blurring
        Returns:
            matrix: The blurred image
    """
    image_blurred = ndimage.gaussian_filter(image, sigma)
    image_blurred = np.clip(image_blurred, 0.0, 1.0)
    return image_blurred

def pad_image_for_shearing(image, shear_angle):
    """ Shearing may cut off parts of the image text. We need to pad the image
        to avoid this.
        Args:
            image (matrix): The image to pad (inverted, i.e. black background)
            shear_angle (float): Angle of the shearing
        Returns:
            matrix: The padded image
    """
    # Specify number of rows and columns to add
    height = image.shape[0]
    width = image.shape[1]
    n_cols = height 
    n_rows = height 

    # Create the additional rows/columns and add them to the image
    additional_rows = np.full((n_rows, width), np.uint8(0))
    additional_cols = np.full((height + n_rows, n_cols), np.uint8(0))
    image = np.append(image, additional_rows, axis=0)
    if shear_angle > 0:
        image = np.append(image, additional_cols, axis=1)
    else:
        image = np.append(additional_cols, image, axis=1)
    return image

def crop_image_text(image):
    """ Crop the text from the image
        Args:
            image (matrix): The image to crop (inverted, i.e. black background)
        Returns:
            matrix: The cropped text image
    """
    # Sum over all columns to find first non-black pixel from left and right
    col_sums = image.sum(axis=0)
    left = np.nonzero(col_sums)[0][0]
    left = max(0, left - 5)
    right = np.nonzero(col_sums)[0][-1]
    right = min(image.shape[1] - 1, right + 5)
    
    # Sum over all rows to find first non-black pixel from top and bottom
    row_sums = image.sum(axis=1)
    top = np.nonzero(row_sums)[0][0]
    top = max(0, top - 5)
    bottom = np.nonzero(row_sums)[0][-1]
    bottom = min(image.shape[0] - 1, bottom + 5)

    # Crop to the found text borders
    return image[top:bottom, left:right]

def shear_image(image, slant):
    """ Shear an image to correct its slant
        Args:
            image (matrix): The image to shear
            slant (float): Slant of the image
        Returns:
            matrix: The sheared image
    """
    # Work with inverted colors, otherwise shearing leads to black parts in image
    inverted_img = util.invert(image)

    # Compute shearing angle and transformation matrix
    shear = math.radians(90-float(slant))
    affine_tf = transform.AffineTransform(shear=shear)

    # Shear the image and crop it
    inverted_img = pad_image_for_shearing(inverted_img, shear)
    sheared_img = transform.warp(inverted_img, inverse_map=affine_tf)
    sheared_img = crop_image_text(sheared_img)

    # Invert back to white background / black text and resize to original height
    sheared_img = util.invert(sheared_img)
    sheared_img = resize_image(sheared_img, new_height=image.shape[0])
    return sheared_img

def preprocess(image, new_height=32, width_stretch=2, sigma=2):
    """ Apply the preprocessing steps to an image to make it ready to be
        fed to the neural network
        Args:
            image (matrix): The image to blur
            new_height (int): Target height of the image in pixels
            width_stretch (float): Width gets strechted by this factor
            sigma (float): Sigma parameter of Gaussian blurring
        Returns:
            matrix: The preprocessed image
    """
    image_processed = binarize_image(image)
    image_processed = blur_image(image_processed, sigma=sigma)
    image_processed = resize_image(image_processed, new_height=new_height,
                                   width_stretch=width_stretch)
    return image_processed    
    
