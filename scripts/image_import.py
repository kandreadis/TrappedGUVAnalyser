"""
This script: A collection of independent image import functions used by other scripts in this project.
For further explanation, read the descriptions at the beginning of each function.
Author: Konstantinos Andreadis @ Koenderink Lab 2023
"""
import os
import sys

import numpy as np
import pandas as pd
import tifffile as tiff
from readlif.reader import LifFile
from tifffile import TiffFile


def import_image(path, image_index, unit):
    """
    This function imports a .lif/.tiff/.tif image
    :param path: Image Path
    :type path: String
    :param image_index: Image Index
    :type image_index: int
    :param unit: unit
    :type unit: String
    :return: image, unit, xscale, yscale, n_channels, file_type, image_index, n_images
    :rtype: raw 6-bit numpy matrix, string, float, float, int, str, int, int
    """
    xscale = None
    yscale = None
    n_channels = None
    n_images = None
    file_type = None
    image = None
    if path[-3:] in ["tif", "iff"]:
        file_type = "tiff"
        image = tiff.imread(path)
        n_channels = image.shape[0]
        n_images = 1
        image_index = 0
        try:
            with TiffFile(path) as tif:
                for page in tif.pages:
                    if 'ImageDescription' in page.tags:
                        x_scale = page.tags["XResolution"].value
                        xscale = x_scale[1] / x_scale[0]
                        y_scale = page.tags["YResolution"].value
                        yscale = y_scale[1] / y_scale[0]
                        break
        except:
            print("Tif(f) Problem: Could not detect xy scaling factors and/or unit! Using px as unit...")
            unit = "px"
            xscale = 1
            yscale = 1
    elif path[-3:] == "lif":
        file_type = "lif"
        image_batch = LifFile(path)
        n_images = image_batch.num_images
        if image_index < n_images:
            image = image_batch.get_image(image_index)
            n_channels = image.channels
            x_scale = image.scale[0]
            xscale = 1 / x_scale
            y_scale = image.scale[1]
            yscale = 1 / y_scale
        else:
            image = None

    else:
        print("ERROR: Could not detect file type...")
    if None in [xscale, yscale, unit]:
        print("Problem: Could not detect xy scaling factors and/or unit! Using px as unit...")
        unit = "px"
        xscale = 1
        yscale = 1

    return image, unit, xscale, yscale, n_channels, file_type, image_index, n_images


def get_single_frame(img, channel_index, file_type, z_index=0, t_index=0):
    """
    This function retrieves a single frame from an image.
    :param img: .lif/.tif(f) image
    :type img: 8-bit image stack
    :param channel_index: Channel Index
    :type channel_index: int
    :param file_type: Image file type
    :type file_type: str
    :param z_index: Z slice index
    :type z_index: int
    :param t_index: Time frame index
    :type t_index: int
    :return: None if failed | raw frame intensity matrix
    :rtype: None | raw 8-bit numpy matrix
    """
    if file_type == "lif":
        return np.asarray(img.get_frame(z=z_index, t=t_index, c=channel_index, m=0))
    elif file_type == "tiff":
        return np.asarray(img[channel_index])
    else:
        print("ERROR: Could not extract image from file...")
        return None


def get_channel_stack(img, file_type, n_channels, path):
    """
    This function combines frames from different channels to one hyperstack. !!
    This re-orders the channels based on the metadata to always start off with the membrane channel, followed by
    protein signals and then the Bright-Field trap contour in the end.
    :param img: .lif/.tif(f) image
    :type img: 8-bit image
    :param file_type: Image file type
    :type file_type: str
    :param n_channels: Channel count
    :type n_channels: int
    :param path: Image path
    :type path: str
    :return: image stack
    :rtype: numpy multi-dimensional matrix
    """
    if n_channels > 2:
        channel_definition = import_manual_metadata(path)[0]
    else:
        channel_definition = import_manual_metadata("standard_empty_guv")[0]
    image_stack = []
    for channel_iter in channel_definition:
        if channel_definition[channel_iter] is not None:
            image_stack.append(get_single_frame(img, channel_definition[channel_iter], file_type))
    return np.asarray(image_stack)


def import_manual_metadata(image_path):
    """
    This function imports the metadata from the input_metadata.xslx spreadsheet.
    :param image_path: Image path
    :type image_path: String
    :return: None if failed OR channel_order | channel_labels | channel_colors
    :rtype: None OR list of ints | dict | dict
    """
    try:
        all_metadata = pd.DataFrame(
            pd.read_excel(
                os.path.normpath(os.path.join(sys.path[1], "input_metadata.xlsx")), sheet_name="channels").to_dict())
    except:
        print("Could not find input_metadata.xlsx!")
        return None
    meta, meta_idx, channel_labels, channel_colors = None, None, None, None

    for idx, metadata in enumerate(all_metadata["Image Path"]):
        if metadata == image_path:
            meta = all_metadata.iloc[idx]
            meta_idx = idx
    if meta_idx is None:
        print("Could not find metadata of image {}! Please check the input_metadata.xlsx file!".format(image_path))
        return None
    else:
        channel_order = {}
        channel_labels = {}
        channel_colors = {}
        for channel_index, field in enumerate(meta.keys()):
            if field != "Image Path":
                ch_idx = channel_index - 1
                if field == "Bright-Field":
                    ch_idx = -1
                if meta[field] == "None":
                    channel_order[ch_idx] = None
                else:
                    channel_order[ch_idx] = meta[field]
                    channel_labels[ch_idx] = field
                    channel_colors[ch_idx] = all_metadata.iloc[0][field]
        return channel_order, channel_labels, channel_colors


def import_manual_tags(img_path, img_idx):
    """
    This function imports manual tags added to input_metadata.xlsx
    :param img_path: Image Path
    :type img_path: String
    :param img_idx: Image Index
    :type img_idx: Int
    :return: None if failed OR Found Tags
    :rtype: None OR Pandas dataframe
    """
    try:
        all_tags = pd.DataFrame(
            pd.read_excel(os.path.normpath(sys.path[1] + "/input_metadata.xlsx"), sheet_name="tags").to_dict())
    except:
        print("Could not find input_metadata.xlsx!")
        return None
    found_tags = None
    for i, path in enumerate(all_tags["Image Path"]):
        for j, index in enumerate(all_tags["Image Index"]):
            if path == img_path and index == img_idx:
                found_tags = all_tags.iloc[i]
    return found_tags
