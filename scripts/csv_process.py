"""
This script: A collection of csv processing functions used by other scripts in this project.
For further explanation, read the descriptions at the beginning of each function.
Author: Konstantinos Andreadis @ Koenderink Lab 2023
"""
import os
import sys

import numpy as np
import pandas as pd


# noinspection PyTypeChecker




def export_dict_as_csv(data, path):
    """
    This function exports data to a .csv file
    :param data: Data
    :type data: dict
    :param path: Saving directory
    :type path: String
    """
    print("- Saving data as .csv...")
    np.set_printoptions(threshold=np.inf)
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path, index=False)


def export_matrix_as_csv(data, path):
    """
    This helper function simply exports data to a file that is
    :param data: matrix
    :type data: numpy matrix
    :param path: Saving directory
    :type path: Strings
    """
    np.savetxt(fname=path, X=data)


def str_to_array(strobject):
    """
    Minor helper function to convert a string object to array
    :param strobject: Object to convert
    :type strobject: String
    :return: converted object
    :rtype: numpy array
    """
    return np.fromstring(strobject.strip('[]'), dtype=int, sep=' ')


def read_csv(channel, path, image_index):
    """
    This function reads a .csv file as dict.
    :param channel: Selected channel index
    :type channel: Integer
    :param path: Image directory
    :type path: String
    :param image_index: Image index
    :type image_index: Integer
    :return: Content of .csv file
    :rtype: list of dicts arranged by their headers
    """
    csv_path = os.path.normpath(
        "results/csv/detection/{}_image_{}_c{}_contours.csv".format(path[5:], image_index, channel + 1))
    try:
        content = pd.read_csv(csv_path, header=0).to_dict(orient="records")
    except:
        content = None
    return content


def import_contour_params(multiple_contours, path, img_idx, ch_idx):
    """
    This function interprets a contour dict from the .csv file and a separate .txt file containing the raw image.
    :param multiple_contours: List of contours
    :type multiple_contours: List of dict
    :param path: Image Directory
    :type path: String
    :param img_idx: Image Index
    :type img_idx: Int
    :param ch_idx: Channel Index
    :type ch_idx: Int
    :return: Reformatted list of contours
    :rtype: List of dict
    """
    txt_path = os.path.normpath("results/csv/raw_frame/{}_image_{}_c{}_raw.csv".format(path[5:], img_idx, ch_idx + 1))
    if multiple_contours is not None:
        for contour in multiple_contours:
            try:
                contour["xscale"] = float(contour["xscale"])
                contour["yscale"] = float(contour["yscale"])
                contour["unit"] = str(["unit"])
                contour["original"] = np.loadtxt(fname=txt_path)
                try:
                    contour["x"] = int(contour["x"])
                    contour["y"] = int(contour["y"])
                    contour["w"] = int(contour["w"])
                    contour["h"] = int(contour["h"])
                    xdata = str_to_array(contour["xdata"])
                    ydata = str_to_array(contour["ydata"])
                    contour["intensities"] = str_to_array(contour["intensities"])
                    contour["xdata"] = xdata
                    contour["ydata"] = ydata
                    contour["aspect"] = float(contour["aspect"])
                    contour["area"] = float(contour["area"])
                except:
                    pass
            except:
                multiple_contours = None
                pass

    return multiple_contours


def csv_to_contour(channel_index, path, image_index):
    """
    Minor helper function that combines two functions to convert a .csv file to a list of contours.
    :param channel_index: Channel index
    :type channel_index: Int
    :param path: Image path
    :type path: String
    :param image_index: Image index
    :type image_index: Int
    :return: contours
    :rtype: native contour format of this project
    """
    return import_contour_params(read_csv(channel=channel_index, path=path, image_index=image_index), path=path,
                                 img_idx=image_index, ch_idx=channel_index)
