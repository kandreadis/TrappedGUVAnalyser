"""
This script: A collection of independent image analysis functions used by other scripts in this project.
For further explanation, read the descriptions at the beginning of each function.
Author: Konstantinos Andreadis @ Koenderink Lab 2023
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage import filters
import os


def threshold_img(img_raw, apply_blur, blur_size, use_inverse):
    """
    This function applies a threshold to (a blurred /& the inverse of an) image with Yen's algorithm.
    :param img_raw: Raw intensity values of 8-bit image.
    :type img_raw: Numpy array of shape NxM filled with integers.
    :param apply_blur: Option to include pre-processing step of blurring the image.
    :type apply_blur: Boolean
    :param blur_size: Kernel size for the Gaussian blur.
    :type blur_size: Uneven (!) integer, so e.g. 3
    :param use_inverse: Option to use inverse of image for thresholding.
    :type use_inverse: Boolean
    :return: Tresholded image | Found Yen's Treshold value.
    :rtype: Numpy array of same shape and type as input image | Float.
    """
    if apply_blur:
        # If blurring is chosen as pre-processing step, a Gaussian blur is applied with set kernel size.
        smooth_type_label = "Gaussian blurred"
        img_raw = cv2.GaussianBlur(img_raw, ksize=(blur_size, blur_size), sigmaX=0, sigmaY=0)
    else:
        smooth_type_label = "raw"
    # Applying Yen's algorithm to find the optimal min. threshold value.
    thresh_min_val = filters.threshold_yen(img_raw)
    thresh_max_val = 255
    if use_inverse:
        # In case of e.g. Bright-Field, the inverse of the image needs to be taken.
        print("- Thresholding {} inverse of image ...".format(smooth_type_label))
        _, img_thres = cv2.threshold(img_raw, thresh_min_val, thresh_max_val, cv2.THRESH_BINARY_INV)
    else:
        # For an improved GUV detection, safety margin of 60% is applied to the Yen estimate.
        print("- Thresholding {} image ...".format(smooth_type_label))
        thresh_min_val *= 0.4
        _, img_thres = cv2.threshold(img_raw, thresh_min_val, thresh_max_val, cv2.THRESH_BINARY)

    return img_thres, thresh_min_val


def filter_contours_by_area(contours, area_min_val, area_max_val):
    """
    This function filters contours by their enclosing rectangle's area, to exclude them from further analysis.
    :param contours: List of contours
    :type contours: native cv2 format for list of contours
    :param area_min_val: Min area in px^2
    :type area_min_val: Float
    :param area_max_val: Max area in px^2
    :type area_max_val: Float
    :return: List of contours indeces to be selected | Their respective areas
    :rtype: List of int | List of floats in px^2
    """
    valid_indeces = []
    contour_areas = []
    for i in range(len(contours)):
        # Fit rectangle to enclose contour and retrieve its global area.
        x, y, w, h = cv2.boundingRect(contours[i])
        area = w * h
        if area_min_val <= area <= area_max_val:
            valid_indeces.append(i)
            contour_areas.append(area)
    contour_areas = np.asarray(contour_areas)
    return valid_indeces, contour_areas


def contour_img(img_thresh, channel_index, channel_count):
    """
    This function detects contours in a thresholded image and filters them.
    :param img_thresh: Thresholded 8-bit image
    :type img_thresh: Numpy array of shape MxN filled with integers.
    :param channel_index: Selected channel
    :type channel_index: int
    :param channel_count: Total count of channels in image
    :type channel_count: int
    :return: list of contours | list of valid contours' indeces
    :rtype: cv2 native format of list of contours | list of int
    """
    canvas_width = img_thresh.shape[0]
    canvas_height = img_thresh.shape[1]
    canvas_area = canvas_width * canvas_height
    # RETR_EXTERNAL and CHAIN_APPROX_NONE refer to the detection algorithm, which should retrieve the external "outer"
    # contour and include all boundary points, respectively.
    if channel_index == 0:
        # Detection of GUV membrane
        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        valid_indeces, valid_areas = filter_contours_by_area(contours, area_min_val=canvas_area * 0.001,
                                                             area_max_val=canvas_area * 0.5)
        if len(valid_indeces) >= 1:
            # Take only largest valid contour
            valid_indeces = [valid_indeces[np.argsort(np.asarray(valid_areas))[-1]]]
    elif channel_index == channel_count - 1:
        # Detection of Bright-Field trap contours
        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        valid_indeces, valid_areas = filter_contours_by_area(contours, area_min_val=canvas_area * 0.05,
                                                             area_max_val=canvas_area)
        largest_area_indeces = np.argsort(np.asarray(valid_areas))[-2:]
        valid_indeces = [valid_indeces[i] for i in largest_area_indeces]
    else:
        # Detection of Protein Contours is not necessary, as they will be "cropped" using the membrane.
        contours, valid_indeces = None, [0]

    print("- Detected {} valid contour(s)".format(len(valid_indeces)))
    return contours, valid_indeces


def extract_valid_contours(contours, for_protein_mask):
    """
    Minor function to crop list of contours by their validity
    :param contours: List of contours
    :type contours: list of dicts containing the cv2 contour format
    :param for_protein_mask: In case of a protein channel, this option surpassess the validity check
    :type for_protein_mask: Boolean
    :return: List of valid contours
    :rtype: list of dicts containing the cv2 contour format
    """
    valid_contours = []
    if not for_protein_mask:
        for (i, contour) in enumerate(contours):
            if contours[i]["valid"]:
                valid_contours.append(contour)
    else:
        valid_contours.append(contours)
    return valid_contours


def characterise_contour(contour, img_raw, fill_inside):
    """
    This function characterises a given (filled) contour.
    :param contour: Contour
    :type contour: cv2 contour format
    :param img_raw: Raw 8-bit image
    :type img_raw: Raw 8-bit image
    :param fill_inside: Option to fill inside of contour for e.g. GUV contours
    :type fill_inside: Boolean
    :return: xy global position | widht, height | xy coordinates and their intensities | aspect ratio, area
    :rtype: float | float, float | array of float | array of int | float, float
    """
    x, y, w, h = cv2.boundingRect(contour)
    aspect = round(w / h, 2)
    area = cv2.contourArea(contour)
    if fill_inside:
        # Fill inside of contour
        mask = np.zeros_like(img_raw)
        cv2.drawContours(mask, contours=[contour], contourIdx=0, color=255, thickness=cv2.FILLED)
        mask = mask.T
        coords = np.column_stack(np.where(mask > 0))
        xdata = coords[:, 0]
        ydata = coords[:, 1]
    else:
        xdata = contour[:, 0, 0]
        ydata = contour[:, 0, 1]
    # Retrieve raw intensity values for xy coordinates of contour
    intensities = img_raw[ydata, xdata]
    return x, y, w, h, xdata, ydata, intensities, aspect, area


def calc_total_intensity(contours_xyi, label):
    """
    Minor function that calculates the total intensity of a contour.
    :param contours_xyi: List of contours in reduced form (only xy coordinates and their intensities)
    :type contours_xyi: list of array of columns: float, float, int
    :param label: For documentation purposes, specify the channel label, e.g. GUV membrane.
    :type label: String
    :return: Total Intensity
    :rtype: arbitrary units
    """
    total_intensity_val = []
    for contour_xy in contours_xyi:
        total_intensity_val.append(round(np.nansum(contour_xy[:, 2]), 2))
    print("- {}: Σ (Intensity) = {}".format(label, total_intensity_val))
    return total_intensity_val


def contour_curve(image):
    """
    This function extracts the exact contour of the Membrane channel.
    :param image: Raw 8-bit image
    :type image: Raw 8-bit image
    :return: Membrane Contour Curve
    :rtype: cv2 contour format
    """
    image = cv2.blur(image, ksize=(3, 3))
    thresh_low = filters.threshold_yen(image) * 0.45
    _, img_threshold = cv2.threshold(image, thresh_low, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_idx = -1
    curve = contours[contour_idx]
    return curve


def calc_perimeter_area(contour_xyi, label, xscale, yscale, unit):
    """
    This function calculates the perimeter and area of a contour.
    :param contour_xyi: Contour in reduced form (only xy coordinates and their intensities)
    :type contour_xyi: array of columns: float, float, int
    :param label: For documentation purposes, specify the channel label, e.g. GUV membrane.
    :type label: String
    :param xscale: X scaling factor converting px to real-world unit
    :type xscale: Float
    :param yscale: Y scaling factor converting px to real-world unit
    :type yscale: Float
    :param unit: Unit
    :type unit: String
    :return: perimeter | area | membrane boundary coordinates
    :rtype: Float | float | array of two columns with floats
    """
    x_values = contour_xyi[:, 0].astype(int)
    y_values = contour_xyi[:, 1].astype(int)
    intensity_values = contour_xyi[:, 2]
    width = np.max(x_values) - np.min(x_values)
    height = np.max(y_values) - np.min(y_values)
    image = np.zeros(shape=(height * 2, width * 2), dtype=np.uint8)
    for i, x in enumerate(x_values):
        try:
            image[y_values[i] + height // 2, x + width // 2] = intensity_values[i]
        except:
            pass

    curve = contour_curve(image=image)
    perimeter = cv2.arcLength(curve, closed=True) * np.average([xscale, yscale])
    area = cv2.contourArea(curve) * xscale * yscale
    print("- {}: Perimeter = {} {}, Area = {} {}^2".format(label, round(perimeter, 2), unit, round(area, 2), unit))
    curve_xy = curve[:, 0, :]
    curve_xy[:, 0] -= width // 2
    curve_xy[:, 1] -= height // 2
    return perimeter, area, curve_xy


def moving_average(array, w, mode="valid"):
    """
    Minor helper function that applies a moving average to an array
    :param mode: Boundary effect mode
    :type mode: String
    :param array: Array
    :type array: Array
    :param w: Window Size
    :type w: Int
    :return: Moving-average of array
    :rtype: array
    """
    return np.convolve(array, np.ones(w), mode=mode) / w


def calc_curvature(contour_xy, xscale, yscale):
    """
    This function calculates the curvature along a contour.
    :param contour_xy: Contour in reduced form (only xy coordinates)
    :type contour_xy: array of columns: float, float
    :param xscale: X scaling factor converting px to real-world unit
    :type xscale: Float
    :param yscale: Y scaling factor converting px to real-world unit
    :type yscale: Float
    :return: Top & Bottom profile, curvatures and xy coordinates
    :rtype: Arrays of two float-columns, float column, arrays of float columns
    """
    print("- Calculating curvatures...")
    x = contour_xy[:, 0] * xscale
    y = contour_xy[:, 1] * yscale
    x = moving_average(x, w=5)
    y = moving_average(y, w=5)

    def diff(array):
        """
        Minor helper function that differentiates an array.
        :param array: Array
        :type array: Array
        :return: Differentiated array
        :rtype: Array
        """
        return np.gradient(array)

    try:
        dx = diff(x)
        dy = diff(y)
        d2x = diff(dx)
        d2y = diff(dy)
        np.seterr(invalid='ignore')
        curvature = (d2x * dy - dx * d2y) / ((dx ** 2 + dy ** 2) ** (3 / 2))

        curvature = moving_average(curvature, w=10, mode="same")
    except:
        curvature = None
    profile_top = []
    profile_bottom = []
    for i in range(len(x)):
        if y[i] >= 0:
            profile_top.append([x[i], y[i]])
        else:
            profile_bottom.append([x[i], y[i]])
    profile_bottom = np.asarray(profile_bottom)
    profile_top = np.asarray(profile_top)
    return profile_top, profile_bottom, curvature, x / xscale, y / yscale


def intensity_curvature_corr(protein_intensities, boundary_curvatures):
    """
    This function correlates protein accumulation (intensity) to the nearest membrane curvature using a kdtree query.
    :param protein_intensities: Protein Intensities
    :type protein_intensities: List of Nx3 arrays (XYI)
    :param boundary_curvatures: Membrane Contour Curvatures
    :type boundary_curvatures: List of Nx1 arrays
    :return: Protein Intensity | corresponding nearest membrane curvature
    :rtype: Nx1 floats | Nx1 floats
    """
    intensity = []
    nearest_curvature = []

    for idx, intensities in enumerate(protein_intensities[1:]):
        intensities = intensities[intensities[:, 2] > 0]  # remove background
        intensity.append(intensities)
        curvatures_xy = boundary_curvatures[:, :2]
        boundary_kdtree = cKDTree(curvatures_xy)
        signal_xy = intensities[:, :2]
        nearest_indices = boundary_kdtree.query(signal_xy, k=1)[1]
        nearest_curvature.append(boundary_curvatures[nearest_indices, 2])
    return intensity, nearest_curvature
