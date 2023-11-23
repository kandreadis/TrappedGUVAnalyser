"""
This script: Detection of GUVs trapped in a (microfluidic) device using image analysis tools.
Do not run this script directly, it is called by main.py!
For further explanation, read the descriptions at the beginning of each function.
Author: Konstantinos Andreadis @ Koenderink Lab 2023
"""
import os

import numpy as np

from scripts.csv_process import export_dict_as_csv
from scripts.image_import import get_channel_stack, import_image
from scripts.image_process import threshold_img, contour_img, extract_valid_contours, characterise_contour
from scripts.visualisation import plot_composite_and_channels, plot_contours


class ContourDetection:
    """
    This class contains all detection modules. For more detail please consult the image analysis workflow steps.
    """

    def __init__(self, detection_params):
        self.images_path = detection_params["images_path"]
        self.img_idx = detection_params["img_idx"]
        self.channel_index = detection_params["channel_index"]
        self.smooth_gauss_kernel = 7
        imgs = import_image(detection_params["images_path"], detection_params["img_idx"], detection_params["unit"])
        self.image, self.unit, self.xscale, self.yscale, self.n_chan, self.img_type, self.img_idx, self.n_imgs = imgs
        self.channels = np.arange(0, self.n_chan, 1)
        self.image_stack = get_channel_stack(img=self.image, file_type=self.img_type, n_channels=self.n_chan,
                                             path=self.images_path)
        self.channel_labels = detection_params["channel_labels"]
        self.channel_colors = detection_params["channel_colors"]
        if self.channel_index == self.n_chan - 1:  # trap
            self.inverse = True
            self.smooth_img = False
        else:
            self.inverse = False
            self.smooth_img = True
        self.threshold_low = None
        short_path = self.images_path[5:]
        csv_detect_folder = "results/csv/detection/"
        csv_raw_frame_folder = "results/csv/raw_frame/"
        self.csv_path = os.path.normpath(
            csv_detect_folder + "{}_image_{}_c{}_contours.csv".format(short_path, self.img_idx, self.channel_index + 1))
        self.txt_path = os.path.normpath(
            csv_raw_frame_folder + "{}_image_{}_c{}_raw.csv".format(short_path, self.img_idx, self.channel_index + 1))

    def document_contour(self, contours, valid_indeces):
        """
        This function documents a contour according to custom protocol.
        :param contours: Contours
        :type contours: cv2 contour format
        :param valid_indeces: List of valid contour indeces
        :type valid_indeces: List of int
        :return: Contour Parameters in custom format
        :rtype: dict
        """
        img_orig = self.image_stack[self.channel_index]
        if contours is not None:
            contours_params = []
            for (i, contour) in enumerate(contours):
                if self.channel_index != self.n_chan - 1:
                    fill = True
                else:
                    fill = False
                contour_properties = characterise_contour(contour=contour, img_raw=img_orig, fill_inside=fill)
                x, y, w, h, xdata, ydata, intensities, aspect, area = contour_properties
                contour_params = {
                    "contour": contour,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "aspect": aspect,
                    "area": area,
                    "xdata": xdata,
                    "ydata": ydata,
                    "intensities": intensities,
                    "valid": 0,
                    "xscale": self.xscale,
                    "yscale": self.yscale,
                    "unit": self.unit
                }
                if i in valid_indeces:
                    contour_params["valid"] = 1

                contours_params.append(contour_params)
        else:
            contours_params = {
                "xscale": self.xscale,
                "yscale": self.yscale,
                "unit": self.unit,
                "empty": True
            }
        return contours_params

    def save_composite(self):
        """
        This function merely saves a composite and channel-specific overview for a given image.
        """
        plot_composite_and_channels(image_index=self.img_idx, channels=self.channels,
                                    channel_colors=self.channel_colors, channel_labels=self.channel_labels,
                                    image_stack=self.image_stack, path=self.images_path, xscale=self.xscale,
                                    yscale=self.yscale, unit=self.unit)

    def run(self):
        """
        This function is run when the class is called. All steps adhere to the detection protocol outlined in the
        Image Analysis Workflow.
        """
        ch_index = self.channel_index
        if self.channel_index == self.n_chan - 1:
            ch_index = -1
        status_message = "IMG #{}/{} ".format(self.img_idx, self.n_imgs - 1)
        status_message += "CH {}/{} ({}) ".format(self.channel_index + 1, self.n_chan, self.channel_labels[ch_index])
        status_message += "| {}".format(self.images_path[5:])
        print(status_message)
        plot_composite_and_channels(image_index=self.img_idx, channels=self.channels,
                                    channel_colors=self.channel_colors, channel_labels=self.channel_labels,
                                    image_stack=self.image_stack, path=self.images_path, xscale=self.xscale,
                                    yscale=self.yscale, unit=self.unit)
        image_channel_threshold, self.threshold_low = threshold_img(img_raw=self.image_stack[self.channel_index],
                                                                    apply_blur=self.smooth_img,
                                                                    blur_size=self.smooth_gauss_kernel,
                                                                    use_inverse=self.inverse)
        image_channel_contours, image_channel_contours_valid_idx = contour_img(img_thresh=image_channel_threshold,
                                                                               channel_index=self.channel_index,
                                                                               channel_count=self.n_chan)

        image_channel_contours = self.document_contour(image_channel_contours, image_channel_contours_valid_idx)
        if self.channel_index == 0 or self.channel_index == self.n_chan - 1:
            contours_valid_count = len(image_channel_contours_valid_idx)
            plot_contours(contour_params=image_channel_contours, image_index=self.img_idx,
                          image_stack=self.image_stack, channel_index=self.channel_index,
                          channel_colors=self.channel_colors, channels=self.channels, threshold=self.threshold_low,
                          thresh_img=image_channel_threshold, valid_count=contours_valid_count, path=self.images_path,
                          xscale=self.xscale, yscale=self.yscale, unit=self.unit)
            for_protein_mask = False
        else:
            contours_valid_count = 1
            for_protein_mask = True
        if contours_valid_count >= 10:
            print("- More than 10 valid contours -> Probably a mistake, not saving .csv!")
        else:
            valid_contours = extract_valid_contours(image_channel_contours, for_protein_mask=for_protein_mask)
            export_dict_as_csv(data=valid_contours, path=self.csv_path)
        np.savetxt(fname=self.txt_path, X=self.image_stack[self.channel_index])
