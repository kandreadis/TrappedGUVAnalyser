"""
This script: Analysis of trapped GUVs using detection results from detection.py-saved .csv files.
Do not run this script directly, it is called by main.py!
For further explanation, read the descriptions at the beginning of each function.
Author: Konstantinos Andreadis @ Koenderink Lab 2023
"""
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scripts.csv_process import csv_to_contour, export_dict_as_csv, export_matrix_as_csv
from scripts.detection import ContourDetection
from scripts.image_import import import_image
from scripts.image_process import calc_perimeter_area, calc_curvature, \
    intensity_curvature_corr
from scripts.params import detection_params
from scripts.visualisation import calc_line_xy, plot_reference_projection, plot_masked, plot_membrane_geometry, \
    plot_protein_accumulation, plot_curvature_correlation


class ContourAnalysis:
    """
    This class contains all analysis modules. For more detail please consult the image analysis workflow steps.
    """

    def __init__(self, detect_params, batch_indeces):
        self.detection_params = detect_params
        self.image_path = detect_params["images_path"]
        img_meta = import_image(self.image_path, detect_params["img_idx"], detect_params["unit"])
        _, self.unit, self.xscale, self.yscale, self.n_chan, self.file_type, self.image_index, self.n_imgs = img_meta
        if self.file_type == "lif":
            if batch_indeces == "all_images":
                print("! all {} images will be analysed, this might take a while !".format(self.n_imgs))
                self.batch_indeces = range(self.n_imgs)
            else:
                self.batch_indeces = batch_indeces
        else:
            self.batch_indeces = None
        self.detection_params["unit"] = self.unit
        self.detection_params["img_idx"] = self.image_index

    @staticmethod
    def extract_xy_pos_intens(contour):
        x_pts = contour["xdata"]
        y_pts = contour["ydata"]
        intensities = contour["intensities"]
        xy_pos = np.column_stack((x_pts, y_pts, intensities))
        return xy_pos

    def extract_trap_reference(self, trap_contours):
        trap_features_xy = []
        for trap_feature in trap_contours:
            trap_feature_xy = self.extract_xy_pos_intens(trap_feature)[:, :2]
            trap_features_xy.append(trap_feature_xy)
        xy_trap = np.concatenate(trap_features_xy)
        pca = PCA(n_components=2)
        pca.fit(xy_trap)
        line_start = np.mean(xy_trap, axis=0)
        line_direction = pca.components_[0]
        line_direction_orth = pca.components_[1]

        line_xy = calc_line_xy(400, line_start, line_direction)
        line_xy_orth = calc_line_xy(400, line_start, line_direction_orth)
        return line_direction, line_direction_orth, line_xy, line_xy_orth

    def contour_axis_projection(self, contour, axis, already_extracted_xy=False, already_extracted_xyi=False):
        if already_extracted_xy:
            contour_ = contour
            point_intensities = None
        elif already_extracted_xyi:
            contour_ = contour
            point_intensities = contour_[:, 2]
        else:
            contour_ = self.extract_xy_pos_intens(contour)
            point_intensities = contour_[:, 2]
        contour_coords = contour_[:, :2]
        axis_phi = np.arctan2(axis[0], axis[1])
        # print("- Found Trap Angle ca. {} degrees".format(round(np.degrees(axis_phi),4)))

        rotation_matrix = np.array([[np.cos(axis_phi), -np.sin(axis_phi)],
                                    [np.sin(axis_phi), np.cos(axis_phi)]])
        contour_mean = np.mean(contour_coords, axis=0)
        rotated_points = np.dot(contour_coords - contour_mean, rotation_matrix.T)
        rotated_points[:, 1] *= -1
        if already_extracted_xy:
            return rotated_points
        else:
            return np.column_stack([rotated_points, point_intensities])

    def project_all_contours(self, object_contours, axis):
        projected_objects = []
        for object_index, object_profile in enumerate(object_contours):
            object_projected_xy = self.contour_axis_projection(object_profile, axis)
            projected_objects.append(object_projected_xy)
        return projected_objects

    def save_composite(self):
        contourSearch = ContourDetection(self.detection_params)
        contourSearch.save_composite()

    def batch_contour_detection(self):
        print("# Detecting contours in images...")
        if self.batch_indeces is not None:
            for i in self.batch_indeces:
                self.detection_params["img_idx"] = i
                contourSearch = ContourDetection(self.detection_params)
                self.n_chan = contourSearch.n_chan
                for j in range(self.n_chan):
                    print("----------------------------")
                    self.detection_params["img_idx"] = i
                    self.detection_params["channel_index"] = j
                    contourSearch = ContourDetection(self.detection_params)
                    contourSearch.run()
        else:
            for j in range(self.n_chan):
                print("----------------------------")
                self.detection_params["channel_index"] = j
                contourSearch = ContourDetection(self.detection_params)
                contourSearch.run()

    @staticmethod
    def mask_protein_channel(membrane_mask_xy, protein):
        protein_masked_intensities = protein[membrane_mask_xy[:, 0], membrane_mask_xy[:, 1]]
        protein_masked = np.column_stack((membrane_mask_xy, protein_masked_intensities.T))
        return np.array(protein_masked)

    def mask_proteins(self, protein_contours, membrane_contour_xy):
        protein_masked = []
        for protein_iter in range(1, self.n_chan - 1):
            protein_raw = protein_contours[protein_iter]["original"].T
            protein_contour_masked = self.mask_protein_channel(membrane_mask_xy=membrane_contour_xy,
                                                               protein=protein_raw)
            protein_masked.append(protein_contour_masked)
        return protein_masked

    def membrane_GUV_mask(self, contours_guv):
        print("- Using membrane as mask for protein channel(s)...")
        membrane_contour = contours_guv[0]
        membrane_contour_xy = self.extract_xy_pos_intens(membrane_contour)
        proteins_masked = self.mask_proteins(contours_guv, membrane_contour_xy[:, :2])
        plot_masked(membrane_contour=membrane_contour_xy, masked_contours=proteins_masked,
                    channel_labels=self.detection_params["channel_labels"],
                    channel_colors=self.detection_params["channel_colors"], image_index=self.image_index,
                    path=self.image_path, xscale=self.xscale,
                    yscale=self.yscale, unit=self.unit)
        contours_GUV_masked = [membrane_contour]
        for protein_iter, protein_masked in enumerate(proteins_masked):
            contour_m = contours_guv[protein_iter + 1]
            contour_m["xdata"], contour_m["ydata"], contour_m["intensities"] = protein_masked[:, 0], \
                protein_masked[:, 1], protein_masked[:, 2]
            contours_GUV_masked.append(contour_m)
        return contours_GUV_masked, proteins_masked

    def GUV_trap_projection(self, contours_guv, contours_trap, trap_axis):
        print("- Detecting trap reference and projecting contours...")
        projected_contours_xy = self.project_all_contours(object_contours=contours_guv, axis=trap_axis[1])
        plot_reference_projection(contours=[contours_guv, contours_trap], image_index=self.image_index,
                                  trap_axis=trap_axis[2:], show_original_img=False, path=self.image_path,
                                  projected_contours=projected_contours_xy,
                                  channel_colors=self.detection_params["channel_colors"],
                                  channel_labels=self.detection_params["channel_labels"], xscale=self.xscale,
                                  yscale=self.yscale, unit=self.unit)
        return projected_contours_xy

    def analyse_contour(self, image_index):
        self.image_index = image_index
        contours_GUV = []
        for ch_i in range(self.n_chan - 1):
            contours_GUV_ch = csv_to_contour(channel_index=ch_i, path=self.image_path, image_index=self.image_index)
            if contours_GUV_ch is not None:
                contours_GUV.append(contours_GUV_ch[0])
        contours_trap = csv_to_contour(channel_index=self.n_chan - 1, path=self.image_path,
                                       image_index=self.image_index)
        if contours_trap is None or contours_GUV == [None] or contours_GUV == []:
            print("Image #", self.image_index, ": No contours were found for {} #{}, "
                                               "analysis is not possible!".format(self.image_path[5:],
                                                                                  self.image_index))
            return None

        print("Image #", self.image_index,
              ": {} Contour(s) were found for {} #{}, starting analysis...".format(
                  len(contours_GUV) + len(contours_trap), self.image_path[5:], self.image_index))
        if "xdata" not in contours_GUV[0].keys():
            print("Order of channels might be incorrect, aborting....")
            return None
        if self.n_chan > 2 and len(contours_GUV) > 1:
            contours_GUV, protein_masked = self.membrane_GUV_mask(contours_guv=contours_GUV)
        else:
            protein_masked = None
        trap_axis = self.extract_trap_reference(contours_trap)

        projected_contours_xy = self.GUV_trap_projection(contours_guv=contours_GUV, trap_axis=trap_axis,
                                                         contours_trap=contours_trap)
        membrane_xyI = projected_contours_xy[0]
        membrane_geometry = calc_perimeter_area(contour_xyi=membrane_xyI, label="Membrane", xscale=self.xscale,
                                                yscale=self.yscale, unit=self.unit)
        membrane_perimeter, membrane_area, membrane_contour = membrane_geometry
        if protein_masked is not None:
            protein_masked = projected_contours_xy
        boundary_curvatures = None
        protein_intensity = None
        nearest_curvature = None
        if len(membrane_xyI) > 2:  # exclude too small membrane contours
            profile_top, profile_bottom, curvatures, xcords, ycords = calc_curvature(contour_xy=membrane_contour,
                                                                                     xscale=self.xscale,
                                                                                     yscale=self.yscale)

            if len(profile_top) > 0 and len(profile_bottom) > 0:
                print("- Saving geometrical analysis...")
                plot_membrane_geometry(xcords=xcords, ycords=ycords, curvatures=curvatures, label="Membrane",
                                       unit=self.unit, perimeter=membrane_perimeter, area=membrane_area,
                                       path=self.image_path, image_index=self.image_index,
                                       channel_colors=detection_params["channel_colors"], xscale=self.xscale,
                                       yscale=self.yscale, projected_membrane=membrane_xyI)
                boundary_curvatures = np.column_stack((xcords, ycords, curvatures))
                if self.n_chan > 2 and protein_masked is not None:
                    protein_intensity, nearest_curvature = intensity_curvature_corr(
                        protein_intensities=protein_masked, boundary_curvatures=boundary_curvatures)
                    plot_protein_accumulation(signal=protein_intensity, boundary_curvature=boundary_curvatures,
                                              xscale=self.xscale, yscale=self.yscale, unit=self.unit,
                                              channel_colors=detection_params["channel_colors"],
                                              channel_labels=detection_params["channel_labels"],
                                              path=self.image_path, image_index=self.image_index)
                    try:
                        plot_curvature_correlation(signal=protein_intensity,
                                                   nearest_curvature=nearest_curvature, unit=self.unit,
                                                   channel_colors=detection_params["channel_colors"],
                                                   channel_labels=detection_params["channel_labels"],
                                                   path=self.image_path, image_index=self.image_index)
                    except:
                        print("error plotting the curvature correlation...")
                    short_path = self.image_path[5:]
                    c_times_I = {}
                    for idx, protein in enumerate(protein_intensity):
                        c_times_I[detection_params["channel_labels"][idx + 1]] = np.average(nearest_curvature[idx],
                                                                                            weights=protein[:, 2])
                    c_times_I = pd.DataFrame.from_dict(c_times_I, orient='index').T
                    export_dict_as_csv(data=c_times_I,
                                       path=os.path.normpath(
                                           "results/csv/analysis/{}_image_{}_cI.csv".format(short_path,
                                                                                            self.image_index)))
        dt_string = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        analysis_summary = {
            "image_path": [self.image_path],
            "image_index": self.image_index,
            "n_channels": self.n_chan,
            "unit": self.unit,
            "data_time_analysis": dt_string,
            "membrane_perimeter": membrane_perimeter,
            "membrane_area": membrane_area,
            "membrane_diameter_estimate": membrane_perimeter / np.pi,
            "channel_labels": ';'.join(detection_params["channel_labels"].values()),
            "channel_colors": ';'.join(detection_params["channel_colors"].values())
        }

        short_path = self.image_path[5:]
        export_dict_as_csv(data=analysis_summary,
                           path=os.path.normpath("results/csv/analysis/"
                                                 "{}_image_{}_summary.csv".format(short_path, self.image_index)))
        if boundary_curvatures is not None:
            export_matrix_as_csv(data=boundary_curvatures,
                                 path=os.path.normpath("results/csv/analysis/"
                                                       "{}_image_{}_membrane"
                                                       "_curvature.csv".format(short_path, self.image_index)))

        if protein_intensity is not None and nearest_curvature is not None:
            for i, protein_i in enumerate(protein_intensity):
                export_matrix_as_csv(data=protein_i,
                                     path=os.path.normpath("results/csv/analysis/{}_image_{}_{}"
                                                           "_protein_accumulation"
                                                           ".csv".format(short_path, self.image_index,
                                                                         detection_params["channel_labels"][
                                                                             i + 1])))
                export_matrix_as_csv(data=nearest_curvature[i],
                                     path=os.path.normpath("results/csv/analysis/"
                                                           "{}_image_{}_{}"
                                                           "_curvature_correlation"
                                                           ".csv".format(short_path, self.image_index,
                                                                         detection_params["channel_labels"][
                                                                             i + 1])))

    def batch_contour_analysis(self):
        print("# Analysing contours in images...")
        if self.file_type == "lif":
            if self.batch_indeces is not None:
                for batch_index in self.batch_indeces:
                    print("----------------------------")
                    self.detection_params["img_idx"] = batch_index
                    contour_analysis = ContourAnalysis(detect_params=self.detection_params,
                                                       batch_indeces=self.batch_indeces)
                    contour_analysis.analyse_contour(image_index=batch_index)
            else:
                print("----------------------------")
                contour_analysis = ContourAnalysis(detect_params=self.detection_params,
                                                   batch_indeces=self.batch_indeces)
                contour_analysis.analyse_contour(image_index=self.image_index)
        else:
            print("----------------------------")
            contour_analysis = ContourAnalysis(detect_params=self.detection_params,
                                               batch_indeces=self.batch_indeces)
            contour_analysis.analyse_contour(image_index=self.image_index)
