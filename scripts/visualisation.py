"""
This script: Collection of visualisation function used by other scripts in this project.
For further explanation, read the descriptions at the beginning of each function.
Author: Konstantinos Andreadis @ Koenderink Lab 2023
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar

resolution_dpi = 300
diverging_cmap = LinearSegmentedColormap.from_list('my_gradient', (
    (0.000, (0.137, 0.314, 1.000)),
    (0.500, (0.706, 0.706, 0.706)),
    (1.000, (0.000, 0.000, 0.000))))


def cmap_to_color(contour_color):
    """
    Minor helper function that converts a matplotlib native cmap fomat to a color
    :param contour_color: Color Map to convert
    :type contour_color: matplotlib cmap String
    :return: Color
    :rtype: String
    """
    if contour_color == "Greens":
        contour_color = "green"
    elif contour_color == "Blues":
        contour_color = "blue"
    elif contour_color == "Greys":
        contour_color = "k"
    elif contour_color == "RdPu":
        contour_color = "purple"
    else:
        contour_color = "k"
    return contour_color


def axes_extent_unit(shape, xscale, yscale):
    """
    Minor helper function that interprets x and y scaling factors for correct plot scale.
    :param shape: Width and height of object to be plotted
    :type shape: tuple of 2 ints
    :param xscale: X scaling factor converting px to real-world unit
    :type xscale: Float
    :param yscale: Y scaling factor converting px to real-world unit
    :type yscale: Float
    :return: Plotting Extents in units
    :rtype: list of 4 floats
    """
    x_extent = [0, shape[0] * xscale]
    y_extent = [0, shape[1] * yscale]
    return [x_extent[0], x_extent[1], y_extent[1], y_extent[0]]


def axis_label_unit(ax, unit):
    """
    Minor helper function that labels the xy axes of a plot.
    :param ax: Plot
    :type ax: Matplotlib ax object (subplot format)
    :param unit: Unit
    :type unit: String
    """
    ax.set_xlabel("x ({})".format(unit))
    ax.set_ylabel("y ({})".format(unit))


def axis_scalebar(ax, unit, dx):
    """
    Minor helper function that adds a scalebar.
    :param ax: Plot object
    :type ax: matplotlib ax object
    :param unit: Unit
    :type unit: String
    :param dx: Length fraction to convert 1? to real unit
    :type dx: float
    """
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(2)
    try:
        if unit == "$\mu$m":
            unit = "um"
        elif unit == "px":
            dx = 1
        ax.add_artist(
            ScaleBar(1, unit, length_fraction=dx, location="upper right", border_pad=0.5, box_color="k", color="w",
                     box_alpha=0.7))
    except:
        pass


def draw_contour_box(ax, tag_label, contour, xscale, yscale, color):
    """
    Minor helper function that draws a box around a contour.
    :param ax: Plot
    :type ax: Matplotlib ax object (subplot format)
    :param tag_label: Label to be plotted next to contour box
    :type tag_label: String
    :param contour: Contour
    :type contour: Project format for contours
    :param xscale: X scaling factor converting px to real-world unit
    :type xscale: Float
    :param yscale: Y scaling factor converting px to real-world unit
    :type yscale: Float
    :param color: Color
    :type color: String
    """
    ax.scatter(contour["xdata"] * xscale, contour["ydata"] * yscale, s=3, c=color)
    ax.add_patch(
        Rectangle((contour["x"] * xscale, contour["y"] * yscale), contour["w"] * xscale, contour["h"] * yscale,
                  edgecolor='red', facecolor="none", lw=2, alpha=0.8))
    ax.text(contour["x"] * xscale, (contour["y"] - 10) * yscale, tag_label)


def draw_valid_contours(valid_contours, label, ax, color, xscale, yscale):
    """
    Minor helper function that draws all valid contours
    :param valid_contours: List of contours
    :type valid_contours: list of dicts
    :param label: Label for legend
    :type label: String
    :param ax: Plot
    :type ax: Matplotlib ax object (subplot format)
    :param color: Color
    :type color: String
    :param xscale: X scaling factor converting px to real-world unit
    :type xscale: Float
    :param yscale: Y scaling factor converting px to real-world unit
    :type yscale: Float
    """
    for (i, contour) in enumerate(valid_contours):
        x_data, y_data, intensities = contour["xdata"], contour["ydata"], contour["intensities"]
        ax.scatter(x_data * xscale, y_data * yscale, s=2, c=cmap_to_color(color), label=label)


def draw_line(line, label, ax, color, xscale, yscale):
    """
    Minor helper function that draws a line.
    :param line: xy coordinates of line
    :type line: array of two float-columns
    :param label: Label for legend
    :type label: String
    :param ax: Plot
    :type ax: Matplotlib ax object (subplot format)
    :param color: Color
    :type color: String
    :param xscale: X scaling factor converting px to real-world unit
    :type xscale: Float
    :param yscale: Y scaling factor converting px to real-world unit
    :type yscale: Float
    """
    ax.plot(line[:, 0] * xscale, line[:, 1] * yscale, color=color, label=label, alpha=0.8)


def calc_line_xy(num_points, start, direction):
    """
    Minor helper function that calculates the coordinates of a parametrised line.
    :param num_points: Number of line points
    :type num_points: Integer
    :param start: Starting coordinate pair
    :type start: list of two floats
    :param direction: Directional vector of line
    :type direction: list of two floats
    :return: Coordinates of parametrised line
    :rtype: array of two float-columns
    """
    line_xy = []
    for t_i in np.arange(-num_points, num_points, 1):
        line_xy.append(start + t_i * direction)
    line_xy = np.asarray(line_xy)
    return line_xy


def plot_composite_and_channels(image_index, channels, channel_colors, channel_labels, image_stack, path, xscale,
                                yscale, unit):
    """
    This function plots the composite of a selected image and the individual channels
    :param image_index: Image Index (for .lif files)
    :type image_index: int
    :param channels: Channel range
    :type channels: array of int
    :param channel_colors: Channel colors as defined by the metadata
    :type channel_colors: List of matplotlib cmap formatted Strings
    :param channel_labels: Channel labels as defined by the metadata
    :type channel_labels: List of string
    :param image_stack: Frames grouped by channels
    :type image_stack: multi-dimensional numpy array/matrix
    :param path: Image Path
    :type path: String
    :param xscale: X scaling factor converting px to real-world unit
    :type xscale: Float
    :param yscale: Y scaling factor converting px to real-world unit
    :type yscale: Float
    :param unit: Unit
    :type unit: String
    """
    print("Saving Composite for {} #{}...".format(path[5:], image_index))
    fig, ax = plt.subplots(figsize=(6, 6))
    n_channels = len(channels)
    plt.title("Composite image #{} for {} channels".format(image_index, n_channels))

    for channel_iter in np.flip(channels):
        alpha = 0.5
        if channel_iter == n_channels - 1:
            ch_index = -1
        else:
            ch_index = channel_iter
        ch_image = image_stack[channel_iter]
        ax.imshow(ch_image, cmap=channel_colors[ch_index], extent=axes_extent_unit(ch_image.shape, xscale, yscale),
                  alpha=alpha)
    axis_scalebar(ax, unit, xscale)
    fig.tight_layout()
    plt.savefig(
        os.path.normpath("results/figures/detection/composite/{}_image_{}_composite.png".format(path[5:], image_index)),
        dpi=resolution_dpi, bbox_inches='tight')
    plt.close()
    # plt.show()

    fig, ax = plt.subplots(ncols=n_channels, figsize=(int(4.5 * n_channels), 4))
    for channel_iter in channels:
        if channel_iter == n_channels - 1:
            ch_index = -1
        else:
            ch_index = channel_iter
        ch_image = image_stack[channel_iter]
        ax[channel_iter].imshow(ch_image, cmap=channel_colors[ch_index],
                                extent=axes_extent_unit(ch_image.shape, xscale, yscale))
        ax[channel_iter].set_title(channel_labels[ch_index])
        ax[channel_iter].set_aspect("equal")
        sm = plt.cm.ScalarMappable(cmap=channel_colors[ch_index])
        sm.set_array(ch_image)
        axis_scalebar(ax[channel_iter], dx=xscale, unit=unit)
        plt.colorbar(sm, ax=ax[channel_iter], fraction=0.046, pad=0.04, label="Intensity (a.u.)")
    fig.tight_layout()
    plt.savefig(
        os.path.normpath(
            "results/figures/detection/separate_channels/{}_image_{}_channels.png".format(path[5:], image_index)),
        dpi=resolution_dpi, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_contours(contour_params, thresh_img, valid_count, image_index, channel_index, channels,
                  image_stack, channel_colors, threshold, path, xscale, yscale, unit):
    """
    This function plots all found (valid) contours.
    :param contour_params: Contour data
    :type contour_params: custom list of dict format
    :param thresh_img: Thresholded image
    :type thresh_img: 8-bit raw image numpy matrix
    :param valid_count: number of valid contours
    :type valid_count: int
    :param image_index: Image Index
    :type image_index: int
    :param channel_index: Channel Index
    :type channel_index: int
    :param channels: Channel range
    :type channels: array of int
    :param channel_colors: Channel colors as defined by the metadata
    :type channel_colors: List of matplotlib cmap formatted Strings
    :param image_stack: Frames grouped by channels
    :type image_stack: multi-dimensional numpy array/matrix
    :param path: Image Path
    :type path: String
    :param xscale: X scaling factor converting px to real-world unit
    :type xscale: Float
    :param yscale: Y scaling factor converting px to real-world unit
    :type yscale: Float
    :param unit: Unit
    :type unit: String
    :param threshold: Lower image threshold value
    :type threshold: Float
    """
    n_channels = len(channels)
    fig, ax = plt.subplots(ncols=4, figsize=(14, 4))
    ax[0].set_title("Channel #{}".format(channel_index + 1))
    ax[1].set_title("Image Threshold = {}".format(round(threshold)))
    ax[2].set_title("Detected {} contour(s)".format(len(contour_params)))
    ax[3].set_title("Detected {} valid contour(s)".format(valid_count))
    ax[2].set_aspect('equal')
    ax[3].set_aspect('equal')
    ch_index = channel_index
    if channel_index == n_channels - 1:
        ch_index = -1
    ch_image = image_stack[channel_index]
    ax[0].imshow(ch_image, cmap=channel_colors[ch_index], extent=axes_extent_unit(ch_image.shape, xscale, yscale))
    extent = axes_extent_unit(thresh_img.shape, xscale, yscale)
    ax[1].imshow(thresh_img, cmap="gray", extent=extent)
    ax[2].imshow(thresh_img, cmap="Greys", extent=extent)
    ax[3].imshow(thresh_img, cmap="Greys", extent=extent)
    for i in [0, 1, 2, 3]:
        axis_scalebar(ax[i], dx=xscale, unit=unit)
    for (contour_index, contour) in enumerate(contour_params):
        tag_label = "#{}".format(contour_index)
        contour_color = channel_colors[ch_index]
        contour_color = cmap_to_color(contour_color)
        draw_contour_box(ax=ax[2], tag_label=tag_label, contour=contour, xscale=xscale, yscale=yscale,
                         color=contour_color)
        if contour_params[contour_index]["valid"]:
            draw_contour_box(ax=ax[3], tag_label=tag_label, contour=contour, xscale=xscale, yscale=yscale,
                             color=contour_color)
    fig.tight_layout()
    plt.savefig(
        os.path.normpath(
            "results/figures/detection/contoured/{}_image_{}_c{}_contoured.png".format(path[5:], image_index,
                                                                                       channel_index + 1)),
        dpi=resolution_dpi, bbox_inches='tight')
    plt.close()


def plot_masked(membrane_contour, masked_contours, channel_colors, channel_labels, xscale, yscale, unit, path,
                image_index):
    """
    This function plots the masked protein channel(s)
    :param membrane_contour: Membrane Contour
    :type membrane_contour: Nx3 (xyI floats)
    :param masked_contours: Masked protein signal
    :type masked_contours: Nx3 (xyI floats)
    :param channel_colors: Channel colors as defined by the metadata
    :type channel_colors: List of matplotlib cmap formatted Strings
    :param channel_labels: Channel labels as defined by the metadata
    :type channel_labels: List of string
    :param xscale: X scaling factor converting px to real-world unit
    :type xscale: Float
    :param yscale: Y scaling factor converting px to real-world unit
    :type yscale: Float
    :param unit: Unit
    :type unit: String
    :param path: Image Path
    :type path: String
    :param image_index: Image Index
    :type image_index: int
    """
    fig, ax = plt.subplots(ncols=len(masked_contours) + 1, figsize=(4.8 * (len(masked_contours) + 1), 4))
    ax[0].set_title("Membrane")
    ax[0].set_aspect("equal")
    ax[0].scatter(membrane_contour[:, 0] * xscale, -membrane_contour[:, 1] * yscale, c=membrane_contour[:, 2],
                  cmap=channel_colors[0], s=3, marker="s", label="Mebrane")
    sm = plt.cm.ScalarMappable(cmap=channel_colors[0])
    sm.set_array(membrane_contour[:, 2])
    plt.colorbar(sm, ax=ax[0], fraction=0.046, pad=0.04, label="Intensity (a.u.)")
    ax[1].set_aspect("equal")
    for object_mask_index, object_mask_profile in enumerate(masked_contours):
        ax[object_mask_index + 1].scatter(object_mask_profile[:, 0] * xscale, -object_mask_profile[:, 1] * yscale,
                                          c=object_mask_profile[:, 2], alpha=0.5,
                                          cmap=channel_colors[object_mask_index + 1], s=3, marker="s")
        sm = plt.cm.ScalarMappable(cmap=channel_colors[object_mask_index + 1])
        sm.set_array(object_mask_profile[:, 2])
        plt.colorbar(sm, ax=ax[object_mask_index + 1], fraction=0.046, label="Intensity (a.u.)")
        ax[object_mask_index + 1].set_aspect("equal")
        ax[object_mask_index + 1].set_title("Masked {}".format(channel_labels[object_mask_index + 1]))

    for axi in ax.ravel():
        axis_scalebar(axi, unit, xscale)
    fig.tight_layout()
    plt.savefig(os.path.normpath(
        "results/figures/analysis/masked_protein/{}_image_{}_masked.png".format(path[5:], image_index)),
        dpi=resolution_dpi, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_reference_projection(contours, image_index, trap_axis, path, show_original_img, projected_contours,
                              channel_colors, channel_labels, unit, xscale, yscale):
    """
    This function plots the projection of contours onto a (trap) reference.
    :param contours: List of contours
    :type contours: list of matrices Nx3 (xyI floats)
    :param image_index:
    :type image_index:
    :param trap_axis:
    :type trap_axis:
    :param path:
    :type path:
    :param show_original_img:
    :type show_original_img:
    :param projected_contours:
    :type projected_contours:
    :param channel_colors:
    :type channel_colors:
    :param channel_labels:
    :type channel_labels:
    :param unit:
    :type unit:
    :param xscale:
    :type xscale:
    :param yscale:
    :type yscale:
    """
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    ax[0].set_title("Detected Reference of image #{}".format(image_index))
    ax[0].set_aspect("equal")
    guv_valid_contours = contours[0]
    trap_valid_contours = contours[1]
    trap_line, trap_line_orth = trap_axis
    original_img = guv_valid_contours[0]["original"]
    if show_original_img:
        ax[0].imshow(original_img, cmap="Greys", extent=axes_extent_unit(original_img.shape, xscale, yscale),
                     alpha=0.5)
    else:
        ax[0].imshow(np.zeros_like(original_img), cmap="Greys",
                     extent=axes_extent_unit(original_img.shape, xscale, yscale), alpha=0.5)
    for trap_index, trap_valid_contour in enumerate(trap_valid_contours):
        draw_valid_contours([trap_valid_contour], "Trap Part {}".format(trap_index), ax[0], "gray", xscale, yscale)
    for guv_index, guv_profile in enumerate(guv_valid_contours):
        label = channel_labels[guv_index].format(guv_index)
        draw_valid_contours([guv_profile], label, ax[0], channel_colors[guv_index], xscale, yscale)
        break
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    draw_line(trap_line, "Reference", ax[0], "red", xscale, yscale)
    draw_line(trap_line_orth, "Reference $_{\perp}$", ax[0], "blue", xscale, yscale)
    ax[1].set_title("GUV projection on reference axis (Membrane)")
    ax[1].axhline(0, color='grey', linestyle='--')
    ax[1].axvline(0, color='grey', linestyle='--')
    ax[1].scatter(projected_contours[0][:, 0] * xscale, projected_contours[0][:, 1] * yscale,
                  c=projected_contours[0][:, 2], cmap=channel_colors[0], s=10)
    ax[1].set_aspect("equal")
    axis_scalebar(ax[0], unit, xscale)
    axis_label_unit(ax[1], unit=unit)
    fig.tight_layout()
    plt.savefig(
        os.path.normpath(
            "results/figures/analysis/reference_projection/{}_image_{}_projection.png".format(path[5:], image_index)),
        dpi=resolution_dpi, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_membrane_geometry(xcords, ycords, curvatures, label, unit, perimeter, area, path,
                           image_index, channel_colors, xscale, yscale, projected_membrane):
    fig, ax = plt.subplots(ncols=2, figsize=(13, 4))
    curvature_label = "Curvature $\kappa$ (" + unit + "$^{-1}$)"
    ax[0].set_title("Membrane " + curvature_label)
    ax[0].set_aspect("equal")
    object_proj_index = 0
    ax[0].scatter(projected_membrane[:, 0] * xscale, projected_membrane[:, 1] * yscale,
                  c=projected_membrane[:, 2], cmap=channel_colors[object_proj_index], s=20, marker=".", alpha=0.6)
    sm = plt.cm.ScalarMappable(cmap=channel_colors[object_proj_index])
    sm.set_array(projected_membrane[:, 2])
    plt.colorbar(sm, ax=ax[0], fraction=0.046, pad=0.14, label="Intensity (a.u.)")
    ax[0].scatter(xcords * xscale, ycords * yscale, c=curvatures, s=10, marker="s", cmap=diverging_cmap, alpha=1)
    ax[0].scatter(0, 0, marker="x")
    sm = plt.cm.ScalarMappable(cmap=diverging_cmap)
    sm.set_array(curvatures)
    plt.colorbar(sm, ax=ax[0], fraction=0.046, pad=0.04, label=curvature_label)
    ax[1].set_title(curvature_label)
    avg_curvature = np.nanmean(curvatures)
    ax[1].hist(curvatures, bins=20, label="avg={} {}{}".format(round(avg_curvature, 4), unit, "$^{-1}$"))
    ax[1].set_ylabel("count")
    ax[1].axvline(avg_curvature, color="red", linestyle="--")
    ax[1].legend(loc="upper left")
    fig.suptitle("{} Geometry (perimeter = {}{}, "
                 "area = {}{}^2, radius ca. {}{})".format(label, round(perimeter, 2), unit, round(area, 2), unit,
                                                          round(perimeter / (np.pi * 2), 2), unit))
    axis_scalebar(ax[0], unit, xscale)
    fig.tight_layout()
    plt.savefig(
        os.path.normpath("results/figures/analysis/membrane_geometry/"
                         "{}_image_{}_membrane-geometry.png".format(path[5:], image_index)),
        dpi=resolution_dpi, bbox_inches='tight')
    plt.close()
    # plt.show()


def plot_protein_accumulation(signal, boundary_curvature, xscale, yscale, channel_colors,
                              channel_labels, unit, path, image_index):
    fig, ax = plt.subplots(ncols=len(signal), figsize=(6 * len(signal), 3))
    curvature_label = "Curvature $\kappa$ (" + unit + "$^{-1}$)"
    for i, signal_ in enumerate(signal):
        if len(signal) == 1:
            axi = ax
        else:
            axi = ax[i]
        signal_xy = signal_[:, :2]
        signal_I = signal_[:, 2]
        axi.set_title("{} Accumulation".format(channel_labels[i + 1]))
        axi.scatter(signal_xy[:, 0] * xscale, signal_xy[:, 1] * yscale, c=signal_I, cmap=channel_colors[i + 1], s=15,
                    marker="s", label=channel_labels[i + 1])
        axi.scatter(boundary_curvature[:, 0] * xscale, boundary_curvature[:, 1] * yscale, c=boundary_curvature[:, 2],
                    cmap=diverging_cmap, s=15, marker="s", alpha=0.8)

        axi.set_aspect("equal")
        sm = plt.cm.ScalarMappable(cmap=diverging_cmap)
        sm.set_array(boundary_curvature[:, 2])
        axis_scalebar(axi, unit, xscale)
        plt.colorbar(sm, ax=axi, fraction=0.046, pad=0.04, label=curvature_label)

    fig.tight_layout()
    plt.savefig(
        os.path.normpath("results/figures/analysis/"
                         "protein_accumulation/{}_image_{}_protein_accumulation.png".format(path[5:], image_index)),
        dpi=resolution_dpi, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_curvature_correlation(signal, nearest_curvature, channel_colors, channel_labels, unit, path, image_index):
    fig, ax = plt.subplots(figsize=(8, 4))
    curvatures = []
    intensities = []
    labels = []
    colors = []

    for i, signal_ in enumerate(signal):
        curvature_near = nearest_curvature[i]
        signal_I = signal_[:, 2]
        curvature_n_bins = 6
        curvature_binned = [np.round((bin_.left + bin_.right) / 2, 2) for bin_ in
                            pd.cut(curvature_near, bins=curvature_n_bins)]
        curvatures = np.append(curvatures, curvature_binned)
        intensities = np.append(intensities, signal_I)
        labels.extend([channel_labels[i + 1] for a in range(len(curvature_binned))])
        colors.extend([cmap_to_color(channel_colors[i + 1])])
    sns.boxplot(ax=ax, x=curvatures, y=intensities, palette=colors, hue=labels, legend="brief",
                medianprops={"color": "w", "linewidth": 2})
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    # sns.violinplot(ax=ax[1], x=curvatures, y=intensities, palette=colors, hue=labels, legend="brief")

    ax.set_xlabel("Curvature $\kappa$ (" + unit + "$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")

    fig.tight_layout()
    plt.savefig(
        os.path.normpath("results/figures/analysis/"
                         "curvature_correlation/{}_image_{}_curvature_correlation.png".format(path[5:], image_index)),
        dpi=resolution_dpi, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_batch_analysis(data, x, y, group_by):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(ax=ax, data=data, x=x, y=y, hue=group_by, legend="brief", medianprops={"color": "w", "linewidth": 2})
    # sns.stripplot(ax=ax, data=data, x=x, y=y, hue=group_by, legend="brief")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    plt.savefig(os.path.normpath("results/figures/result_summary/{}_vs_{}_by_{}.png".format(y,x, group_by)),
                dpi=resolution_dpi,
                bbox_inches='tight')
    # plt.show()
    plt.close()
