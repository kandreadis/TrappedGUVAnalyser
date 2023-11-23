"""
This script: Analysis of results from previous analyses that were saved as .csv files.
Do not run this script directly, it is called by main.py!
For further explanation, read the descriptions at the beginning of each function.
Author: Konstantinos Andreadis @ Koenderink Lab 2023
"""
import os
import sys

import pandas as pd

from scripts.image_import import import_manual_tags
from scripts.visualisation import plot_batch_analysis

resolution_dpi = 200


def import_analysis_csv_files(relative_dir):
    """
    This function reads, interprets and visualises all previous analyses done on trapped GUVs.
    :param relative_dir: Relative directory to folder of all analysis .csv files.
    :type relative_dir: String
    """
    print("Searching for analysis results...")
    all_files = os.listdir(os.path.normpath(sys.path[1] + "/" + relative_dir))
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    print("Found {} csv results analysis file(s)".format(len(csv_files)))
    summary_dict = {
        "image_path": [],
        "image_index": [],
        "n_channels": [],
        "unit": [],
        "data_time_analysis": [],
        "membrane_perimeter": [],
        "membrane_area": [],
        "membrane_diameter_estimate": [],
        "channel_labels": [],
        "channel_colors": [],

    }

    for csv_file in csv_files:
        if "summary" in csv_file:  # and ".lif" in csv_file:
            result = pd.read_csv(os.path.normpath(sys.path[1] + "/" + relative_dir + "/" + csv_file), header=0).to_dict(
                orient="records")[0]
            for key in summary_dict.keys():
                summary_dict[key].append(result[key])
            tags = import_manual_tags(os.path.normpath("data/" + csv_file), img_idx=0)
            if tags is not None:
                print("Found tag(s):{}".format(list(tags.keys())))
        if "cI" in csv_file:
            for found_file in summary_dict["image_path"]:
                if found_file[5:] in os.path.normpath(sys.path[1] + "/" + relative_dir + "/" + csv_file):
                    c_i = pd.DataFrame(
                        pd.read_csv(os.path.normpath(sys.path[1] + "/" + relative_dir + "/" + csv_file),
                                    header=0).to_dict())
    plot_batch_analysis(summary_dict, x="n_channels", y="membrane_diameter_estimate",
                        group_by="image_path")
