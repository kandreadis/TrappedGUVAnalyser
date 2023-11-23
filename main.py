"""
This script: Main run function without any user interface. To run please consult the README file or execute gui.py.
For further explanation, read the descriptions at the beginning of each function.
Author: Konstantinos Andreadis @ Koenderink Lab 2023
"""
import os.path

from scripts.analysis import ContourAnalysis
from scripts.image_import import import_manual_metadata
from scripts.params import detection_params, run_params, batch_analysis_indeces
from scripts.result_analysis import import_analysis_csv_files


def run(detection_params_, run_params_, batch_indeces_):
    """
    This function is the master for all other scripts in this project.
    :param detection_params_: Parameters for detection
    :type detection_params_: dict
    :param run_params_: Global run parameters
    :type run_params_: dict
    :param batch_indeces_: All images OR Indeces of images for batch analysis OR disabler
    :type batch_indeces_: String OR list of int OR None
    """
    print("|=== START ===|")
    detection_params_["images_path"] = os.path.normpath(detection_params_["images_path"])
    try:
        open(detection_params_["images_path"])
        ch_order, ch_labels, ch_colors = import_manual_metadata(detection_params_["images_path"])
        detection_params_["channel_labels"] = ch_labels
        detection_params_["channel_colors"] = ch_colors
    except:
        print("ERROR: Please add [{}] to input_metadata.xlsx !".format(detection_params_["images_path"]))
        return None
    if run_params_["detectContours"] or run_params_["analyseContours"]:
        master = ContourAnalysis(detect_params=detection_params_, batch_indeces=batch_indeces_)
        if run_params_["detectContours"]:
            master.batch_contour_detection()
        if run_params_["analyseContours"]:
            print("=====================")
            master.batch_contour_analysis()
    else:
        master = ContourAnalysis(detect_params=detection_params_, batch_indeces=batch_indeces_)
        master.save_composite()
    if run_params_["analyseBatchResults"]:
        print("=====================")
        relative_dir = os.path.join("results", "csv", "analysis")
        import_analysis_csv_files(relative_dir=relative_dir)
    print("|=== END ===|")


if __name__ == "__main__":
    print("####   Welcome to the terminal of TrappedGUVAnalyser   ####")
    run(detection_params_=detection_params, run_params_=run_params, batch_indeces_=batch_analysis_indeces)
