"""
This script: Collection of parameters used by other scripts. All changes here affect the whole project at once.
For further explanation, read the descriptions at the beginning of each function.
Author: Konstantinos Andreadis @ Koenderink Lab 2023
"""

detection_params = {
    "images_path": "data/test_trapped_septin_anillin.tif",
    "img_idx": 0,
    "channel_index": 0,
    "unit": "$\mu$m",
}

run_params = {
    "detectContours": False,
    "analyseContours": False,
    "analyseBatchResults": True
}

# batch_analysis_indeces = None
batch_analysis_indeces = "all_images"
