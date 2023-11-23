# TrappedGUVAnalyser

# Description

Detection and analysis of Giant Unilamellar Vesicles (GUVs) in microfluidic traps.

# Installation

Use your package manager to install all packages specified in **requirements.txt** if not already pre-installed.

# Usage

The file to be run is **main.py** for purely text-based interaction, for a graphical user interface (app) run
**gui.py**.

## Quick Start Interface

**input_metadata.xlsx**:

1. In the tab "channels", add your .tif(f)/.lif with the corresponding channel order. The index starts counting from 0,
   meaning the channel order from Stellaris could be interpreted as follows: If TransPMT (Bright-Field) is chosen for
   first channel out of two, it will be "between" both.
   This then e.g. leads an order of 0 - 2 - 1 with Membrane - Protein - Bright-Field, respectively.
   Easy approach: Change the order until the channel view in the GUI is correct or just copy an existing setting from a
   previous succesful analysis of a comparable experiment.
2. Optional: Change the channel labels (so e.g. "CIN85"). Changing the colors (e.g. "Greens") is possible, yet requires
   the matplotlib cmap format and is not possible for all colors

**gui.py**

1. select your .lif/.tif(f) file
2. select the image index (e.g. #2 is the 3rd image from the .lif file)
3. Click on "Run"
4. On the right, the composite image and individual channels with their metadata should appear
5. [ ! ] If not, check the input_metadata.xlsx file!
5. Below, there are 2 panels: The Detection & Analysis Panel.
6. Detection
    7. If you want to (re-)run the detection of contours, check "Run New Detection?"
    8. The (previous/new) detection plots should appear below.
8. Analysis
    9. If you want to (re-)run the analysis, check "Run New Analysis?"
    10. [ ! ] If no detection data is available, the analysis won't run.
    8. The (previous/new) analysis plots should appear below.
10. ! if you want to run your chosen settings for "(!) For all images?"
11. Click "Run" to apply your chosen options.

# Authors

Author: Konstantinos Andreadis @ Koenderink Lab 2023

# License

Copyright (c) 2023 Konstantinos Andreadis @ Koenderink Lab 2023