"""
This script: Complete User Interface of TrappedGUVAnalyser.
For further explanation, read the descriptions at the beginning of each function.
Author: Konstantinos Andreadis @ Koenderink Lab 2023
"""
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image
from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QPushButton, QComboBox, \
    QCheckBox, QMessageBox

from main import run
from scripts.image_import import import_image, import_manual_metadata
from scripts.params import detection_params, run_params


class TrappedGUVAnalyser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.channel_labels = {}
        self.channel_colors = {}
        self.run_button, self.select_image_dropdown, self.select_file_dropdown = None, None, None
        self.run_button = QPushButton("Run")
        self.label_detection = "Run New Detection?"
        self.label_analysis = "Run New Analysis?"
        self.label_batch = "(!) For all images?"
        self.checkBox_detection = QCheckBox(text=self.label_detection)
        self.checkBox_analysis = QCheckBox(text=self.label_analysis)
        self.checkBox_batch = QCheckBox(text=self.label_batch)
        self.image_file_name, self.image_index, self.n_channels, self.file_index, self.num_images = "", 0, 0, 0, 1
        self.data_folder = "data"
        self.file_extensions = [".tif", ".tiff", ".lif"]
        self.setWindowTitle("TrappedGUVAnalyser")
        self.setGeometry(20, 100, 1400, 800)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout()
        self.initialise_widgets()

        print("####   Welcome to the GUI of TrappedGUVAnalyser   ####")
        self.central_widget.setLayout(self.layout)

    def initialise_widgets(self):
        self.check_boxes()
        self.file_select()
        self.import_image_metadata()
        self.image_select()
        self.button_run()
        self.box_composite_channels()
        self.box_detection()
        self.box_analysis()

    def check_boxes(self):
        self.checkBox_detection.stateChanged.connect(self.onStateChanged)
        self.checkBox_analysis.stateChanged.connect(self.onStateChanged)
        self.layout.addWidget(self.checkBox_detection, 2, 0, 1, 1)
        self.layout.addWidget(self.checkBox_analysis, 3, 0, 1, 1)
        self.layout.addWidget(self.checkBox_batch, 4, 0, 1, 1)

    def onStateChanged(self):
        if self.checkBox_detection.isChecked():
            self.checkBox_detection.setText(self.label_detection)
        else:
            self.checkBox_detection.setText(self.label_detection)
        if self.checkBox_analysis.isChecked():
            self.checkBox_analysis.setText(self.label_analysis)
        else:
            self.checkBox_analysis.setText(self.label_analysis)
        if self.checkBox_batch.isChecked():
            self.checkBox_batch.setText(self.label_batch)
        else:
            self.checkBox_batch.setText(self.label_batch)

    @staticmethod
    def resize_height(img, path, width):
        original_width, original_height = Image.open(path).size
        img.setFixedSize(width, (original_height * width) // original_width)
        return img

    @staticmethod
    def resize_width(img, path, height):
        original_width, original_height = Image.open(path).size
        img.setFixedSize((original_width * height) // original_height, height)
        return img

    def run_analysis(self, detect_contours, analyse_contours, batch_run):
        detection_params["images_path"] = os.path.normpath("data/" + self.image_file_name)
        detection_params["img_idx"] = self.image_index
        run_params["detectContours"] = detect_contours
        run_params["analyseContours"] = analyse_contours
        run_params["analyseBatchResults"] = False
        if batch_run:
            batch_indeces = "all_images"
        else:
            batch_indeces = None
        run(detection_params_=detection_params, run_params_=run_params, batch_indeces_=batch_indeces)

    def refresh(self):
        self.import_image_metadata()
        self.run_analysis(detect_contours=self.checkBox_detection.isChecked(),
                          analyse_contours=self.checkBox_analysis.isChecked(),
                          batch_run=self.checkBox_batch.isChecked())
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget in [None, self.select_file_dropdown, self.run_button, self.checkBox_detection,
                          self.checkBox_analysis, self.checkBox_batch]:
                continue
            self.layout.removeWidget(widget)
            widget.deleteLater()
        self.image_select()
        self.box_composite_channels()
        self.box_detection()
        self.box_analysis()

    def file_refresh(self):
        self.import_image_metadata()
        self.run_analysis(detect_contours=False, analyse_contours=False, batch_run=False)
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget in [None, self.select_file_dropdown, self.run_button, self.checkBox_detection,
                          self.checkBox_analysis, self.checkBox_batch]:
                continue
            self.layout.removeWidget(widget)
            widget.deleteLater()
        self.image_select()
        self.box_composite_channels()
        self.box_detection()
        self.box_analysis()

    def file_select(self):
        self.select_file_dropdown = QComboBox()
        self.select_file_dropdown.addItem("Select File...")
        try:
            file_list = [f for f in os.listdir(self.data_folder) if f.endswith(tuple(self.file_extensions))]
            # file_list = [file_list[1]]
        except:
            dlg = QMessageBox(self)
            dlg.setText("ERROR: No image file(s) found in [data/] => Aborting!")
            dlg.exec()
            sys.exit(0)
        self.select_file_dropdown.addItems(file_list)
        try:
            self.select_file_dropdown.setCurrentIndex(1)
        except:
            self.select_file_dropdown.setCurrentIndex(0)
        self.image_file_name = self.select_file_dropdown.currentText()
        self.select_file_dropdown.currentIndexChanged.connect(self.file_refresh)
        self.layout.addWidget(self.select_file_dropdown, 0, 0, 1, 1)

    def import_image_metadata(self):
        self.image_file_name = self.select_file_dropdown.currentText()
        try:
            self.image_index = int(self.select_image_dropdown.currentText()[1:])
        except:
            pass
        imgs = import_image(os.path.normpath("data/" + self.image_file_name), self.image_index,
                            detection_params["unit"])
        image, _, _, _, n_chan, _, img_idx, n_imgs = imgs
        warning_made = False

        try:
            _, self.channel_labels, self.channel_colors = import_manual_metadata(
                os.path.normpath("data/" + self.image_file_name))
        except:
            warning_made = True
            self.image_index = 0
            self.select_file_dropdown.setCurrentIndex(0)
            n_chan = 0
            n_imgs = 0
            dlg = QMessageBox(self)
            dlg.setText(
                "ERROR: Please add [{}] to input_metadata.xlsx !".format(
                    "data/" + self.image_file_name))
            dlg.exec()
        if image is None:
            self.image_index = 0
            imgs = import_image(os.path.normpath("data/" + self.image_file_name), self.image_index,
                                detection_params["unit"])
            image, _, _, _, n_chan, _, img_idx, n_imgs = imgs
            try:
                _, self.channel_labels, self.channel_colors = import_manual_metadata(
                    os.path.normpath("data/" + self.image_file_name))
            except:
                if not warning_made:
                    self.image_index = 0
                    self.select_file_dropdown.setCurrentIndex(0)
                    dlg = QMessageBox(self)
                    dlg.setText(
                        "ERROR: Please add [{}] to input_metadata.xlsx !".format(
                            "data/" + self.image_file_name))
                    dlg.exec()
        if n_chan is not None:
            self.n_channels = n_chan
            self.num_images = n_imgs
        else:
            self.n_channels = 0
            self.num_images = 1
        if self.image_index <= self.num_images - 1 and self.num_images != 1:
            self.image_index = img_idx
        else:
            self.image_index = 0

    def image_select(self):
        self.select_image_dropdown = QComboBox()
        self.select_image_dropdown.addItems(["#" + str(i) for i in range(self.num_images)])
        self.select_image_dropdown.setCurrentIndex(self.image_index)
        self.select_image_dropdown.currentIndexChanged.connect(self.file_refresh)
        self.layout.addWidget(self.select_image_dropdown, 1, 0, 1, 1)

    def button_run(self):
        self.run_button.setStyleSheet("background-color: lightblue; color:black")
        self.run_button.clicked.connect(self.refresh)
        self.layout.addWidget(self.run_button, 5, 0, 1, 1)

    def box_composite_channels(self):
        composite_image = QLabel()
        composite_image.setScaledContents(True)
        composite_height = 130
        channels_image_height = 160
        self.image_file_name = self.select_file_dropdown.currentText()
        comp_path = os.path.normpath(
            "results/figures/detection/composite/{}_image_{}_composite.png".format(self.image_file_name,
                                                                                   self.image_index))
        if os.path.isfile(comp_path):
            pixmap = QPixmap(comp_path)
            composite_image = self.resize_width(img=composite_image, path=comp_path, height=composite_height)
            composite_image.setPixmap(pixmap)
        else:
            composite_image.setStyleSheet("background-color: grey;")
        self.layout.addWidget(composite_image, 0, 1, 6, 1)

        channels_image = QLabel()
        channels_image.setScaledContents(True)
        sep_path = os.path.normpath(
            "results/figures/detection/separate_channels/{}_image_{}_channels.png".format(self.image_file_name,
                                                                                          self.image_index))
        if os.path.isfile(sep_path):
            pixmap = QPixmap(sep_path)
            channels_image = self.resize_width(img=channels_image, path=sep_path, height=channels_image_height)
            channels_image.setPixmap(pixmap)
        else:
            channels_image.setStyleSheet("background-color: grey;")
        self.layout.addWidget(channels_image, 0, 3, 6, 1)

        for channel_index in range(self.n_channels):
            ch_i = channel_index
            if ch_i == self.n_channels - 1:
                ch_i = -1
            channel_label = QLabel(" Ch {} ({})".format(channel_index + 1, self.channel_labels[ch_i]))
            color_box = QLabel()
            color = self.extractColorFromCmap(self.channel_colors[ch_i])
            color_box.setStyleSheet("background-color: {};".format(color.name()))
            self.layout.addWidget(color_box, channel_index, 4, 1, 1)
            self.layout.addWidget(channel_label, channel_index, 4, 1, 1)
        space_1 = QLabel()
        space_1.setStyleSheet("background-color: rgba(255, 255, 255, 0.1);")
        space_1.setFixedWidth(10)
        self.layout.addWidget(space_1, 7, 2, 4, 1)
        space_2 = QLabel()
        space_2.setFixedHeight(10)
        space_2.setStyleSheet("background-color: rgba(255, 255, 255, 0.1);")
        self.layout.addWidget(space_2, 6, 0, 1, 6)
        space_3 = QLabel()
        space_3.setFixedWidth(10)
        space_3.setStyleSheet("background-color: rgba(255, 255, 255, 0.1);")
        self.layout.addWidget(space_3, 0, 2, 6, 1)

    @staticmethod
    def extractColorFromCmap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        rgba_color = cmap(0.5)
        return QColor.fromRgbF(rgba_color[0], rgba_color[1], rgba_color[2])

    def box_detection(self):
        detection_label = QLabel()
        detection_label.setText("Detection Results:")
        detection_label.setFixedHeight(20)
        plot_width = 400
        self.layout.addWidget(detection_label, 7, 0, 1, 2)
        for ch_i in range(self.n_channels):
            detection_image = QLabel()
            detection_image.setScaledContents(True)
            det_img_path = os.path.normpath("results/figures/detection/contoured/" \
                                            "{}_image_{}_c{}_contoured.png".format(self.image_file_name,
                                                                                   self.image_index, ch_i + 1))
            if os.path.isfile(det_img_path):
                pixmap = QPixmap(det_img_path)
                detection_image = self.resize_height(img=detection_image, path=det_img_path, width=plot_width)
                detection_image.setPixmap(pixmap)
            else:
                detection_image.setStyleSheet("background-color: grey;")
            if self.n_channels > 2:
                if ch_i == 0:
                    self.layout.addWidget(detection_image, 8, 0, 1, 2)
                elif ch_i == self.n_channels - 1:
                    self.layout.addWidget(detection_image, 9, 0, 1, 2)
            else:
                self.layout.addWidget(detection_image, 8 + ch_i, 0, 1, 2)

    def box_analysis(self):
        analysis_label = QLabel()
        analysis_label.setText("Analysis Results:")
        analysis_label.setFixedHeight(20)
        self.layout.addWidget(analysis_label, 7, 3, 1, 3)
        plot_height = 130
        analysis_images_mask = QLabel()
        analysis_images_mask.setScaledContents(True)
        mask_img_path = os.path.normpath("results/figures/" \
                                         "analysis/masked_protein/{}_image_{}_masked.png".format(self.image_file_name,
                                                                                                 self.image_index))
        if os.path.isfile(mask_img_path):
            pixmap = QPixmap(mask_img_path)
            analysis_images_mask = self.resize_width(img=analysis_images_mask, path=mask_img_path,
                                                     height=int(plot_height * 1.1))
            analysis_images_mask.setPixmap(pixmap)
        else:
            analysis_images_mask.setStyleSheet("background-color: grey;")
        self.layout.addWidget(analysis_images_mask, 8, 3, 1, 1)

        analysis_images_projection = QLabel()
        analysis_images_projection.setScaledContents(True)
        proj_img_path = os.path.normpath("results/figures/" \
                                         "analysis/reference_projection/{}_image_{}_projection.png".format(
            self.image_file_name, self.image_index))
        if os.path.isfile(proj_img_path):
            pixmap = QPixmap(proj_img_path)
            analysis_images_projection = self.resize_width(img=analysis_images_projection, path=proj_img_path,
                                                           height=int(plot_height * 1.1))
            analysis_images_projection.setPixmap(pixmap)
        else:
            analysis_images_projection.setStyleSheet("background-color: grey;")
        self.layout.addWidget(analysis_images_projection, 8, 4, 1, 1)

        analysis_images_geometry = QLabel()
        analysis_images_geometry.setScaledContents(True)
        geom_img_path = os.path.normpath("results/figures/analysis/membrane_geometry/" \
                                         "{}_image_{}_membrane-geometry.png".format(self.image_file_name,
                                                                                    self.image_index))
        if os.path.isfile(geom_img_path):
            pixmap = QPixmap(geom_img_path)
            analysis_images_geometry = self.resize_width(img=analysis_images_geometry, path=geom_img_path,
                                                         height=int(plot_height * 1.3))
            analysis_images_geometry.setPixmap(pixmap)
        else:
            analysis_images_geometry.setStyleSheet("background-color: grey;")
        self.layout.addWidget(analysis_images_geometry, 9, 3, 1, 3)

        analysis_images_accumulation = QLabel()
        analysis_images_accumulation.setScaledContents(True)
        accum_img_path = os.path.normpath("results/figures/analysis/protein_accumulation/" \
                                          "{}_image_{}_protein_accumulation.png".format(self.image_file_name,
                                                                                        self.image_index))
        if os.path.isfile(accum_img_path):
            pixmap = QPixmap(accum_img_path)
            analysis_images_accumulation = self.resize_width(img=analysis_images_accumulation, path=accum_img_path,
                                                             height=int(plot_height * 1.3))
            analysis_images_accumulation.setPixmap(pixmap)
        else:
            analysis_images_accumulation.setStyleSheet("background-color: grey;")
        self.layout.addWidget(analysis_images_accumulation, 10, 3, 1, 3)

        analysis_images_correlation = QLabel()
        analysis_images_correlation.setScaledContents(True)
        corr_img_path = os.path.normpath("results/figures/analysis/curvature_correlation/" \
                                         "{}_image_{}_curvature_correlation.png".format(self.image_file_name,
                                                                                        self.image_index))
        if os.path.isfile(corr_img_path):
            pixmap = QPixmap(corr_img_path)
            analysis_images_correlation = self.resize_width(img=analysis_images_correlation, path=corr_img_path,
                                                            height=int(plot_height * 1.3))
            analysis_images_correlation.setPixmap(pixmap)
        else:
            analysis_images_correlation.setStyleSheet("background-color: grey;")
        self.layout.addWidget(analysis_images_correlation, 10, 4, 1, 3)


def main():
    app = QApplication(sys.argv)
    window = TrappedGUVAnalyser()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
