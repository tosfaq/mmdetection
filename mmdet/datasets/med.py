import os.path as osp
from mmengine.fileio import get_local_path
from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset

from transforms.loading import load_dicom

import os
import numpy as np


@DATASETS.register_module()
class MedicalDataset(BaseDetDataset):
    """Dataset class that reads image paths from text files and supports YOLO-style annotations."""

    def __init__(self, data_root, ann_file, **kwargs):
        """
        data_root = /kaggle/input/coco-med-output-ring-exp1/coco_med_output_ring_exp1/
        ann_file = med/train.txt
        """
        super().__init__(**kwargs)
        self.data_root = data_root
        # self.data_prefix = data_prefix
        self.ann_file = ann_file  # This should be the .txt file with image paths

        self.img_files = self.load_image_paths(osp.join(data_root, ann_file))
        self.label_files = self._img2label_paths(self.img_files)

    def load_image_paths(self, txt_path):
        """Load image file paths from a txt file."""
        try:
            with open(txt_path, 'r') as file:
                paths = file.read().strip().splitlines()
                parent = osp.dirname(txt_path) + os.sep
                # local to global path
                paths = [path.replace('./', parent) if path.startswith('./') else path for path in paths]
            return sorted(paths)
        except Exception as e:
            raise Exception(f"Error loading image paths from {txt_path}: {str(e)}")

    def _img2label_paths(self, img_paths):
        """Convert image file paths to label file paths."""
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
        return [img_path.replace(sa, sb, 1).rsplit('.', 1)[0] + ".txt" for img_path in img_paths]

    def load_data_list(self) -> list:
        """Load image and annotation data."""
        data_list = []
        for i, (img_path, label_path) in enumerate(zip(self.img_files, self.label_files)):
            img = load_dicom(img_path)
            height, width = img.shape
            img_info = {
                'file_name': osp.basename(img_path),
                'height': height,  # should be set correctly based on actual image read
                'width': width,  # should be set correctly
                'id': i  # osp.splitext(osp.basename(img_path))[0]
            }
            ann_info = self._load_annotations(label_path, width, height)
            data_list.append({'img_info': img_info, 'ann_info': ann_info})
        return data_list

    def _load_annotations(self, label_path, width, height):
        """Load annotations and convert from relative to absolute coordinates."""
        if not osp.exists(label_path):
            return []

        annotations = []
        with open(label_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            cls, x_center_rel, y_center_rel, w_rel, h_rel = map(float, line.strip().split())
            # Convert from relative to absolute coordinates
            x_center_abs = x_center_rel * width
            y_center_abs = y_center_rel * height
            w_abs = w_rel * width
            h_abs = h_rel * height

            x_min = x_center_abs - (w_abs / 2)
            y_min = y_center_abs - (h_abs / 2)
            x_max = x_center_abs + (w_abs / 2)
            y_max = y_center_abs + (h_abs / 2)

            annotations.append({
                'category_id': int(cls),
                'bbox': [x_min, y_min, x_max, y_max],
                'area': w_abs * h_abs,
            })
        return annotations

    def __repr__(self):
        return f"{self.__class__.__name__}(data_root={self.data_root}, ann_file={self.ann_file})"