import os
# from tools.oss_util import upload_dir
import logging
from datetime import datetime
from collections import OrderedDict
import cv2
import numpy as np
import shutil
import zipfile

from .capture_frame import capture_frame

# from .segmentation_infer.infer import segmentation_infer



def process(video_path):

    PIPELINE = OrderedDict([
        ("VIDEO", video_proc),
        ("SELECT", select_proc),
        ("ZIP", zip_proc),
    ])

    video_dir = os.path.dirname(video_path)
    case_id = os.path.basename(video_path).split('.')[0]

    for _k, _func in PIPELINE.items():
        # input config
        _config_in = {}

        _config_in['video_path'] = video_path
        _config_in['root_dir'] = video_dir
        _config_in['case_id'] = case_id

        # case config setting
        _config_in['images_ori_dir'] = os.path.join(
            video_dir, case_id, "images_ori")
        _config_in['ground_images_dir'] = os.path.join(
            video_dir, case_id, "ground_images")

        is_success = _func(_config_in)
        if not is_success:
            print(_k, " Failed.")

    return True


def video_proc(config):
    capture_frame(config['video_path'],
                  config['images_ori_dir'],
                  sample_rate=4,
                  static_time=0,
                  scale='1/1',
                  img_max_size=1280)
    return True


def select_proc(config):
    filenames = os.listdir(config['images_ori_dir'])
    filenames = [x for x in filenames if "png" in x]
    filenames.sort()

    # select 3 images to annotate ground
    nums_ground = 3
    if 'nums_ground' in config.keys():
        nums_ground = config['nums_ground']
    print("[INFO] Select 3 images.")
    if not os.path.exists(config['ground_images_dir']):
        os.mkdir(config['ground_images_dir'])
    interval_ = len(filenames) / nums_ground
    idx_list = [int(x) for x in list(np.arange(0, len(filenames), interval_))]
    for i in idx_list:
        image_name = filenames[i]
        shutil.copy(os.path.join(config['images_ori_dir'], image_name), os.path.join(
            config['ground_images_dir'], image_name))

    return True, None


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


def zip_proc(config):
    zipf = zipfile.ZipFile(os.path.join(
        config['root_dir'], config['case_id'] + ".zip"), 'w', zipfile.ZIP_DEFLATED)
    zipdir(config['ground_images_dir'], zipf)
    zipf.close()

    return True, None
