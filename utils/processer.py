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

from .segmentation_infer.infer import segmentation_infer

#from .upload_file import upload_file


def process(video_path, seg_type):

    PIPELINE = OrderedDict([
           #("VIDEO", video_proc),
           #("SELECT", select_proc),
           #("SEGMENTATION", segmentation_proc),
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
        _config_in['seg_type'] = seg_type

        # case config setting
        _config_in['images_ori_dir'] = os.path.join(
            video_dir, case_id, "images_ori")
        _config_in['full_images_dir'] = os.path.join(
            video_dir, case_id, "full_images")
        _config_in['ground_images_dir'] = os.path.join(
            video_dir, case_id, "ground_images")
        _config_in['segmentation_images_dir'] = os.path.join(
            video_dir, case_id, "segmentation_images")

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

    # resize all images to 1/8
    scale = 6
    if not os.path.exists(config['full_images_dir']):
        os.mkdir(config['full_images_dir'])

    print("[INFO] Select mini resolution images.")
    for image_name in filenames:
        image = cv2.imread(os.path.join(
            config['images_ori_dir'], image_name), 1)
        h, w, c = image.shape
        image = cv2.resize(image, (w//scale, h//scale))
        cv2.imwrite(os.path.join(config['full_images_dir'], image_name), image)

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

    # select 90 images to seg
    if not os.path.exists(config['segmentation_images_dir']):
        os.mkdir(config['segmentation_images_dir'])
    segmentation_images_dir_rgb = os.path.join(
        config['segmentation_images_dir'], "images_ori")
    if not os.path.exists(segmentation_images_dir_rgb):
        os.mkdir(segmentation_images_dir_rgb)

    print("[INFO] Select 90 images.")
    nums_seg = 90
    if 'nums_seg' in config.keys():
        nums_seg = config['nums_seg']

    interval_ = len(filenames) / float(nums_seg)
    idx_list = [int(x) for x in list(np.arange(0, len(filenames), interval_))]
    for i in idx_list:
        image_name = filenames[i]
        shutil.copy(os.path.join(config['images_ori_dir'], image_name), os.path.join(
            segmentation_images_dir_rgb, image_name))
    return True, None


def segmentation_proc(config):
    try:
        current_dir = os.path.realpath(os.path.dirname(__file__))
        seg_model_dir = os.path.join(current_dir, "assets/")
        

        seg_type = 1
        if "seg_type" in config.keys():
            seg_type = config['seg_type']
        
        if seg_type == 1:
            seg_model_dir = seg_model_dir + "furniture/"
        else:
            seg_model_dir = seg_model_dir + "small_object/"
        print(seg_model_dir)
        # segmentation
        print("[INFO] doing segmentation.")
        segmentation_infer(
            config['segmentation_images_dir'], seg_model_dir, 540)
        return True, None
    except Exception as e:
        return False


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
    zipdir(config['full_images_dir'], zipf)
    zipdir(config['ground_images_dir'], zipf)
    zipdir(config['segmentation_images_dir'], zipf)
    zipf.close()

    return True, None
