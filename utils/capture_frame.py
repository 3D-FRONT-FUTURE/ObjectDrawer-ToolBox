import argparse
import ffmpeg    # requirements: pip install ffmpeg-python
import numpy as np
import cv2
import os
import os.path
import sys
sys.path.append(os.getcwd())


quality_thr = 1  # 10


def check_rotation(path_video_file):
    rotateCode = None
    try:
        # this returns meta-data of the video file in form of a dictionary
        meta_dict = ffmpeg.probe(path_video_file)
        # from the dictionary, meta_dict['streams'][i]['tags']['rotate'] is the key we are looking for
        for stream in meta_dict['streams']:
            if 'rotate' not in stream['tags']:
                continue
            # check rotate tag
            if int(stream['tags']['rotate']) == 90:
                rotateCode = cv2.ROTATE_90_CLOCKWISE
            elif int(stream['tags']['rotate']) == 180:
                rotateCode = cv2.ROTATE_180
            elif int(stream['tags']['rotate']) == 270:
                rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
            # tips:
            if rotateCode is not None and ('codec_type' in stream and stream['codec_type'] != 'video'):
                print(stream['codec_type'])
                print('[Stream] >>>> : ', stream)
    except Exception as e:
        print(e)
        print(meta_dict)

    return rotateCode


def load_video(video_path):
    # check rotate flag
    rotateCode = check_rotation(video_path)
    # load video
    vc = cv2.VideoCapture(video_path)
    # transform frame size by rotateCode
    if rotateCode in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        frameSize = (int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)))
    else:
        frameSize = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return vc, rotateCode, frameSize


def correct_rotation(frame, rotateCode):
    if rotateCode is not None:
        return cv2.rotate(frame, rotateCode)
    else:
        return frame


def variance_of_laplacian(img):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(img, cv2.CV_64F).var()


def capture_frame(video_path, imgs_save_dir, sample_rate=4, static_time=1, scale=1/1, img_max_size=960, max_frame_count=550):

    os.makedirs(imgs_save_dir, exist_ok=True)

    video_format = video_path.rsplit('.', 1)[-1]
    if video_format not in ('mp4', 'MP4'):
        print(f"[WARNING] video format is {video_format}, not in `mp4, MP4`.")
    print(video_path)
    vc, rotateCode, frameSize = load_video(video_path)
    print("------------")

    if vc.isOpened():  # 判断是否正常打开
        print("success open")
        #rval, frame = vc.read()
        rval = True
        fps = vc.get(cv2.CAP_PROP_FPS)
        dur = vc.get(cv2.CAP_PROP_FRAME_COUNT)/fps

        # frame count limit for reduce time-cost (in colmap)
        sample_frame_count = vc.get(
            cv2.CAP_PROP_FRAME_COUNT) / int(fps/sample_rate)
        if sample_frame_count < max_frame_count:
            interval = int(fps/sample_rate)
        else:
            interval = int(vc.get(cv2.CAP_PROP_FRAME_COUNT) // max_frame_count)
    else:
        rval = False
        print("fail open")
        return False

    # interval = int(fps/sample_rate)
    #init_interval = 1000
    #thr_interval  = 1000
    #c = init_interval
    try:
        vc.set(cv2.CAP_PROP_POS_MSEC, int(static_time*1000))
    except Exception as e:
        print('The captured video is too short.')
        rval = False

    count = 0
    img_num = 0
    quality = quality_thr
    img = None
    while rval:  # 循环读取视频帧
        # print()
        rval, curr_img = vc.read()
        if not rval:
            print("finish")
            break

        if vc.get(cv2.CAP_PROP_POS_MSEC)/1000. + static_time > dur:
            print("finish")
            break

        # rotate frame
        curr_img = correct_rotation(curr_img, rotateCode)

        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_quality = variance_of_laplacian(curr_gray)
        # print(curr_quality)
        if curr_quality > quality:
            img = curr_img.copy()
            quality = curr_quality
        count += 1

        if count == interval:
            count = 0
            quality = quality_thr
            #img_save_path = os.path.join(imgs_save_dir, video_id + '_' + str(int(img_num)).zfill(8) + '.png')
            img_save_path = os.path.join(
                imgs_save_dir, str(int(img_num)).zfill(8) + '.png')
            if img is not None:
                scale_base = float(img_max_size)/max(img.shape)
                scale_base = min(1, scale_base)
                scale_u = int(scale.split('/')[0])
                scale_d = int(scale.split('/')[1])
                width = int(img.shape[1] * scale_u / scale_d * scale_base)
                height = int(img.shape[0] * scale_u / scale_d * scale_base)
                dsize = (width, height)
                img = cv2.resize(img, dsize)
                cv2.imwrite(img_save_path, img)  # 存储为图像
                if img_num % 50 == 0:
                    print("write image " + str(img_num))
                img_num += 1
            img = None
    # cv2.waitKey(1)
    vc.release()
    print("==================================")
