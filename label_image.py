import argparse
import os
from utils.label import startLabelZip
from utils.processer import process
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Parse a video to images, and generate pixel-wise ground plane masks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--video_path', type=str, default='./test.mp4',
        help='Path to the input video')
    opt = parser.parse_args()

    if not os.path.exists(opt.video_path):
        print("Invaid path for %s",  opt.video_path)
        exit(-1)
    process(opt.video_path)

    video_dir = os.path.dirname(opt.video_path)
    case_id = os.path.basename(opt.video_path).split('.')[0]
    zip_path = os.path.join(video_dir, case_id + ".zip")

    if not os.path.exists(zip_path):
        print("Zip file %s not exists." % zip_path)

    startLabelZip(zip_path)

