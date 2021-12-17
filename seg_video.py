
import argparse
import os
from utils.processer import process
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Parse a video to images, and generate pixel-wise mask for salient object.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--video_path', type=str, default='./test.mp4',
        help='Path to the input video')

    parser.add_argument(
        '--seg_type', type=int, default=1,
        help='0 for objects on desk, 1 for furnitures.')

    opt = parser.parse_args()
    
    if not os.path.exists( opt.video_path):
        print("Invaid path for %s",  opt.video_path)
        exit(-1)
    
    # process
    process(opt.video_path, opt.seg_type)

    



    

