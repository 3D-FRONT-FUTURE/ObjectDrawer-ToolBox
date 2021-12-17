# video_path: the path of the input video
# seg_type: 0 for small objects (e.g. shoes, toy, cup and so on)
#           1 for furnitures(e.g. chair, table, sofa, bed and so on)

python seg_video.py \
  --video_path /Path/to/your/videos/test.mp4 \
  --seg_type 1