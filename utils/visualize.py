import os
import cv2
import numpy as np


def vis_arrangeed_images(current_dir_path, selected_image_names, rows, cols, scale = 4):

    h, w = 0, 0
    scaled_h, scaled_w = 0, 0
    padding = 10

    images = []
    for image_name in selected_image_names:
        image = cv2.imread(os.path.join(current_dir_path, image_name), 1)
        h, w, c = image.shape
        scaled_h, scaled_w = h // scale, w // scale
        image = cv2.resize(image, (scaled_w, scaled_h))
        image = cv2.copyMakeBorder(
            image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        images.append(image)
    scaled_h, scaled_w, c = images[0].shape

    vis_image = np.zeros((scaled_h * rows, scaled_w * cols, 3), dtype=np.uint8)
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        vis_image[row * scaled_h: row * scaled_h + scaled_h,
                  col * scaled_w: col * scaled_w + scaled_w] = images[i]
    return vis_image, scaled_h, scaled_w