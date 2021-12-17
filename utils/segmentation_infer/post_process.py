import cv2

import numpy as np

def mask_refine(mask):
    mask = np.copy(mask)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    max_area, target_index = 0, -1
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            target_index = i
            max_area = area
    for i, contour in enumerate(contours):
        if i == target_index:
            continue
        if hierarchy[0][i][3] == target_index:
            continue
        else:
            cv2.drawContours(mask, contours, i, (0), -1)
    return mask
