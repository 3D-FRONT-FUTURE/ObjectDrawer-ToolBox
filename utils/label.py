import os
import cv2
import numpy as np
import math
from sys import exit
import zipfile
import shutil
from .visualize import vis_arrangeed_images

# -----------config-------
dist_threshold = 5
padding_size = 20
annotated_skip = 20
max_annotated = 3


static_time = 4
sample_rate = 3
scale = 1
img_max_size = 960
quality_thr = 1
# ------------------------

# ---------------global variables
selected_zip_dir = ""
selected_image_dir = ""
mode = 0  # 0 for running, 1 for finished
fore_points = []
fore_candidate_point = (0, 0)
src_points = []
src_candidate_point = (0, 0)


# double click params
click_last_time = None
vis_all_images = None
double_clicked = False
clicked_x, clicked_y = 0, 0
# -------------------------------

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

def pixel_dist(p, q):
    return np.sqrt(np.sum((np.array(p)-np.array(q))**2))
    # return math.dist(p, q)

#-----------------select image functions-------------------------------------
def double_click_for_selected(event, x, y, flags, params):
    global click_last_time, double_clicked, clicked_x, clicked_y
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(vis_all_images, (x, y), 10, (0, 0, 255), -1)
        clicked_x, clicked_y = x, y
        double_clicked = True


def get_image_index(scaled_h, scaled_w, cols):
    global double_clicked, clicked_x, clicked_y
    double_clicked = False
    selected_col = clicked_x // scaled_w
    selected_row = clicked_y // scaled_h
    index = selected_row * cols + selected_col
    return index

def add_src_point(event, x, y, flags, param):
    global src_points, src_candidate_point
    src_candidate_point = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        src_points.append((x, y))
        
def add_fore_point(event, x, y, flags, param):
    global fore_points, fore_candidate_point
    fore_candidate_point = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        fore_points.append((x, y))


def draw_points(img, points, candidate_point):
    global src_candidate_point, fore_candidate_point, mode
    img_show = np.copy(img)
    if len(points) == 0:
        return img_show

    if len(points) >= 4:
        if pixel_dist(points[0], points[-1]) < dist_threshold:
            mode = 1
        else:
            mode = 0

    if mode == 0:
        # red dot
        for i in range(len(points)):
            cv2.circle(img_show, points[i], 2, (0, 0, 255), -1)
        # red line
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            cv2.line(img_show, p1, p2, (0, 0, 255), 1)
        # candidate line
        if candidate_point != (0, 0):
            cv2.line(img_show, points[-1], candidate_point, (255, 0, 0), 1)

    elif mode == 1:
        # black line
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            cv2.line(img_show, p1, p2, (0, 0, 0), 1, lineType=16)
        cv2.line(img_show, points[0], points[-1], (0, 0, 0), 1, lineType=16)
        src_candidate_point = (0, 0)
        fore_candidate_point = (0, 0)

    return img_show


def clear_points():
    global src_points, fore_points, src_candidate_point, fore_candidate_point
    src_points = []
    src_candidate_point = (0, 0)
    fore_points = []
    src_candidate_point = (0, 0)


def get_white_mask(image):
    mask_r = image[:, :, 2] == 255
    mask_g = image[:, :, 1] == 255
    mask_b = image[:, :, 0] == 255
    mask = np.logical_and(np.logical_and(mask_r, mask_g), mask_b)
    mask = (1 - mask.astype(np.uint8)) * 255
    return mask

def rotate_clockwise(image):
    return cv2.flip(cv2.transpose(image), 1)
def rotate_anti_clockwise(image):
    return cv2.flip(cv2.transpose(image), 0)



def startLabelZip(zip_path):
    global selected_zip_dir
    global selected_image_dir, mode, vis_all_images
    global double_clicked, clicked_x, clicked_y
    selected_zip_dir = os.path.dirname(zip_path)
    valid_filenames = [os.path.basename(zip_path)]
    
    for filename in valid_filenames:
        scene_name = filename[:-4]
        scene_dir = os.path.join(selected_zip_dir, scene_name)
        if not os.path.exists(scene_dir):
            os.mkdir(scene_dir)
        
        label_scene_dir =  os.path.join(selected_zip_dir, "label_" + scene_name)
        if not os.path.exists(label_scene_dir):
            os.mkdir(label_scene_dir)
        
        with zipfile.ZipFile(os.path.join(selected_zip_dir, filename), 'r') as zip_ref:
            zip_ref.extractall(scene_dir)

        # init label ground
        #----------------------------------------------------------------#
        output_ground_dir = os.path.join(label_scene_dir, "ground_mask")
        if not os.path.exists(output_ground_dir):
            os.mkdir(output_ground_dir)

        
        #---------------------------------------------------------#
        #-----------label ground--------------#
        cv2.namedWindow("fore", cv2.WINDOW_AUTOSIZE)
        ground_dir = os.path.join(scene_dir, "ground_images")
        ground_image_names = os.listdir(ground_dir)
        ground_image_names.sort()

        for image_index, ground_image_name in enumerate(ground_image_names):
            fore = cv2.imread(os.path.join(ground_dir, ground_image_name))
            h, w, c = fore.shape
            is_rotate = False
            if h > w:
                fore =  rotate_clockwise(fore)
                is_rotate = True
            h, w, c = fore.shape
            

            while 1:
                padding_image_fore = np.zeros(
                    (h + padding_size*2, w+padding_size*2, 3), dtype=np.uint8) + 255
                padding_image_fore[padding_size:padding_size +
                                h, padding_size:padding_size+w] = fore

                fore_imshow = draw_points(
                    padding_image_fore, fore_points, fore_candidate_point)

                fore_imshow = cv2.putText(fore_imshow, 'label ground  index:[%d/%d]' % (image_index+1, max_annotated), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow("fore", fore_imshow)
                key = cv2.waitKey(100)

                if key == 27:  # Esc
                    clear_points()
                    mode = 0

                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    exit(0)

                elif key == 13 or key == ord('d'):  # Enter
                    mode = 0
                    clear_points()
                    mask = get_white_mask(fore)
                    if is_rotate:
                        mask = rotate_anti_clockwise(mask)
                    cv2.imwrite(os.path.join(output_ground_dir, ground_image_names[image_index]),  mask)
                    image_index += 1
                    break

                elif key == ord('a') and image_index >= 1:
                    mode = 0
                    clear_points()
                    image_index -= 1
                    break

                elif key == 117 or key == ord('s') or key == ord('w'):
                    if mode == 1:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        if len(fore_points) != 0:
                            points = np.array(fore_points).astype(
                                np.int32) - padding_size
                            cv2.fillPoly(mask, [points], (255), 8, 0)
                            fore[mask > 0] = 255
                            fore_points.clear()
                        mode = 0
                cv2.setMouseCallback("fore", add_fore_point, 0)
        cv2.destroyWindow("fore")



        ###----zip--------
        f = zipfile.ZipFile(os.path.join(selected_zip_dir, "label_" + scene_name + ".zip"), 'w', zipfile.ZIP_DEFLATED)
        #zipdir(label_scene_dir, f)
        zipdir(output_ground_dir, f)
        f.close()

        shutil.rmtree(label_scene_dir)
        shutil.rmtree(scene_dir)

        print("Label Finished.")
