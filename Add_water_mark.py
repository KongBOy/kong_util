import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def add_water_mark(img_path, water_mark_path, result_path, result_ratio=None, u_crop=None, d_crop=None, l_crop=None, r_crop=None):
    # img_path = "scan_0002.jpg"
    # water_mark_path = "TKUWB_logo.jpg"
    img        = cv2.imread(img_path)
    if(d_crop is not None): img = img[      :d_crop,       :      , :]
    if(r_crop is not None): img = img[      :      ,       :r_crop, :]
    if(u_crop is not None): img = img[u_crop:      ,       :      , :]
    if(l_crop is not None): img = img[      :      , l_crop:      , :]
    img_h, img_w, img_c = img.shape

    water_mark = cv2.imread(water_mark_path)

    water_mark_size = int(min(img_h, img_w) * 0.85)  ### 取 img 的 短邊 然後縮小些
    water_mark = cv2.resize(water_mark, (water_mark_size, water_mark_size))
    water_mark_h, water_mark_w, _ = water_mark.shape

    t = img_h // 2 - water_mark_h // 2
    d = t + water_mark_h
    l = img_w // 2 - water_mark_w // 2
    r = l + water_mark_w

    img = img.astype(np.float32)
    water_mark = water_mark.astype(np.float32)

    img[t:d, l:r] = img[t:d, l:r] - (255 - water_mark) * 0.18
    img = img.clip(0, 255)
    img = img.astype(np.uint8)

    if(result_ratio is not None): img = cv2.resize(img, (int(img_w * result_ratio), int(img_h * result_ratio)))
    cv2.imwrite(result_path, img)
    print(f"{result_path} water_marked ok")
    # plt.imshow(img)
    # plt.show()

result_ratio = 0.45
src_dir = "1 Forest of eternity/0 FS"
result_ratio = 0.50
src_dir = "1 LUNA/0 FS"
d_crop = None
r_crop = None

result_ratio = 0.40
src_dir = "1 The red shadow on the water ripples/1 5 Player ver/0 FS"
d_crop = 3488
r_crop = 2476


dst_dir = f"{src_dir}/water_marked_small"
os.makedirs(dst_dir, exist_ok=True)

file_names = [file_name for file_name in os.listdir(src_dir) if ".jpg" in file_name.lower() ]
file_paths = [f"{src_dir}/{file_name}" for file_name in file_names]

for go_f, file_name in enumerate(file_names):
    result_path = f"{dst_dir}/{file_name}"
    add_water_mark(img_path = file_paths[go_f], water_mark_path = "TKUWB_logo.jpg", result_ratio=result_ratio, result_path = result_path, d_crop=d_crop, r_crop=r_crop)
