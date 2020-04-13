import sys 
sys.path.append("..")

from step0_access_path import access_path
from util import get_dir_certain_img, get_dir_img

import cv2

def Video_combine_from_imgs(imgs, file_name="combine.avi"):
    h, w = imgs[0].shape[:2]
    size = (w,h) ### 注意opencv size相關都是 w先再h喔！
    
    out = cv2.VideoWriter(ord_dir + "/" + file_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i, img in enumerate(imgs): 
        out.write(img)
        print("frame %04i comibne ok"%i)
    out.release()

def Video_combine_from_certain_dir(ord_dir, file_name="combine.avi"):
    imgs = get_dir_certain_img( ord_dir, ".png", float_return=False)
    Video_combine_from_imgs(imgs, file_name)

def Video_combine_from_dir(ord_dir, file_name="combine.avi"):
    imgs = get_dir_img( ord_dir, float_return=False)
    Video_combine_from_imgs(imgs, file_name)


ord_dir = access_path + "result" + "/" + "wei_book_tf1_db_20200408-225902_model5_rect2" + "/epoch_add_num"
Video_combine_from_certain_dir(ord_dir, "combine.avi")
# ord_dir = access_path + "result" + "/" + "wei_book_tf1_db_20200410-025655_model6_mrf_rect2" + "/epoch_add_num"
# Video_combine_from_certain_dir(ord_dir, "combine.avi")