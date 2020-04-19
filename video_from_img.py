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

if(__name__=="__main__"):
    from epoch_add_num_into_img import epoch_add_num_into_img

    # result_name = "wei_book_2_tf1_db_20200408-225902_model5_rect2"
    result_names = ["wei_book_1_type4_complex+page_more_like_20200413-230418_model5_rect2_127.40-epoch-392",
                    "wei_book_1_type4_complex+page_more_like_20200413-230835_model6_mrf_rect2_128.242-epoch=203",
                    "wei_book_2_tf1_db_20200408-225902_model5_rect2",
                    "wei_book_2_tf1_db_20200410-025655_model6_mrf_rect2",
                    "wei_book_3_tf1_db+type4_complex+page_more_like_20200413-220059_model5_rect2",
                    "wei_book_3_tf1_db+type4_complex+page_more_like_20200413-220341_model6_mrf_rect2_128.243-epoch=183"
                   ]   
    for result_name in result_names:
        ### 先把 epoch數字 寫上img
        ord_dir = access_path + "result" + "/" + result_name 
        dst_dir = ord_dir    + "/" + "epoch_add_num_into_img"
        epoch_add_num_into_img(ord_dir, dst_dir)

        ### 再把img 串成影片
        ord_dir = access_path + "result" + "/" + result_name + "/" + "epoch_add_num_into_img"
        Video_combine_from_certain_dir(ord_dir, "combine.avi")

    