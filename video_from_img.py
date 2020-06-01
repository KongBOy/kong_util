import sys 
sys.path.append("..")

# from step0_access_path import access_path
from util import get_dir_certain_img, get_dir_img

import cv2
import numpy as np 
from tqdm import tqdm
def Video_combine_from_imgs(imgs, dst_dir, file_name="combine.avi", tail_long=False):
    h, w = imgs[0].shape[:2]
    size = (w,h) ### 注意opencv size相關都是 w先再h喔！
    
    if(tail_long):
        second = 2
        temp_imgs = np.tile(imgs[-1:], (15*second,1,1,1)) 
        imgs = np.concatenate( (imgs,temp_imgs), axis=0 )

    out = cv2.VideoWriter(dst_dir + "/" + file_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    print("combining frames~~")
    for i, img in enumerate(tqdm(imgs)): 
        out.write(img)
        # print("frame %04i comibne ok"%i)
        # print(".", end="")
        # if(i+1%100==0): print()
    out.release()

def Video_combine_from_certain_dir(ord_dir, dst_dir, file_name="combine.avi"):
    imgs = get_dir_certain_img( ord_dir, ".png", float_return=False)
    Video_combine_from_imgs(imgs, dst_dir, file_name)

def Video_combine_from_dir(ord_dir, dst_dir, file_name="combine.avi", tail_long=True):
    print("doing Video_combine_from_dir")
    imgs = get_dir_img( ord_dir, float_return=False)
    Video_combine_from_imgs(imgs, dst_dir, file_name, tail_long=tail_long)

def Video_combine_from_2_certain_dir(ord_dir1, ord_dir2, dst_dir, file_name="combine_2_dir_imgs.avi"):
    imgs1 = get_dir_certain_img( ord_dir1, ".png", float_return=False)
    imgs2 = get_dir_certain_img( ord_dir2, ".png", float_return=False)
    # print("imgs1.shape",imgs1.shape)
    # print("imgs2.shape",imgs2.shape)
    
    imgs_1_amount = len(imgs1)
    imgs_2_amount = len(imgs2)
    min_amount = min(imgs_1_amount, imgs_2_amount)
    combine_imgs = np.concatenate( (imgs1[:min_amount], imgs2[:min_amount]), axis=1 ) 
    for epoch_string, combine_img in enumerate(tqdm(combine_imgs)):
        cv2.putText(combine_img, "%04i"%(epoch_string), (10, int(combine_img.shape[0]/2) ), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
        print("%04i combine_img write epoch_string ok"%(epoch_string))
    # print("combine_imgs.shape",combine_imgs.shape)
    # cv2.imshow("combine", combine_imgs[0])
    # cv2.waitKey()
    Video_combine_from_imgs(combine_imgs, dst_dir, file_name)

def Video_combine_from_certain_dirs(ord_dirs, dst_dir, file_name="combine_2_dir_imgs.avi"):
    for go_ord_dir, ord_dir in enumerate(tqdm(ord_dirs)):
        print("doing go_ord_dir:", go_ord_dir)
        result_imgs = None
        if(go_ord_dir == 0): ### head
            head_imgs = get_dir_certain_img( ord_dir, ".png", float_return=False)
            result_imgs = head_imgs
        else:
            body_imgs = get_dir_certain_img( ord_dir, ".png", float_return=False)
            min_amount = min(len(head_imgs), len(body_imgs))
            head_imgs = np.concatenate( (head_imgs[:min_amount], body_imgs[:min_amount]), axis=1 )
            result_imgs = head_imgs

    for epoch_string, result_img in enumerate(tqdm(result_imgs)):
        cv2.putText(result_img, "%04i"%(epoch_string), (10, int(result_img.shape[0]/2) ), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
        print("%04i result_img write epoch_string ok"%(epoch_string))

    Video_combine_from_imgs(result_imgs, dst_dir, file_name)


if(__name__=="__main__"):
    from epoch_add_num_into_img import epoch_add_num_into_img
    dst_dir  = access_path + "result" + "/" +"pure_rect2_right-loss_have_shuffle"
    ord_dir1 = access_path + "result" + "/" +"pure_rect2_right-loss_have_shuffle/wei_book_1_type4_complex+page_more_like_20200422-005728_model6_mrf_rect2_127.40_317"
    ord_dir2 = access_path + "result" + "/" +"pure_rect2_right-loss_have_shuffle/wei_book_1_type4_complex+page_more_like_20200422-012527_model5_rect2_128.242_165"       
    ord_dir3 = access_path + "result" + "/" +"pure_rect2_right-loss_have_shuffle/wei_book_2_tf1_db_20200420-145132_model6_mrf_rect2_finish"
    ord_dir4 = access_path + "result" + "/" +"pure_rect2_right-loss_have_shuffle/wei_book_2_tf1_db_20200420-145843_model5_rect2_finish"       
    ord_dir5 = access_path + "result" + "/" +"pure_rect2_right-loss_have_shuffle/wei_book_3_tf1_db+type4_complex+page_more_like_20200422-011813_model5_rect2_127.35_284"
    ord_dir6 = access_path + "result" + "/" +"pure_rect2_right-loss_have_shuffle/wei_book_3_tf1_db+type4_complex+page_more_like_20200422-012313_model6_mrf_rect2_128.245_143"       

    ord_dirs = [ord_dir1,ord_dir2,ord_dir3,ord_dir4,ord_dir5,ord_dir6]    
    Video_combine_from_certain_dirs(ord_dirs, dst_dir, "combine_1,2,3,4,5,6.avi")

    ######################################################
    # Video_combine_from_2_certain_dir(ord_dir1, ord_dir2, dst_dir, "combine_1,2.avi")
    # Video_combine_from_2_certain_dir(ord_dir3, ord_dir4, dst_dir, "combine_3,4.avi")
    # Video_combine_from_2_certain_dir(ord_dir5, ord_dir6, dst_dir, "combine_5,6.avi")

    # result_names = [
    #                 # "wei_book_1_type4_complex+page_more_like_20200413-230418_model5_rect2_127.40-epoch-392",
    #                 # "wei_book_1_type4_complex+page_more_like_20200413-230835_model6_mrf_rect2_128.242-epoch=203",
    #                 # "wei_book_2_tf1_db_20200408-225902_model5_rect2",
    #                 # "wei_book_2_tf1_db_20200410-025655_model6_mrf_rect2",
    #                 # "wei_book_3_tf1_db+type4_complex+page_more_like_20200413-220059_model5_rect2",
    #                 # "wei_book_3_tf1_db+type4_complex+page_more_like_20200413-220341_model6_mrf_rect2_128.243-epoch=183",
                    
    #                 "pure_rect2_right-loss_have_shuffle/wei_book_1_type4_complex+page_more_like_20200422-005728_model6_mrf_rect2_127.40_317",
    #                 "pure_rect2_right-loss_have_shuffle/wei_book_1_type4_complex+page_more_like_20200422-012527_model5_rect2_128.242_165",
    #                 "pure_rect2_right-loss_have_shuffle/wei_book_2_tf1_db_20200420-145132_model6_mrf_rect2_finish",
    #                 "pure_rect2_right-loss_have_shuffle/wei_book_2_tf1_db_20200420-145843_model5_rect2_finish",
    #                 "pure_rect2_right-loss_have_shuffle/wei_book_3_tf1_db+type4_complex+page_more_like_20200422-011813_model5_rect2_127.35_284",
    #                 "pure_rect2_right-loss_have_shuffle/wei_book_3_tf1_db+type4_complex+page_more_like_20200422-012313_model6_mrf_rect2_128.245_143",

    #                ]   
    # for result_name in result_names:
    #     ### 先把 epoch數字 寫上img
    #     ord_dir = access_path + "result" + "/" + result_name 
    #     dst_dir = ord_dir    + "/" + "epoch_add_num_into_img"
    #     epoch_add_num_into_img(ord_dir, dst_dir)

    #     ### 再把img 串成影片
    #     ord_dir = access_path + "result" + "/" + result_name + "/" + "epoch_add_num_into_img"
    #     Video_combine_from_certain_dir(ord_dir, dst_dir, "combine.avi")

    