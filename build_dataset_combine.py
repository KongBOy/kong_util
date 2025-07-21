import os
import shutil
import cv2
import numpy as np
import random
from kong_util.util import get_dir_certain_file_names, get_dir_img_file_names, get_dir_exr, get_dir_mats, method1, get_dir_jpg_names
from kong_util.multiprocess_util import multi_processing_interface

from tqdm import tqdm

import time

def build_datasets(src_in_dir,
                   src_in_word,
                   src_gt_dir,
                   src_gt_word,
                   dst_db_dir,
                   db_name,
                   db_in_name,
                   db_gt_name,
                   train_amount=None,
                   src_rec_hope_dir=None,
                   src_rec_hope_word=None):
    '''
    src_in_dir  ： 要拿來建立db的 model輸入資料 的 dir
    src_in_word ： 要拿來建立db的 model輸入資料 的 dir 內的檔案 要抓什麼關鍵字，比如 ".jpg", ".png" 之類的
    src_gt_dir  ： 要拿來建立db的 model輸出資料 的 dir
    src_gt_word ： 要拿來建立db的 model輸出資料 的 dir 內的檔案 要抓什麼關鍵字，比如 ".jpg", ".png" 之類的
    dst_db_dir  ： 建出來的 db 要放在哪裡
    db_name,    ： 建出來的 db 要叫啥名字
    db_in_name  ： 建出來的 db model輸入資料 的 dir 要叫啥名字，例如 dis_imgs
    db_gt_name  ： 建出來的 db model輸出資料 的 dir 要叫啥名字，例如 flows
    train_amount： 會自動幫你分train, test， 其中 train 的個數要多少
    src_rec_hope_dir  ： 要拿來建立db的 model輸出 做完後處理希望達到最理想效果 的 dir
    src_rec_hope_word ： 要拿來建立db的 model輸出 做完後處理希望達到最理想效果 的 dir 內的檔案 要抓什麼關鍵字，比如 ".jpg", ".png" 之類的
    '''
    ###########################################################################################################
    ### 抓出 src 的檔名
    in_file_names  = get_dir_certain_file_names(src_in_dir, certain_word=src_in_word)
    gt_file_names  = get_dir_certain_file_names(src_gt_dir, certain_word=src_gt_word)
    if(src_rec_hope_dir is not None):
        rec_hope_list = get_dir_certain_file_names(src_rec_hope_dir, certain_word=src_rec_hope_word)
    data_amount = len(in_file_names)
    if(train_amount is None): train_amount = int(data_amount * 0.9)
    # test_amount = data_amount - train_amount

    ###########################################################################################################
    ### 定位各個 dst資料夾位置
    dst_train_dir    = dst_db_dir + "/" + db_name + "/" + "train"                ### 定位 train 資料夾
    dst_train_in_dir = dst_db_dir + "/" + db_name + "/" + "train/" + db_in_name  ### 定位 train_in 資料夾
    dst_train_gt_dir = dst_db_dir + "/" + db_name + "/" + "train/" + db_gt_name  ### 定位 train_gt 資料夾
    dst_test_dir     = dst_db_dir + "/" + db_name + "/" + "test"                 ### 定位 test 資料夾
    dst_test_in_dir  = dst_db_dir + "/" + db_name + "/" + "test/"  + db_in_name  ### 定位 test_in 資料夾
    dst_test_gt_dir  = dst_db_dir + "/" + db_name + "/" + "test/"  + db_gt_name  ### 定位 test_gt 資料夾
    if(src_rec_hope_dir is not None):
        dst_train_rec_hope_dir  = dst_db_dir + "/" + db_name + "/" + "train/" + "/" + "rec_hope"  ### 定位 train_rec_hope 資料夾
        dst_test_rec_hope_dir   = dst_db_dir + "/" + db_name + "/" + "test/"  + "/" + "rec_hope"  ### 定位 test_rec_hope 資料夾

    ### 建立各個資料夾
    Check_dir_exist_and_build_new_dir(dst_train_dir)
    Check_dir_exist_and_build_new_dir(dst_train_in_dir)
    Check_dir_exist_and_build_new_dir(dst_train_gt_dir)
    Check_dir_exist_and_build_new_dir(dst_test_dir)
    Check_dir_exist_and_build_new_dir(dst_test_in_dir)
    Check_dir_exist_and_build_new_dir(dst_test_gt_dir)
    if(src_rec_hope_dir is not None):
        Check_dir_exist_and_build_new_dir(dst_train_rec_hope_dir)
        Check_dir_exist_and_build_new_dir(dst_test_rec_hope_dir)

    ###########################################################################################################
    # ### src ---複製--> dst
    def copy_util(src_dir, dst_dir, file_names, indexes):
        for i in indexes:
            src_in_path = src_dir + "/" + file_names[i]    ### 定位 src_in_path
            dst_in_path = dst_dir + "/" + file_names[i]    ### 定位 dst_in_path
            shutil.copy(src=src_in_path, dst=dst_in_path)  ### src ---複製--> dst

    copy_util(src_in_dir, dst_train_in_dir, in_file_names, range(train_amount))   ### in -> train
    copy_util(src_gt_dir, dst_train_gt_dir, gt_file_names, range(train_amount))   ### gt -> train
    copy_util(src_in_dir, dst_test_in_dir, in_file_names, range(train_amount, data_amount))  ### in -> test
    copy_util(src_gt_dir, dst_test_gt_dir, gt_file_names, range(train_amount, data_amount))  ### gt -> test
    if(src_rec_hope_dir is not None):
        copy_util(src_rec_hope_dir, dst_train_rec_hope_dir, rec_hope_list, range(train_amount))               ### rec_hope -> train
        copy_util(src_rec_hope_dir, dst_test_rec_hope_dir,  rec_hope_list, range(train_amount, data_amount))  ### rec_hope -> test

##############################################################################################################################################################
##############################################################################################################################################################


def Check_img_filename(file_name):
    if(".jpg" in file_name.lower() or "jpeg" in file_name.lower() or ".png" in file_name.lower() or ".bmp" in file_name.lower()):
        return True
    else:
        return False


### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
def Check_dir_exist_and_build(dir_name, show_msg=False):
    if(os.path.isdir(dir_name)):  ### 如果有上次建立的結果要先刪掉
        if(show_msg): print(dir_name, "已存在，不建立新資料夾")
    else:
        os.makedirs( dir_name, exist_ok=True)
        if(show_msg): print("建立 %s 資料夾 完成" % dir_name)

### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
def Check_dir_exist_and_build_new_dir(dir_name, show_msg=False):
    if(os.path.isdir( dir_name)):  ### 如果有上次建立的結果要先刪掉
        if(show_msg): print(dir_name, "已存在，刪除已存在的資料夾，並建立新的資料夾")
        shutil.rmtree( dir_name)
    os.makedirs( dir_name, exist_ok=True)
    if(show_msg): print("建立 %s 資料夾 完成" % dir_name)

### 把圖片重新命名成 流水號
def Page_num(ord_dir, dst_dir):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = os.listdir(ord_dir)
    # print(file_names)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]

    file_names.sort()

    for i, file_name in enumerate(file_names):
        shutil.copy(ord_dir + "/" + file_name, dst_dir + "/" + "%04i.jpg" % (i + 1))
        # print( ord_dir+"/"+ file_name,"copy to", ord_dir+"/"+ "%06i.jpg"%(i+1), "finished!" )
    print(dst_dir, "page_num finish")
##############################################################################################################################################################
##############################################################################################################################################################

def Crop(ord_dir = "",
         dst_dir = "",
         left = 480,
         top  = 300,
         crop_window_size_w = 674 * 3,
         crop_window_size_h = 674 * 4,
         seed = 10,
         name = ""):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    ### .jpg的檔名抓出來
    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]

    ### 抓取影像長寬資訊
    ord_img = cv2.imread(ord_dir + "/" + file_names[0])
    height, width, channel = ord_img.shape

    ### 開始crop囉
    for j, file_name in enumerate(file_names[:]):
        ord_img = cv2.imread(ord_dir + "/" + file_name)

        crop = ord_img[top:top + crop_window_size_h, left:left + crop_window_size_w]  ### 從windows的左上角 框一個 windows的影像出來
        # cv2.imshow("test", crop)
        # cv2.waitKey(0)

        result_file_name = dst_dir + "/" + "%s%s-left=%04i-top=%04i.jpg" % (name, file_name[:-4], left, top)
        cv2.imwrite(result_file_name, crop)
        print(result_file_name, "finished!!")

def Crop_use_center(ord_dir = "",
         dst_dir = "",
         center_xy_file = None,
         crop_window_size_w = 674 * 3,
         crop_window_size_h = 674 * 4,
         seed = 10,
         lt_s_y =   0,  ### left_top_shift_y
         lt_s_x =   0,  ### left_top_shift_x
         lt_a_h =   0,  ### left_top_add_h
         rt_s_y =   0,  ### right_top_shift_y
         rt_s_x =   0,  ### right_top_shift_x
         rt_a_h =   0,  ### right_top_add_h
         ld_s_y =   0,  ### left_down_shift_y
         ld_s_x =   0,  ### left_down_shift_y
         ld_a_h =   0,  ### left_down_add_height
         rd_s_y =   0,  ### right_down_shift_y
         rd_s_x =   0,  ### right_down_shift_x
         rd_a_h =   0   ### right_down_add_height):
         ):

    ### 讀取外部傳進來的 有放center_xy資訊的 .txt
    center_x_list = []
    center_y_list = []
    if(center_xy_file is not None):
        with open(center_xy_file, "r") as f:
            for line in f:
                x, y = line.rstrip("\n").split(",")
                center_x_list.append(int(x))
                center_y_list.append(int(y))
    else:  ### 沒有則設None
        center_x_list = None
        center_y_list = None


    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    ### .jpg的檔名抓出來pip
    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]

    ### 抓取影像長寬資訊
    ord_img = cv2.imread(ord_dir + "/" + file_names[0])
    height, width, channel = ord_img.shape

    ### 開始crop囉
    for j, file_name in enumerate(file_names[:]):
        # if(j != 314):continue
        ord_img = cv2.imread(ord_dir + "/" + file_name)
        if(center_x_list is None and center_y_list is None):
            center_x = int(width / 2)
            center_y = int(height / 2)
            crop_window_size_w = int(width / 2)
            crop_window_size_h = int(height / 2)
        else:
            center_x = center_x_list[j]
            center_y = center_y_list[j]
        ### add_height 或 add_width 的部分 可以自己看結果，往左上(用-) 或 右下(用+) 補都可以，
        ltt = center_y - crop_window_size_h + lt_s_y - lt_a_h * 1             ; ltt = max(ltt, 0)
        ltd = center_y + lt_s_y
        ltl = center_x - crop_window_size_w + lt_s_x - int(lt_a_h * 0.6847)   ; ltl = max(ltl, 0)
        ltr = center_x + lt_s_x
        rtt = center_y - crop_window_size_h + rt_s_y - rt_a_h * 1             ; rtt = max(rtt, 0)
        rtd = center_y + rt_s_y
        rtl = center_x + rt_s_x - int(rt_a_h * 0.6487)
        rtr = center_x + crop_window_size_w + rt_s_x                          ; rtr = min(rtr, width)
        ldt = center_y + ld_s_y
        ldd = center_y + crop_window_size_h + ld_s_y + ld_a_h * 1             ; ldd = min(ldd, height)
        ldl = center_x - crop_window_size_w + ld_s_x                          ; ldl = max(ldl, 0)
        ldr = center_x + ld_s_x + int(ld_a_h * 0.6487)
        rdt = center_y + rd_s_y
        rdd = center_y + crop_window_size_h + rd_s_y + rd_a_h * 1             ; rdd = min(rdd, height)
        rdl = center_x + rd_s_x
        rdr = center_x + crop_window_size_w + rd_s_x + int(rd_a_h * 0.6487)   ; rdr = min(rdr, width)

        crop_left_top   = ord_img[ ltt: ltd, ltl: ltr]  ### 從windows的左上角 框一個 windows的影像出來
        crop_right_top  = ord_img[ rtt: rtd, rtl: rtr]  ### 從windows的左上角 框一個 windows的影像出來
        crop_left_down  = ord_img[ ldt: ldd, ldl: ldr]  ### 從windows的左上角 框一個 windows的影像出來
        crop_right_down = ord_img[ rdt: rdd, rdl: rdr]  ### 從windows的左上角 框一個 windows的影像出來
        # print("center_y-crop_window_size_h + lt_s_y - lt_a_h*1",center_y-crop_window_size_h + lt_s_y - lt_a_h*1) ### 小心有時會變負的
        print("doing:", j + 1, end=", ")
        print("center_y-crop_window_size_h + lt_s_y - lt_a_h*1:", center_y - crop_window_size_h + lt_s_y - lt_a_h * 1, end=", ")
        print("center_y-crop_window_size_h + rt_s_y - rt_a_h*1: ", center_y - crop_window_size_h + rt_s_y - rt_a_h * 1, end=", ")  ### 小心有時會變負的
        print("")
        # cv2.imshow("test", crop)
        # cv2.waitKey(0)

        left_top_file_name   = dst_dir + "/" + "1-%s-%s-cx=%04i-cy=%04i.jpg" % ("lt", file_name[:-4], center_x, center_y)
        right_top_file_name  = dst_dir + "/" + "2-%s-%s-cx=%04i-cy=%04i.jpg" % ("rt", file_name[:-4], center_x, center_y)
        left_down_file_name  = dst_dir + "/" + "3-%s-%s-cx=%04i-cy=%04i.jpg" % ("ld", file_name[:-4], center_x, center_y)
        right_down_file_name = dst_dir + "/" + "4-%s-%s-cx=%04i-cy=%04i.jpg" % ("rd", file_name[:-4], center_x, center_y)
        cv2.imwrite(left_top_file_name  , crop_left_top)
        cv2.imwrite(right_top_file_name , crop_right_top)
        cv2.imwrite(left_down_file_name , crop_left_down)
        cv2.imwrite(right_down_file_name, crop_right_down)
        # print(left_top_file_name, "finished!!")

def Pick_manually(ord_dir, dst_dir, pick_page_indexes):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = get_dir_img_file_names(ord_dir)
    for page_index in pick_page_indexes:
        index = page_index - 1
        shutil.copy(ord_dir + "/" + file_names[index], dst_dir + "/" + file_names[index])


def Resize_hw(ord_dir, dst_dir, height, width, method="cv2"):
    '''
    在這裡寫註解，就可以看到
    '''

    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]
    for file_name in file_names:
        ord_img = cv2.imread(ord_dir + "/" + file_name)
        if(method == "cv2"):
            print("use cv2", end = ", ")
            resize = cv2.resize(ord_img, (width, height), interpolation=cv2.INTER_AREA)  ### neareat, linear, area, cubic, lanczos4 都是過了，area的最好 且 跟scipy 87%像！
        else:
            print("use scipy", end = ", ")
            import scipy.misc
            resize = scipy.misc.imresize(ord_img, [height, width])


        # cv2.imshow("resize",resize)
        # cv2.waitKey(0)
        cv2.imwrite(dst_dir + "/" + file_name, resize)
        print("Resize:", dst_dir + "/" + file_name, "finished!")

def Crop_row_random(ord_dir = "", dst_dir = "", seed=10, crop_num = 4,
                    image_range_width = 800, image_range_height = 1068 ,
                    base_left = 0, base_top = 230,
                    crop_window_size_h = 128 , crop_window_size_w = 800,
                    name = ""):
    random.seed( seed )

    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    ### .jpg的檔名抓出來
    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if ".jpg" in Check_img_filename(file_name)]

    ### 抓取影像長寬資訊
    ord_img = cv2.imread(ord_dir + "/" + file_names[0])
    height, width, channel = ord_img.shape
    # print("ord_img.shape",ord_img.shape)

    max_left = image_range_width  - crop_window_size_w  ### window 左座標 最大範圍
    max_top  = image_range_height - crop_window_size_h  ### window 下座標 最大範圍

    ### 開始crop囉
    for go_file, file_name in enumerate(file_names[:]):
        ord_img = cv2.imread(ord_dir + "/" + file_name)
        for go_crop in range(1, 1 + crop_num):
            left = random.randint(base_left, max_left)  ### 取 左~左最大範圍 間的 隨機一個點當 windows的左
            top  = random.randint(base_top , max_top )  ### 取 上~上最大範圍 間的 隨機一個點當 windows的上
            crop_row = ord_img[top:top + crop_window_size_h, left:left + crop_window_size_w]  ### 從windows的左上角 框一個 windows的影像出來

            # print("top=%i, top_crop_window_size_h=%i, left=%i, left_crop_window_size_w=%i"%(top, top+crop_window_size_h, left,left+crop_window_size_w))
            # cv2.imshow("test", crop_row)
            # cv2.waitKey(0)

            # ignore_list = np.array([2,5,7,10,12,46,48,50,54,56,64])
            # ignore_list -= 1
            # if(go_file in ignore_list):
            #     continue

            # use_list = np.array([3,4,6,8,9,11,13,14,37,55,57,59,60,61,62,63,75,76,80,81,82,83]) -1
            # if(go_file not in use_list):
            #     #print("here")
            #     continue

            result_file_name = dst_dir + "/" + "%s%s-crop_row%i-left=%04i-top=%04i.jpg" % (name, file_name[:-4], go_crop, left, top)
            cv2.imwrite(result_file_name, crop_row)
            print(result_file_name, "finished!!")


def Select_lt_rt_ld_rd_train_test_see(ord_dir, dst_dir, result_dir_name, train_4page_index_list, test_4page_index_list, see_train_4page_index_list):
    for test_page_index in test_4page_index_list:
        if(test_page_index in see_train_4page_index_list):
            print("test_page_index 不可和 see_train_index 重複喔！")
            return

    file_names = get_dir_img_file_names(ord_dir)
    file_amount = len(file_names)       ### 總共有多少個檔案，共分 lt, rt, ld, rd 四種
    page_amount = int(file_amount / 4)  ### 一個種類有多少個檔案
    see_dir   = dst_dir + "/" + "see"   + "/" + result_dir_name
    train_dir = dst_dir + "/" + "train" + "/" + result_dir_name
    test_dir  = dst_dir + "/" + "test"  + "/" + result_dir_name

    Check_dir_exist_and_build_new_dir(see_dir)
    Check_dir_exist_and_build_new_dir(train_dir)
    Check_dir_exist_and_build_new_dir(test_dir)

    ### 先把所有檔案 copy進train資料夾
    # for file_name in file_names:
    #     shutil.copy(ord_dir + "/" + file_name, train_dir + "/" + file_name)

    ### 先把train的檔案 copy進train資料夾
    for page_index in train_4page_index_list:
        ### 定位出lt, rt, ld, rd 的index，注意page_index 要-1 才會等於 array的index喔！
        lt_i = page_index - 1 + page_amount * 0
        rt_i = page_index - 1 + page_amount * 1
        ld_i = page_index - 1 + page_amount * 2
        rd_i = page_index - 1 + page_amount * 3

        shutil.copy(ord_dir + "/" + file_names[lt_i], train_dir + "/" + file_names[lt_i])
        shutil.copy(ord_dir + "/" + file_names[rt_i], train_dir + "/" + file_names[rt_i])
        shutil.copy(ord_dir + "/" + file_names[ld_i], train_dir + "/" + file_names[ld_i])
        shutil.copy(ord_dir + "/" + file_names[rd_i], train_dir + "/" + file_names[rd_i])
    print("%s train finish" % result_dir_name)

    ### 把 test的部分 從train抽除來放進 test和see資料夾
    for page_index in test_4page_index_list:
        ### 定位出lt, rt, ld, rd 的index，注意page_index 要-1 才會等於 array的index喔！
        lt_i = page_index - 1 + page_amount * 0
        rt_i = page_index - 1 + page_amount * 1
        ld_i = page_index - 1 + page_amount * 2
        rd_i = page_index - 1 + page_amount * 3

        ### 先把 test 的部分，從 train資料夾 copy 進 test資料夾/see資料夾
        shutil.copy(ord_dir + "/" + file_names[lt_i], test_dir + "/" + file_names[lt_i])
        shutil.copy(ord_dir + "/" + file_names[rt_i], test_dir + "/" + file_names[rt_i])
        shutil.copy(ord_dir + "/" + file_names[ld_i], test_dir + "/" + file_names[ld_i])
        shutil.copy(ord_dir + "/" + file_names[rd_i], test_dir + "/" + file_names[rd_i])
        shutil.copy(ord_dir + "/" + file_names[lt_i],  see_dir + "/" + "test_" + file_names[lt_i])
        shutil.copy(ord_dir + "/" + file_names[rt_i],  see_dir + "/" + "test_" + file_names[rt_i])
        shutil.copy(ord_dir + "/" + file_names[ld_i],  see_dir + "/" + "test_" + file_names[ld_i])
        shutil.copy(ord_dir + "/" + file_names[rd_i],  see_dir + "/" + "test_" + file_names[rd_i])

        ### copy 完後 把 train資料夾內的 test部分刪除
        if(page_index in train_4page_index_list):
            os.remove(train_dir + "/" + file_names[lt_i])
            os.remove(train_dir + "/" + file_names[rt_i])
            os.remove(train_dir + "/" + file_names[ld_i])
            os.remove(train_dir + "/" + file_names[rd_i])
    print("%s test and test_see finish" % result_dir_name)

    ### 把 想看的train 放進去 see資料夾
    for page_index in see_train_4page_index_list:
        lt_i = page_index - 1 + page_amount * 0
        rt_i = page_index - 1 + page_amount * 1
        ld_i = page_index - 1 + page_amount * 2
        rd_i = page_index - 1 + page_amount * 3
        shutil.copy(train_dir + "/" + file_names[lt_i],  see_dir + "/" + "train_" + file_names[lt_i])
        shutil.copy(train_dir + "/" + file_names[rt_i],  see_dir + "/" + "train_" + file_names[rt_i])
        shutil.copy(train_dir + "/" + file_names[ld_i],  see_dir + "/" + "train_" + file_names[ld_i])
        shutil.copy(train_dir + "/" + file_names[rd_i],  see_dir + "/" + "train_" + file_names[rd_i])
    print("%s train_see finish" % result_dir_name)

    print("Select_lt_rt_ld_rd_train_test_see finish~~")




def Smooth_curl_fold_page_select_see(ord_dir, train_indexes, test_indexes):
    train_in_dir = ord_dir + "/train/dis_imgs/"
    train_gt_dir = ord_dir + "/train/move_maps/"
    test_in_dir = ord_dir + "/test/dis_imgs/"
    test_gt_dir = ord_dir + "/test/move_maps/"

    see_in_dir = ord_dir + "/see/dis_imgs/"
    see_gt_dir = ord_dir + "/see/move_maps/"
    Check_dir_exist_and_build_new_dir(see_in_dir)
    Check_dir_exist_and_build_new_dir(see_gt_dir)

    ### 000000-3a1-I1-patch.bmp
    ### 000000_train.npy
    for i, train_index in enumerate(train_indexes):
        describe = ""
        if  (i == 0): describe = "000-train_curl_str"
        elif(i == 1): describe = "001-train_curl_img"
        elif(i == 2): describe = "002-train_curl_lin"
        elif(i == 3): describe = "003-train_fold_str"
        elif(i == 4): describe = "004-train_fold_img"
        elif(i == 5): describe = "005-train_fold_lin"
        elif(i == 6): describe = "006-train_page_str"
        elif(i == 7): describe = "007-train_page_img"
        elif(i == 8): describe = "008-train_page_lin"
        shutil.copy(train_in_dir + "%06i-3a1-I1-patch.bmp" % train_index, see_in_dir + describe + "-%06i-3a1-I1-patch.bmp" % train_index)
        shutil.copy(train_gt_dir + "%06i_train.npy" % train_index,        see_gt_dir + describe + "-%06i_train.npy" % train_index)

    for i, test_index in enumerate(test_indexes):
        describe = ""
        if  (i == 0): describe = "009-test_curl_str"
        elif(i == 1): describe = "010-test_curl_img"
        elif(i == 2): describe = "011-test_curl_lin"
        elif(i == 3): describe = "012-test_fold_str"
        elif(i == 4): describe = "013-test_fold_img"
        elif(i == 5): describe = "014-test_fold_lin"
        elif(i == 6): describe = "015-test_page_str"
        elif(i == 7): describe = "016-test_page_img"
        elif(i == 8): describe = "017-test_page_lin"
        shutil.copy(test_in_dir + "%06i-3a1-I1-patch.bmp" % test_index, see_in_dir + describe + "-%06i-3a1-I1-patch.bmp" % test_index)
        shutil.copy(test_gt_dir + "%06i_test.npy" % test_index,         see_gt_dir + describe + "-%06i_test.npy" % test_index)



def Split_train_test(ord_dir, dst_dir, train_dir_name = "", test_dir_name = "",
                     train_amount=70, test_amount = 13):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    train_dir = train_dir_name
    test_dir = test_dir_name

    Check_dir_exist_and_build(dst_dir + "/" + train_dir)
    Check_dir_exist_and_build(dst_dir + "/" + test_dir)
    os.makedirs( dst_dir + "/" + train_dir, exist_ok = True )
    os.makedirs( dst_dir + "/" + test_dir, exist_ok = True )

    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]

    # ignore_list = np.array([2,5,7,10,12,48,56]) -1

    for i, file_name in enumerate(file_names):
        # if( i in ignore_list):
        #     continue
        if( i < train_amount):
            shutil.copy( ord_dir + "/" + file_name, dst_dir + "/" + train_dir  + "/" + file_name )
            print(       ord_dir + "/" + file_name, dst_dir + "/" + train_dir  + "/" + file_name, "finished!" )
        else:
            shutil.copy( ord_dir + "/" + file_name, dst_dir + "/" + test_dir  + "/" + file_name )
            print(       ord_dir + "/" + file_name, dst_dir + "/" + test_dir  + "/" + file_name, "finished!" )

##############################################################################################################################################################
##############################################################################################################################################################

def Find_db_left_top_right_down(ord_dir, padding = 0, search_amount=-1):  ### padding是 印表機 印出來旁邊自動padding的空白，要自己去嘗試為多少喔！
    print("    Find_db_left_top_right_down")
    padding = int(padding)  ### 可以把最後找到的 ltrd 往外pad，ex：left-padding, top-padding, right+padding, down+padding

    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]
    file_names.sort()

    ### 抓取影像長寬資訊
    ord_img = cv2.imread(ord_dir + "/" + file_names[0])
    height, width = ord_img.shape[:2]

    lefts  = []
    tops   = []
    rights = []
    downs  = []
    if(search_amount == -1): search_amount = len(file_names)
    for file_name in tqdm(file_names[0:search_amount]):
        ord_img = cv2.imread(ord_dir + "/" + file_name, 0)
        _, thresh1 = cv2.threshold(ord_img, 127, 255, cv2.THRESH_BINARY_INV)  ### 二值化影像
        # cv2.imshow("thresh1",thresh1)
        # cv2.waitKey(0)

        ### 以下使用argmax()的道理：因為 用 width_sum!=0，結果會是True/False，False代表0，True代表1，
        ### 用argmax()時，如果最大值相同，會回傳第一個找到的最大值，所以才會取到 最左邊或上邊 而不是 最右邊或下邊 的index喔！
        width_sum = thresh1.sum(axis=0)  ### shape 為 (2481,)   ### 統計 垂直 值條圖的概念，找出左右最大的range
        lefts.append( (width_sum != 0).argmax() )  ### left
        rights.append( width - (width_sum != 0)[::-1].argmax() )  ### right

        height_sum = thresh1.sum(axis=1)  ### shape 為 (3508,)  ### 統計 水平 值條圖的概念，找出上下最大的range
        tops.append( (height_sum != 0).argmax() )  ### top
        downs.append( height - (height_sum != 0)[::-1].argmax() )  ### down

        # print(lefts[-1],rights[-1],tops[-1], downs[-1])

    left = min(lefts) - padding   ; left  = max(0, left )      ### 注意padding完可能超出邊界，超出去要拉回來到邊界上喔！
    top  = min(tops)  - padding   ; top   = max(0, top  )      ### 注意padding完可能超出邊界，超出去要拉回來到邊界上喔！
    right = max(rights) + padding ; right = min(width, right)  ### 注意padding完可能超出邊界，超出去要拉回來到邊界上喔！
    down = max(downs)   + padding ; down  = min(height, down)  ### 注意padding完可能超出邊界，超出去要拉回來到邊界上喔！

    print("left", left, "top", top, "right", right, "down", down)

    return left, top, right, down


def Center_draw_photo_frame(ord_dir, dst_dir, thick_v=200, thick_h=100, color = (0, 0, 0)):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]
    file_names.sort()

    ### 抓取影像長寬資訊
    ord_img = cv2.imread(ord_dir + "/" + file_names[0])
    height, width, channel = ord_img.shape

    # white = (255, 255, 255)
    # black = (  0,   0,   0)
    # color = white
    # thick = 200

    for file_name in file_names[0:]:
        ord_img = cv2.imread(ord_dir + "/" + file_name)
        # canvas = np.zeros((300, 300, 3), dtype="uint8") #3
        # cv2.line( img = canvas, pt1 = (0, 0), pt2 = (300, 300), color = color,thickness = 20) ###不能用畫線的，因為thickness大到一定程度就大不上去了

        proc_img = ord_img.copy()
        cv2.rectangle(proc_img, (int(width / 2 - thick_v / 2) , 0), (int(width / 2 + thick_v / 2), height), color, -1 )  ### 畫 vertical
        cv2.rectangle(proc_img, (0, int(height / 2 - thick_h / 2)), (width, int(height / 2 + thick_h / 2)), color, -1 )  ### 畫 horizontal
        cv2.imwrite(dst_dir + "/" + file_name, proc_img)

        # cv2.imshow("proc_img",proc_img)
        # cv2.waitKey(0)
        print(dst_dir + "/" + file_name, "finished!!")

def Photo_frame_padding(ord_dir, dst_dir, left_pad=140, top_pad=50, right_pad=140, down_pad=50, color=(255, 255, 255)):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]
    file_names.sort()

    ### 抓取影像長寬資訊
    ord_img = cv2.imread(ord_dir + "/" + file_names[0])
    height, width, channel = ord_img.shape

    pad_height = height + top_pad + down_pad
    pad_width  = width  + left_pad + right_pad

    color = np.array(color, dtype = np.uint8)
    canvas = np.tile(color, pad_height * pad_width).reshape(pad_height, pad_width, 3)
    for file_name in file_names[:]:
        ord_img = cv2.imread(ord_dir + "/" + file_name)
        canvas[top_pad:top_pad + height, left_pad:left_pad + width] = ord_img
        cv2.imwrite(dst_dir + "/" + file_name, canvas)

        # cv2.imshow("canvas",canvas)
        # cv2.waitKey(0)
        print(dst_dir + "/" + file_name, "finished!!")

def Pad_white_board(ord_dir, dst_dir, pad_direction=3, board=50):
    """
    pad_direction: 1l, 2t, 3r, 4d 補白色邊
    board: 白邊寬度
    """
    file_names = get_dir_img_file_names(ord_dir)
    Check_dir_exist_and_build_new_dir(dst_dir)


    pad_direction = 3
    board = 80

    for file_name in file_names:
        img = cv2.imread(file_name)
        canvas = np.ones(shape=img.shape, dtype=np.uint8) * 255
        if  (pad_direction == 1): canvas[:, board:, :] = img[:, :-board, :]  ### 左補白邊
        elif(pad_direction == 3): canvas[:, :-board, :] = img[:, board:, :]  ### 右補白邊
        elif(pad_direction == 2): canvas[board:, :, :] = img[:-board, :, :]  ### 頂補白邊
        elif(pad_direction == 4): canvas[:-board, :, :] = img[board:, :, :]  ### 底補白邊

        cv2.imwrite(dst_dir + "/" + file_name, canvas)


def Save_as_gray(ord_dir, dst_dir, gray_three_channel=True):
    print("doing Save_as_gray")
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = [file_name for file_name in os.listdir(ord_dir) if Check_img_filename(file_name)]
    for file_name in tqdm(file_names):
        gray_img = cv2.imread(ord_dir + "/" + file_name, 0)
        if(gray_three_channel):
            gray_img = gray_img[:, :, np.newaxis]
            gray_img = np.tile(gray_img, (1, 1, 3))
        cv2.imwrite(dst_dir + "/" + file_name, gray_img)

def _save_img(start_index, amount, image_type, ord_dir, dst_dir, file_names, gray, gray_three_channel, delete_ord_file, show_msg, quality_list):
    for file_name in tqdm(file_names[start_index:start_index + amount]):  ### 以下是拿 bmp 當例子，可以自行套其他格式喔！
        file_title, file_ext = file_name.split(".")      ### 把 檔名前半 後半 分開
        if(file_ext != image_type):                      ### 如果附檔名不是bmp，把圖讀出來，存成bmp
            img = cv2.imread(ord_dir + "/" + file_name)  ### 把圖讀出來
            if  ( gray ):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  ### 如果要轉灰階，就轉灰階，轉完注意只有 h,w，沒有channel喔！
                if(gray_three_channel):            ### (h,w)   轉成 (h,w,3)
                    img = img[:, :, np.newaxis]    ### (h,w)   轉 (h,w,1)
                    img = np.tile(img, (1, 1, 3))  ### (h,w,1) 擴增成 (h,w,3)
            if(quality_list is None): cv2.imwrite(dst_dir + "/" + file_title + "." + image_type, img)  ### 存成bmp
            else: cv2.imwrite(dst_dir + "/" + file_title + "." + image_type, img, quality_list)       ### 存成bmp，且 用指定的壓縮品質
            if(show_msg): print("Save_as_%s" % image_type, ord_dir + "/" + file_name, "save as", dst_dir + "/" + file_title + ".%s" % image_type, "finish~~")
            # print("Save_as_%s"%image_type ,"finish~~")

            if(delete_ord_file): os.remove(ord_dir + "/" + file_name)  ### 把原檔刪掉，做一張刪一張比較不占空間且時間跟 全部做完再刪印象中是差不多的~~


def _Save_as_certain_image_type(image_type, ord_dir, dst_dir, gray=False, gray_three_channel=False, delete_ord_file=False, show_msg=False, quality_list=None, multiprocess=False, core_amount=1):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    ### 以下註解都是用 bmp當例子
    Check_dir_exist_and_build(dst_dir)

    file_names = [file_name for file_name in os.listdir(ord_dir) if Check_img_filename(file_name)]
    if(multiprocess and core_amount > 1):  ### 有用multiprocess
        from kong_util.multiprocess_util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount, task_amount=len(file_names), task=_save_img, task_args= [image_type, ord_dir, dst_dir, file_names, gray, gray_three_channel, delete_ord_file, show_msg, quality_list] )
    else:  ### 沒有用multiprocess
        _save_img(0, len(file_names), image_type, ord_dir, dst_dir, file_names, gray, gray_three_channel, delete_ord_file, show_msg, quality_list)

def Save_as_jpg(ord_dir, dst_dir, gray=False, gray_three_channel=False, delete_ord_file=False, quality_list=None, multiprocess=True, core_amount=8):  ### jpg才有失真壓縮的概念，bmp沒有喔！
    print("doing Save_as_jpg")
    start_time = time.time()
    _Save_as_certain_image_type("jpg", ord_dir, dst_dir, gray=gray, gray_three_channel=gray_three_channel, delete_ord_file=delete_ord_file, quality_list=quality_list, multiprocess=multiprocess, core_amount=core_amount)
    print("Save_as_jpg cost time:", time.time() - start_time)

def Save_as_bmp(ord_dir, dst_dir, gray=False, gray_three_channel=False, delete_ord_file=False, multiprocess=False, core_amount=1):
    _Save_as_certain_image_type("bmp", ord_dir, dst_dir, gray=gray, gray_three_channel=gray_three_channel, delete_ord_file=delete_ord_file)

def Save_as_png(ord_dir, dst_dir, gray=False, gray_three_channel=False, delete_ord_file=False, multiprocess=False, core_amount=1):
    _Save_as_certain_image_type("png", ord_dir, dst_dir, gray=gray, gray_three_channel=gray_three_channel, delete_ord_file=delete_ord_file)


##############################################################################################################################################################
##############################################################################################################################################################
def _matplot_visual(imgs, file_names, dst_dir, img_type=None):
    import matplotlib.pyplot as plt
    ### 建立放結果的資料夾
    Check_dir_exist_and_build(dst_dir + "/" + "matplot_visual")

    for i, file_name in enumerate(tqdm(file_names)):
        name = file_name.split(".")[0]  ### 把 檔名的 名字部分取出來
        if  (img_type is None         ): plt.imshow(imgs[i])
        elif(img_type in ["bm", "uv"] ): plt.imshow(method1(imgs[i, ..., 0], imgs[i, ..., 1] * -1))

        plt.savefig(dst_dir + "/" + "matplot_visual" + "/" + name)
        plt.close()


def Save_exr_as_mat(ord_dir, dst_dir, key_name, matplot_visual=False, print_msg=False):
    ### 建立放結果的資料夾
    Check_dir_exist_and_build(dst_dir)

    from hdf5storage import savemat
    from kong_util.util import get_dir_exr

    file_names = get_dir_certain_file_names(ord_dir, ".exr")
    imgs = get_dir_exr(ord_dir)

    for i, file_name in enumerate(tqdm(file_names)):
        name = file_name.split(".")[0]
        savemat(dst_dir + "/" + name + ".mat", {key_name: imgs[i]} )
        if(print_msg): print(dst_dir + "/" + name + ".mat")

    if(matplot_visual): _matplot_visual(imgs, file_names, dst_dir, key_name)

def Save_mat_as_npy(ord_dir, dst_dir, key_name, matplot_visual=False ):
    ### 建立放結果的資料夾
    Check_dir_exist_and_build(dst_dir)

    imgs = get_dir_mats(ord_dir, key_name)
    file_names = get_dir_certain_file_names(ord_dir, ".mat")
    for i, file_name in enumerate(tqdm(file_names)):
        name = file_name.split(".")[0]
        np.save( dst_dir + "/" + name , imgs[i])

    if(matplot_visual): _matplot_visual(imgs, file_names, dst_dir, key_name)

def Save_exr_as_npy(ord_dir, dst_dir, rgb=False, matplot_visual=False):  ### 不要 float_return = True 之類的，因為他存的時候不一定用float32喔！rgb可以轉，已用網站生成的結果比較確認過囉～https://www.onlineconvert.com/exr-to-mat
    ### 建立放結果的資料夾
    Check_dir_exist_and_build(dst_dir)
    if(matplot_visual): Check_dir_exist_and_build(dst_dir + "/" + "matplot_visual")

    imgs = get_dir_exr(ord_dir, rgb)  ### 把exr 轉成 imgs
    file_names = get_dir_certain_file_names(ord_dir, ".exr")  ### 拿到exr的檔名
    print("Save_exr_as_npy")
    for i, file_name in enumerate(tqdm(file_names)):
        name, _ = file_name.split(".")  ### 把 檔名的 名字部分取出來
        np.save( dst_dir + "/" + name, imgs[i] )  ### 改存成 .npy

    if(matplot_visual): _matplot_visual(imgs, file_names, dst_dir)


from kong_util.util import get_exr
def _save_exr_as_npy(start_i, amount, ord_dir, dst_dir, file_names):
    for i in tqdm(range(start_i, start_i + amount)):
        file_name = file_names[i]
        name, _ = file_name.split(".")  ### 把 檔名的 名字部分取出來
        ord_exr_path = ord_dir + "/" + file_name
        dst_npy_path = dst_dir + "/" + name

        exr_img = get_exr(ord_exr_path)
        np.save( dst_npy_path, exr_img )  ### 改存成 .npy

def Save_exr_as_npy2(ord_dir, dst_dir, rgb=False, matplot_visual=False):  ### 不要 float_return = True 之類的，因為他存的時候不一定用float32喔！rgb可以轉，已用網站生成的結果比較確認過囉～https://www.onlineconvert.com/exr-to-mat
    ### 建立放結果的資料夾
    Check_dir_exist_and_build(dst_dir)
    if(matplot_visual): Check_dir_exist_and_build(dst_dir + "/" + "matplot_visual")

    file_names = get_dir_certain_file_names(ord_dir, ".exr")  ### 拿到exr的檔名
    print("Save_exr_as_npy")
    multi_processing_interface(core_amount=30, task_amount=len(file_names), task=_save_exr_as_npy, task_args=[ord_dir, dst_dir, file_names])

##############################################################################################################################################################
def Save_npy_path_as_knpy(src_path, dst_path):
    with open(src_path, "rb") as fr:        ### 把 .npy 用 open 且 read byte 的形式打開
        byte_strs = []                      ### fr不能直接用，要用iter的方式才能讀內容
        for byte_str in fr:                 ### 所以丟進去 for 把所有 byte_str抓出來囉！
            byte_strs.append(byte_str)

        with open(dst_path, "wb") as fw:    ### 經過觀察，只要去掉第一個byte_str就可以去掉numpy的標頭檔資訊拉
            for byte_str in byte_strs[1:]:  ### 所以從 byte_strs的第二個元素開始把 byte_str寫進新檔案，且命名為 ".knpy"，kong_numpy的概念ˊ口ˋ
                fw.write(byte_str)

def _save_npy_dir_as_knpy(start_i, amount, ord_dir, dst_dir, npy_file_names):
    for i in tqdm(range(start_i, start_i + amount)):
        npy_file_name = npy_file_names[i]
        file_name = npy_file_name.split(".")[0]  ### 把 "檔案名". "npy" 分開，只抓檔案名等等才好存 ".knpy"
        src_path = ord_dir + "/" + npy_file_name
        dst_path = dst_dir + "/" + file_name + ".knpy"
        Save_npy_path_as_knpy(src_path, dst_path)


### knpy 是 kong_numpy的意思喔ˊ口ˋ，存的內容是把 numpy 的開頭資訊拿掉，讓tensorflow 可以直接decode！
def Save_npy_dir_as_knpy(ord_dir, dst_dir, core_amount=1):
    ### 建立放結果的資料夾
    Check_dir_exist_and_build(dst_dir)

    npy_file_names = get_dir_certain_file_names(ord_dir, ".npy")   ### 把想轉換的 .npy 的檔案名讀出來
    print("Save_npy_dir_as_knpy")
    if(core_amount <= 1): _save_npy_dir_as_knpy(start_i=0, amount=len(npy_file_names), ord_dir=ord_dir, dst_dir=dst_dir, npy_file_names=npy_file_names)
    else: multi_processing_interface(core_amount=core_amount, task_amount=len(npy_file_names), task=_save_npy_dir_as_knpy, task_args=[ord_dir, dst_dir, npy_file_names])
    # for npy_file_name in tqdm(npy_file_names):
    #     file_name = npy_file_name.split(".")[0]                 ### 把 "檔案名". "npy" 分開，只抓檔案名等等才好存 ".knpy"
    #     with open(ord_dir + "/" + npy_file_name, "rb") as fr:   ### 把 .npy 用 open 且 read byte 的形式打開
    #         byte_strs = []                                      ### fr不能直接用，要用iter的方式才能讀內容
    #         for byte_str in fr:                                 ### 所以丟進去 for 把所有 byte_str抓出來囉！
    #             byte_strs.append(byte_str)

    #         with open(dst_dir + "/" + file_name + ".knpy", "wb") as fw:  ### 經過觀察，只要去掉第一個byte_str就可以去掉numpy的標頭檔資訊拉
    #             for byte_str in byte_strs[1:]:                           ### 所以從 byte_strs的第二個元素開始把 byte_str寫進新檔案，且命名為 ".knpy"，kong_numpy的概念ˊ口ˋ
    #                 fw.write(byte_str)
##############################################################################################################################################################
##############################################################################################################################################################

def _use_ltrd_crop(start_index, amount, ord_dir, dst_dir, file_names, l, t, r, d, crop_according_lr_page, odd_x_shift, even_x_shift):
    for go_file_name, file_name in enumerate(tqdm(file_names[start_index:start_index + amount])):
        ord_img = cv2.imread(ord_dir + "/" + file_name)
        crop_img = ord_img.copy()
        if(crop_according_lr_page is False): crop_img = ord_img[t:d, l:r, ...]  ### 正常crop
        else:  ### 如果要分 左右page不同的shift
            if  ((go_file_name + 1) % 2 == 0): crop_img = ord_img[t:d, l + even_x_shift : r + even_x_shift , ...]  ### 偶數頁，注意我的page_num是從1開始喔~~ 有+1才對的到
            elif((go_file_name + 1) % 2 == 1): crop_img = ord_img[t:d, l + odd_x_shift  : r + odd_x_shift  , ...]  ### 基數頁，注意我的page_num是從1開始喔~~ 有+1才對的到

        # cv2.imshow("crop_img", crop_img)
        # cv2.waitKey()
        cv2.imwrite(dst_dir + "/" + file_name, crop_img)

def _use_ltrd_crop_multiprocess(ord_dir, dst_dir, file_names, l, t, r, d, crop_according_lr_page, odd_x_shift, even_x_shift, core_amount=8, task_amount=100):
    from kong_util.multiprocess_util import multi_processing_interface
    multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=_use_ltrd_crop, task_args= [ord_dir, dst_dir, file_names, l, t, r, d, crop_according_lr_page, odd_x_shift, even_x_shift] )


def Find_ltrd_and_crop(ord_dir, dst_dir, padding=50, search_amount=-1, crop_according_lr_page=False, odd_x_shift=0, even_x_shift=0, multiprocess=True, core_amount=8):
    print("doing Find_ltrd_and_crop")
    if(crop_according_lr_page is True and multiprocess is True):  ### 防呆
        multiprocess = False
        print("因為 crop_according_lr_page 模式有開啟，無法使用multiprocess，要不然 左、右 頁的 index 可能會讀錯，自動把multiprocess關掉囉~")

    start_time = time.time()
    ### 建立放結果的資料夾
    Check_dir_exist_and_build(dst_dir)

    l, t, r, d = Find_db_left_top_right_down(ord_dir, padding=padding, search_amount=search_amount)
    file_names = get_dir_img_file_names(ord_dir)
    if(multiprocess and core_amount > 1):
        _use_ltrd_crop_multiprocess(ord_dir, dst_dir, file_names, l, t, r, d, crop_according_lr_page, odd_x_shift, even_x_shift, core_amount=core_amount, task_amount=len(file_names))
    else:
        _use_ltrd_crop(0, len(file_names), ord_dir, dst_dir, file_names, l, t, r, d, crop_according_lr_page, odd_x_shift, even_x_shift)
    print("Find_ltrd_and_crop cost time:", time.time() - start_time)




def Pad_lrtd_and_resize_same_size(ord_dir, dst_dir, l, r, t, d):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    print("ord_dir", ord_dir)
    file_names = [file_name for file_name in os.listdir(ord_dir) if Check_img_filename(file_name)]
    print("file_names", file_names)
    for file_name in file_names:
        img = cv2.imread(ord_dir + "/" + file_name)       ### 把圖讀出來
        ord_h, ord_w = img.shape[:2]
        pad_img = np.pad(img, ( (t, d), (l, r), (0, 0) ), "constant")
        pad_resize_img = cv2.resize(pad_img, (ord_w, ord_h), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(dst_dir + "/" + file_name, pad_resize_img)
        print("Pad and Resize:", dst_dir + "/" + file_name, "finished!!")
##############################################################################################################################################################
##############################################################################################################################################################


def Convert_jpg_to_png(ord_dir, dst_dir="jpg_to_png", print_msg=False):
    import cv2
    Check_dir_exist_and_build(dst_dir)

    file_names = get_dir_jpg_names(ord_dir)  ### 取得dir 內的 .png
    for file_name in file_names:
        ord_jpg_path = f"{ord_dir}/{file_name}"           ### 定位出 來源jpg 的 path
        dst_png_path = f"{dst_dir}/{file_name[:-4]}.png"  ### 定位出 目標png 的 path
        img = cv2.imread(ord_jpg_path)
        cv2.imwrite(dst_png_path, img)
        if(print_msg): print("ord_jpg_path", ord_jpg_path, "->", dst_png_path, "finish")


if(__name__ == "__main__"):
    ### 從這開始改
    same_crop_window_size_w = 674 * 3  ### 這應該不會再動到了，從 3000*4000 原始影像 crop的 width
    same_crop_window_size_h = 674 * 4  ### 這應該不會再動到了，從 3000*4000 原始影像 crop的 height

    same_resize_w = 408  # 512 #408 #3*266 +2 # 800 # 177 * 3 + 1 #532 #332 # 336 # 348 # 330
    same_resize_h = 544  # 512 #544 #4*266 +4 #1068 # 177 * 4     #708 #440 # 448 # 464 # 440


    same_crop_row_num = 4  # 1 # 2
    same_crop_row_width  = 96  # same_resize_w
    same_crop_row_height = 96  # same_resize_h #512 #355 +1

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### curve
    curve_left = 480
    curve_top  = 300
    curve_crop_window_size_w = same_crop_window_size_w
    curve_crop_window_size_h = same_crop_window_size_h

    curve_resize_width  = same_resize_w
    curve_resize_height = same_resize_h

    curve_crop_num = same_crop_row_num
    ##############################################################################
    curve_dir = "A camera_ord_img"
    step_1_dir = curve_dir + "/" + "1 delete_the_page_not_in_pdf"
    step_2_dir = curve_dir + "/" + "2 page_num_ok"
    step_3_dir = curve_dir + "/" + "3 crop_ok"
    step_4_dir = curve_dir + "/" + "4 resize_ok"
    step_5_dir = curve_dir + "/" + "5 crop_row_ok"
    step_6_dir = curve_dir + "/" + "6 datasetA_ok"

    Page_num(ord_dir = step_1_dir, dst_dir = step_2_dir)
    Crop(ord_dir = step_2_dir, dst_dir = step_3_dir,     left = curve_left, top = curve_top,     crop_window_size_w = curve_crop_window_size_w, crop_window_size_h=curve_crop_window_size_h)
    Resize_hw(ord_dir = step_3_dir, dst_dir = step_4_dir,     width = curve_resize_width, height = curve_resize_height)
    Crop_row_random(ord_dir = step_4_dir, dst_dir = step_5_dir, seed = 10, crop_num = same_crop_row_num,
                    image_range_width = curve_resize_width, image_range_height = curve_resize_height ,
                    base_left = 0, base_top = 0,
                    crop_window_size_h = same_crop_row_height , crop_window_size_w = same_crop_row_width,
                    name = "")


    ## 想看 檔案數量
    file_num = len(  [file_name for file_name in os.listdir(step_5_dir) if ".jpg" in file_name.lower()]   )
    test_file_num  = 20  #int(file_num * 0.2)
    train_file_num = file_num - test_file_num

    Split_train_test(ord_dir=step_5_dir, dst_dir=step_6_dir,     train_dir_name="trainA", test_dir_name="testA",    train_amount=train_file_num, test_amount=test_file_num)
    ###############################################################################
    ###############################################################################
    ###############################################################################



    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### straight
    straight_left = 250
    straight_top  = 100  #250
    straight_crop_window_size_w = same_crop_window_size_w
    straight_crop_window_size_h = same_crop_window_size_h

    straight_resize_width  = same_resize_w
    straight_resize_height = same_resize_h

    straight_crop_num = same_crop_row_num
    ###############################################################################
    straight_dir = "B paper_to_jpg"
    step_1_dir = straight_dir + "/" + "1 pdf_to_jpg"
    step_2_dir = straight_dir + "/" + "2 page_num_ok"
    step_3_dir = straight_dir + "/" + "3 crop_ok"
    step_4_dir = straight_dir + "/" + "4 resize_ok"
    step_5_dir = straight_dir + "/" + "5 crop_row_ok"
    step_6_dir = straight_dir + "/" + "6 datasetB_ok"

    Page_num(ord_dir = step_1_dir, dst_dir = step_2_dir)

    Crop(ord_dir = step_2_dir, dst_dir = step_3_dir,
        left = straight_left, top = straight_top,
        crop_window_size_w = straight_crop_window_size_w, crop_window_size_h = straight_crop_window_size_h)

    Resize_hw(ord_dir = step_3_dir, dst_dir = step_4_dir,
        width = straight_resize_width, height = straight_resize_height)

    Crop_row_random(ord_dir = step_4_dir, dst_dir = step_5_dir, seed = 10, crop_num = same_crop_row_num,
                    image_range_width = straight_resize_width, image_range_height = straight_resize_height ,
                    base_left = 0, base_top = 0,
                    crop_window_size_h = same_crop_row_height , crop_window_size_w = same_crop_row_width,
                    name = "")

    ### 想看 檔案數量
    file_num = len(  [file_name for file_name in os.listdir(step_5_dir) if ".jpg" in file_name.lower()]   )

    test_file_num  = 20  #int(file_num * 0.2)
    train_file_num = file_num - test_file_num

    Split_train_test(ord_dir = step_5_dir, dst_dir = step_6_dir,
                    train_dir_name = "trainB", test_dir_name = "testB",
                    train_amount = train_file_num, test_amount = test_file_num)
    ###############################################################################
    ###############################################################################
    ###############################################################################
