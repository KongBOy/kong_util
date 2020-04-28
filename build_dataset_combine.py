import os
import shutil
import cv2
import numpy as np
import random


def Check_img_filename(file_name):
    if(".jpg" in file_name.lower() or "jpeg" in file_name.lower() or ".png" in file_name.lower() or ".bmp" in file_name.lower()):
        return True
    else:
        return False


### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
def Check_dir_exist_and_build(dir_name):
    if(os.path.isdir( dir_name)): ### 如果有上次建立的結果要先刪掉
        print(dir_name,"已存在，不建立新資料夾")
    else:
        os.makedirs( dir_name, exist_ok=True)

### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
def Check_dir_exist_and_build_new_dir(dir_name):
    if(os.path.isdir( dir_name)): ### 如果有上次建立的結果要先刪掉
        print(dir_name,"已存在，刪除已存在的資料夾，並建立新的資料夾")
        shutil.rmtree( dir_name)
    os.makedirs( dir_name, exist_ok=True)

### 把圖片重新命名成 流水號
def Page_num(ord_dir, dst_dir):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = os.listdir(ord_dir)
    # print(file_names)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]
    
    file_names.sort()
    
    for i,file_name in enumerate(file_names):
        shutil.copy( ord_dir+"/"+ file_name, dst_dir+"/"+ "%06i.jpg"%(i+1) )
        # print( ord_dir+"/"+ file_name,"copy to", ord_dir+"/"+ "%06i.jpg"%(i+1), "finished!" )
    print(dst_dir,"page_num finish")
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
    ord_img = cv2.imread(ord_dir + "/"+ file_names[0])
    height, width, channel = ord_img.shape

    ### 開始crop囉
    for j,file_name in enumerate(file_names[:]):
        ord_img = cv2.imread(ord_dir + "/"+ file_name)

        crop = ord_img[top:top+crop_window_size_h, left:left+crop_window_size_w] ### 從windows的左上角 框一個 windows的影像出來
        #cv2.imshow("test", crop)
        #cv2.waitKey(0)

        result_file_name = dst_dir + "/" + "%s%s-left=%04i-top=%04i.jpg"%(name,file_name[:-4], left, top) 
        cv2.imwrite(result_file_name,crop)
        print(result_file_name, "finished!!")

def Crop_use_center(ord_dir = "", 
         dst_dir = "", 
         center_x_list  = None,
         center_y_list  = None,
         crop_window_size_w = 674 * 3,
         crop_window_size_h = 674 * 4,
         seed = 10,
         lt_s_y =   0, ### left_top_shift_y
         lt_s_x =   0, ### left_top_shift_x
         lt_a_h =   0, ### left_top_add_h
         rt_s_y =   0, ### right_top_shift_y
         rt_s_x =   0, ### right_top_shift_x
         rt_a_h =   0, ### right_top_add_h
         ld_s_y =   0, ### left_down_shift_y
         ld_s_x =   0, ### left_down_shift_y
         ld_a_h =   0, ### left_down_add_height
         rd_s_y =   0, ### right_down_shift_y
         rd_s_x =   0, ### right_down_shift_x
         rd_a_h =   0  ### right_down_add_height):
         ):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    ### .jpg的檔名抓出來
    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]

    ### 抓取影像長寬資訊
    ord_img = cv2.imread(ord_dir + "/"+ file_names[0])
    height, width, channel = ord_img.shape

    ### 開始crop囉
    for j,file_name in enumerate(file_names[:]):
        ord_img = cv2.imread(ord_dir + "/"+ file_name)
        if(center_x_list is None and center_y_list is None):
            center_x = int(width/2)
            center_y = int(height/2)
            crop_window_size_w = int(width/2)
            crop_window_size_h = int(height/2)
        else:
            center_x = center_x_list[j]
            center_y = center_y_list[j]
        ### add_height 或 add_width 的部分 可以自己看結果，往左上(用-) 或 右下(用+) 補都可以，
        crop_left_top   = ord_img[center_y-crop_window_size_h + lt_s_y - lt_a_h*1 : center_y + lt_s_y ,  center_x-crop_window_size_w + lt_s_x - int(lt_a_h*0.6847) : center_x + lt_s_x] ### 從windows的左上角 框一個 windows的影像出來
        crop_right_top  = ord_img[center_y-crop_window_size_h + rt_s_y - rt_a_h*1 : center_y + rt_s_y ,  center_x + rt_s_x - int(rt_a_h*0.6487) : center_x+crop_window_size_w + rt_s_x] ### 從windows的左上角 框一個 windows的影像出來
        crop_left_down  = ord_img[center_y + ld_s_y : center_y+crop_window_size_h + ld_s_y + ld_a_h*1 ,  center_x-crop_window_size_w + ld_s_x : center_x + ld_s_x + int(ld_a_h*0.6487)] ### 從windows的左上角 框一個 windows的影像出來
        crop_right_down = ord_img[center_y + rd_s_y : center_y+crop_window_size_h + rd_s_y + rd_a_h*1 ,  center_x + rd_s_x : center_x+crop_window_size_w + rd_s_x + int(rd_a_h*0.6487)] ### 從windows的左上角 框一個 windows的影像出來
        # print("center_y-crop_window_size_h + lt_s_y - lt_a_h*1",center_y-crop_window_size_h + lt_s_y - lt_a_h*1) ### 小心有時會變負的
        print("center_y-crop_window_size_h + rt_s_y ",center_y-crop_window_size_h + rt_s_y - rt_a_h*1 ) ### 小心有時會變負的
        
        #cv2.imshow("test", crop)
        #cv2.waitKey(0)

        left_top_file_name   = dst_dir + "/" + "1-%s-%s-center_x=%04i-center_y=%04i.jpg"%("left_top",   file_name[:-4],center_x, center_y) 
        right_top_file_name  = dst_dir + "/" + "2-%s-%s-center_x=%04i-center_y=%04i.jpg"%("right_top",  file_name[:-4],center_x, center_y) 
        left_down_file_name  = dst_dir + "/" + "3-%s-%s-center_x=%04i-center_y=%04i.jpg"%("left_down",  file_name[:-4],center_x, center_y) 
        right_down_file_name = dst_dir + "/" + "4-%s-%s-center_x=%04i-center_y=%04i.jpg"%("right_down", file_name[:-4],center_x, center_y) 
        cv2.imwrite(left_top_file_name  , crop_left_top)
        cv2.imwrite(right_top_file_name , crop_right_top)
        cv2.imwrite(left_down_file_name , crop_left_down)
        cv2.imwrite(right_down_file_name, crop_right_down)
        # print(left_top_file_name, "finished!!")


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
        if(method=="cv2"):
            resize = cv2.resize(ord_img, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            import scipy.misc
            resize = scipy.misc.imresize(ord_img, [height,width])

        #cv2.imshow("resize",resize)
        #cv2.waitKey(0)
        cv2.imwrite(dst_dir + "/" + file_name, resize)
        print("Resize:", dst_dir + "/" + file_name,"finished!")

def Crop_row_random(ord_dir = "",dst_dir = "",seed = 10,crop_num = 4,
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
    ord_img = cv2.imread(ord_dir + "/"+ file_names[0])
    height, width, channel = ord_img.shape
    # print("ord_img.shape",ord_img.shape)

    max_left = image_range_width  - crop_window_size_w  ### window 左座標 最大範圍
    max_top  = image_range_height - crop_window_size_h  ### window 下座標 最大範圍

    ### 開始crop囉
    for go_file,file_name in enumerate(file_names[:]):
        ord_img = cv2.imread(ord_dir + "/"+ file_name)
        for go_crop in range(1,1+crop_num):
            left = random.randint(base_left,max_left) ### 取 左~左最大範圍 間的 隨機一個點當 windows的左
            top  = random.randint(base_top ,max_top ) ### 取 上~上最大範圍 間的 隨機一個點當 windows的上 
            crop_row = ord_img[top:top+crop_window_size_h, left:left+crop_window_size_w] ### 從windows的左上角 框一個 windows的影像出來
            
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

            result_file_name =  dst_dir + "/" + "%s%s-crop_row%i-left=%04i-top=%04i.jpg"%(name,file_name[:-4], go_crop, left, top) 
            cv2.imwrite(result_file_name,crop_row)
            print(result_file_name, "finished!!")


def Split_train_test(ord_dir,dst_dir,train_dir_name = "", test_dir_name = "", 
                     train_amount = 70,test_amount = 13):
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

    for i,file_name in enumerate(file_names):
        # if( i in ignore_list):
        #     continue
        if( i < train_amount):
            shutil.copy( ord_dir+"/"+ file_name, dst_dir + "/" + train_dir  + "/" +  file_name )
            print( ord_dir+"/"+ file_name, dst_dir + "/" + train_dir  + "/" +  file_name, "finished!" )
        else:
            shutil.copy( ord_dir+"/"+ file_name, dst_dir + "/" + test_dir  + "/" +  file_name )
            print( ord_dir+"/"+ file_name, dst_dir + "/" + test_dir  + "/" +  file_name, "finished!" )

##############################################################################################################################################################
##############################################################################################################################################################

def fine_db_left_top_right_down(ord_dir,padding = 0):### padding是 印表機 印出來旁邊自動padding的空白，要自己去嘗試為多少喔！
    padding = int(padding)

    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]
    file_names.sort()
    
    ### 抓取影像長寬資訊
    ord_img = cv2.imread(ord_dir + "/"+ file_names[0])
    height, width, channel = ord_img.shape

    lefts  = []
    tops   = []
    rights = []
    downs  = []
    for file_name in file_names[0:]:
        ord_img = cv2.imread(ord_dir + "/" + file_name,0)
        ret,thresh1 = cv2.threshold(ord_img,127,255,cv2.THRESH_BINARY_INV) ### 二值化影像
        # cv2.imshow("thresh1",thresh1)
        # cv2.waitKey(0)

        width_sum = thresh1.sum(axis=0) ### shape 為 (2481,)   ### 統計 垂直 值條圖的概念，找出左右最大的range
        lefts.append( (width_sum!=0).argmax() ) ### left
        rights.append( width - (width_sum!=0)[::-1].argmax() ) ### right
        
        height_sum = thresh1.sum(axis=1) ### shape 為 (3508,)  ### 統計 水平 值條圖的概念，找出上下最大的range
        tops.append( (height_sum!=0).argmax() ) ### top
        downs.append( height - (height_sum!=0)[::-1].argmax() ) ### down

        #print(lefts[-1],rights[-1],tops[-1], downs[-1])

    left = min(lefts) - padding
    top  = min(tops)  - padding
    right = max(rights) + padding
    down = max(downs)   + padding
    print("left",left,"top",top,"right",right,"down",down)

    return left,top,right,down

        
def Center_draw_photo_frame(ord_dir, dst_dir,thick_v = 200,thick_h=100,color = (0,0,0)):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]
    file_names.sort()
    
    ### 抓取影像長寬資訊
    ord_img = cv2.imread(ord_dir + "/"+ file_names[0])
    height, width, channel = ord_img.shape

    white = (255,255,255)
    black = (  0,  0,  0)
    # color = white
    # thick = 200

    for file_name in file_names[0:]:
        ord_img = cv2.imread(ord_dir + "/" + file_name)
        # canvas = np.zeros((300, 300, 3), dtype="uint8") #3
        # cv2.line( img = canvas, pt1 = (0, 0), pt2 = (300, 300), color = color,thickness = 20) ###不能用畫線的，因為thickness大到一定程度就大不上去了

        proc_img = ord_img.copy()
        cv2.rectangle(proc_img, (int(width/2-thick_v/2) ,0), (int(width/2+thick_v/2),height), color,-1 ) ### 畫 vertical
        cv2.rectangle(proc_img, (0,int(height/2-thick_h/2)), (width,int(height/2+thick_h/2)), color,-1 ) ### 畫 horizontal
        cv2.imwrite(dst_dir + "/" + file_name, proc_img)

        # cv2.imshow("proc_img",proc_img)
        # cv2.waitKey(0)
        print(dst_dir + "/" + file_name, "finished!!")

def Photo_frame_padding(ord_dir, dst_dir, left_pad=140, top_pad=50, right_pad=140, down_pad=50, color=(255,255,255)):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = os.listdir(ord_dir)
    file_names = [file_name for file_name in file_names if Check_img_filename(file_name)]
    file_names.sort()
    
    ### 抓取影像長寬資訊
    ord_img = cv2.imread(ord_dir + "/"+ file_names[0])
    height, width, channel = ord_img.shape

    pad_height = height + top_pad + down_pad
    pad_width  = width  + left_pad + right_pad

    color = np.array(color,dtype = np.uint8)
    canvas = np.tile(color,pad_height*pad_width).reshape(pad_height,pad_width,3)
    for file_name in file_names[:]:
        ord_img = cv2.imread(ord_dir + "/" + file_name)
        canvas[top_pad:top_pad+height, left_pad:left_pad+width] = ord_img
        cv2.imwrite(dst_dir + "/" + file_name, canvas)
        
        # cv2.imshow("canvas",canvas)
        # cv2.waitKey(0)
        print(dst_dir + "/" + file_name, "finished!!")


def Save_as_gray(ord_dir, dst_dir, gray_three_channel=True):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = [file_name for file_name in os.listdir(ord_dir) if Check_img_filename(file_name)]
    for file_name in file_names:
        gray_img = cv2.imread(ord_dir + "/" + file_name, 0)
        if(gray_three_channel):
            gray_img = gray_img[:,:,np.newaxis]
            gray_img = np.tile(gray_img, (1,1,3))
        cv2.imwrite(dst_dir + "/" + file_name, gray_img)
    

def Save_as_bmp(ord_dir, dst_dir, gray=False, gray_three_channel=False):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    file_names = [file_name for file_name in os.listdir(ord_dir) if Check_img_filename(file_name)]
    for file_name in file_names:
        file_title, file_ext = file_name.split(".") ### 把 檔名前半 後半 分開
        if(file_ext != "bmp"):                ### 如果附檔名不是bmp，把圖讀出來，存成bmp
            import cv2 
            img = cv2.imread(ord_dir + "/" + file_name)       ### 把圖讀出來
            if  ( gray ): 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  ### 如果要轉灰階，就轉灰階
                if(gray_three_channel):
                    img = img[:,:,np.newaxis]
                    img = np.tile(img, (1,1,3))
                
            print(img.shape)

            cv2.imwrite(dst_dir + "/" + file_title+".bmp", img) ### 存成bmp
            print("Save_as_bmp", ord_dir + "/" + file_name, "save as", dst_dir + "/" + file_title+".bmp", "finish~~")



def Pad_lrtd_and_resize_same_size(ord_dir, dst_dir,l,r,t,d):
    ### 建立放結果的資料夾，如果有上次建立的結果要先刪掉
    Check_dir_exist_and_build(dst_dir)

    print("ord_dir",ord_dir)
    file_names = [file_name for file_name in os.listdir(ord_dir) if Check_img_filename(file_name)]
    print("file_names",file_names)
    for file_name in file_names:
        img = cv2.imread(ord_dir + "/" + file_name)       ### 把圖讀出來
        ord_h, ord_w = img.shape[:2]
        pad_img = np.pad(img, ( (t,d), (l,r), (0,0) ), "constant")
        pad_resize_img = cv2.resize(pad_img, (ord_w, ord_h), interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(dst_dir + "/" + file_name, pad_resize_img)       
        print("Pad and Resize:", dst_dir + "/" + file_name, "finished!!")
##############################################################################################################################################################
##############################################################################################################################################################

if(__name__ == "__main__"):
    ### 從這開始改
    same_crop_window_size_w = 674 * 3 ### 這應該不會再動到了，從 3000*4000 原始影像 crop的 width
    same_crop_window_size_h = 674 * 4 ### 這應該不會再動到了，從 3000*4000 原始影像 crop的 height

    same_resize_w = 408 # 512 #408 #3*266 +2 # 800 # 177 * 3 + 1 #532 #332 # 336 # 348 # 330 
    same_resize_h = 544 # 512 #544 #4*266 +4 #1068 # 177 * 4     #708 #440 # 448 # 464 # 440


    same_crop_row_num = 4 # 1 # 2
    same_crop_row_width  = 96 # same_resize_w
    same_crop_row_height = 96 # same_resize_h #512 #355 +1

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

    Page_num(ord_dir = step_1_dir ,dst_dir = step_2_dir)
    Crop(ord_dir = step_2_dir, dst_dir = step_3_dir,     left = curve_left, top = curve_top,     crop_window_size_w = curve_crop_window_size_w, crop_window_size_h=curve_crop_window_size_h)
    Resize(ord_dir = step_3_dir, dst_dir = step_4_dir,     width = curve_resize_width, height = curve_resize_height)
    Crop_row_random(ord_dir = step_4_dir, dst_dir = step_5_dir, seed = 10, crop_num = same_crop_row_num,
                    image_range_width = curve_resize_width, image_range_height = curve_resize_height ,
                    base_left = 0, base_top = 0,
                    crop_window_size_h = same_crop_row_height , crop_window_size_w = same_crop_row_width,
                    name = "")


    ## 想看 檔案數量
    file_num = len(  [file_name for file_name in os.listdir(step_5_dir) if ".jpg" in file_name.lower()]   )
    test_file_num  = 20 #int(file_num * 0.2) 
    train_file_num = file_num - test_file_num

    Split_train_test(ord_dir = step_5_dir, dst_dir = step_6_dir,     train_dir_name = "trainA", test_dir_name = "testA",    train_amount = train_file_num,test_amount = test_file_num)
    ###############################################################################
    ###############################################################################
    ###############################################################################



    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### straight
    straight_left = 250
    straight_top  = 100 #250
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

    Page_num(ord_dir = step_1_dir ,dst_dir = step_2_dir)

    Crop(ord_dir = step_2_dir, dst_dir = step_3_dir, 
        left = straight_left, top = straight_top, 
        crop_window_size_w = straight_crop_window_size_w, crop_window_size_h = straight_crop_window_size_h)

    Resize(ord_dir = step_3_dir, dst_dir = step_4_dir, 
        width = straight_resize_width, height = straight_resize_height)

    Crop_row_random(ord_dir = step_4_dir, dst_dir = step_5_dir, seed = 10, crop_num = same_crop_row_num,
                    image_range_width = straight_resize_width, image_range_height = straight_resize_height ,
                    base_left = 0, base_top = 0,
                    crop_window_size_h = same_crop_row_height , crop_window_size_w = same_crop_row_width,
                    name = "")

    ### 想看 檔案數量
    file_num = len(  [file_name for file_name in os.listdir(step_5_dir) if ".jpg" in file_name.lower()]   )

    test_file_num  = 20 #int(file_num * 0.2)
    train_file_num = file_num - test_file_num

    Split_train_test(ord_dir = step_5_dir, dst_dir = step_6_dir,
                    train_dir_name = "trainB", test_dir_name = "testB",
                    train_amount = train_file_num, test_amount = test_file_num)
    ###############################################################################
    ###############################################################################
    ###############################################################################