import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def imread_chinese(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    # img = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
    return img

def imwrite_chinese(img_path:str, img):
    cv2.imencode("." + img_path.split(".")[-1], img)[1].tofile(img_path)

def add_water_mark(img_path, logo_path, result_path, result_ratio=None, u_crop=None, d_crop=None, l_crop=None, r_crop=None):
    '''
    u, d, l, r_crop : 上浮水印前是否要先對原影像做crop
    restul_ratio    : 上浮水印完成的結果圖縮小比例
    '''
    # img_path = "scan_0002.jpg"
    # logo_path = "TKUWB_logo.jpg"
    img        = imread_chinese(img_path)
    ''' 附加 crop功能, 注意順序 很重要 d_crop先於u_crop, r_crop先於l_crop '''
    if(d_crop is not None): img = img[      :d_crop,       :      , :]
    if(r_crop is not None): img = img[      :      ,       :r_crop, :]
    if(u_crop is not None): img = img[u_crop:      ,       :      , :]
    if(l_crop is not None): img = img[      :      , l_crop:      , :]

    ''' crop後 才抓 img.shape 才合理喔 '''
    img_h, img_w, img_c = img.shape

    ''' 開始 上浮水印 '''
    ### 讀取影像
    water_mark = imread_chinese(logo_path)

    ### logo縮縫成 img 的短邊 的 85%
    water_mark_size = int(min(img_h, img_w) * 0.85)  ### 取 img 的 短邊 然後縮小些
    water_mark = cv2.resize(water_mark, (water_mark_size, water_mark_size))
    water_mark_h, water_mark_w, _ = water_mark.shape

    ### logo定位出 t, d, l, r
    t = img_h // 2 - water_mark_h // 2
    d = t + water_mark_h
    l = img_w // 2 - water_mark_w // 2
    r = l + water_mark_w

    ### 圖片前處理 先弄成 float32
    img = img.astype(np.float32)
    water_mark = water_mark.astype(np.float32)
    ### 浮水印 黑變白 來 和 img 相減, 後面 * 的東西 是浮水印的強度
    img[t:d, l:r] = img[t:d, l:r] - (255 - water_mark) * 0.18
    img = img.clip(0, 255)
    img = img.astype(np.uint8)

    ''' 附加的 上完浮水印的圖 是否要縮小 '''
    if(result_ratio is not None): img = cv2.resize(img, (int(img_w * result_ratio), int(img_h * result_ratio)))

    ### 完成, 寫入檔案
    imwrite_chinese(result_path, img)
    print(f"{result_path} water_marked ok")
    # plt.imshow(img)
    # plt.show()

def Dir_Add_WaterMark(src_dir, dst_dir=None, logo_path=None, result_ratio=None, u_crop=None, d_crop=None, l_crop=None, r_crop=None):
    '''
    src_dir         : 來源地的dir位置
    dst_dir         : 目的地的dir位置
    logo_path       : logo的位置
    u, d, l, r_crop : 上浮水印前是否要先對原影像做crop
    restul_ratio    : 上浮水印完成的結果圖縮小比例
    '''

    ''' 如果 沒有指定 dst_dir, 指定 在src_dir內建立 water_marded資料夾 當 dst_dir '''
    if(dst_dir is None):
        dst_dir = f"{src_dir}/water_marked"
        if(result_ratio is not None):
            dst_dir += "_small"
        os.makedirs(dst_dir, exist_ok=True)

    ''' 如果沒有指定 logo_path, 嘗試從 kong_util 裡面找 TKUWB_logo '''
    if(logo_path is None):
        code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
        code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
        code_dir = "\\".join(code_exe_path_element[:-1])
        logo_path = f"{code_dir}\\TKUWB_logo.jpg"

    ''' 從 src_dir 讀出 待處理的.jpg 的 path '''
    file_names = [file_name for file_name in os.listdir(src_dir) if ".jpg" in file_name.lower() ]
    file_paths = [f"{src_dir}/{file_name}" for file_name in file_names]

    ''' 定位出 每個 src_path 的 dst_path, 並 把 每個 src_path 的影像 加上浮水印 存入 dst_path'''
    for go_f, file_name in enumerate(file_names):
        result_path = f"{dst_dir}/{file_name}"
        add_water_mark(img_path = file_paths[go_f],
                       logo_path = logo_path,
                       result_ratio=result_ratio,
                       result_path = result_path,
                       u_crop=u_crop,
                       d_crop=d_crop,
                       l_crop=l_crop,
                       r_crop=r_crop
                       
                       )

if(__name__ == "__main__"):
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


    result_ratio = 1.00
    src_dir = r"C:\Users\VAIO\Desktop\嘗試中文"
    Dir_Add_WaterMark(src_dir)
