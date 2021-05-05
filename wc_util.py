import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

def uv_2d_plot(uv, figsize=(10, 10)):
    '''
    uv  ： float32, h, w, 3
    '''
    ### 2D 先把整張wc的值 整個弄到 0~1 之間 後 直接畫出來
    uv_2d_v = (uv - uv.min()) / (uv.max() - uv.min())

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(*figsize)
    ax.imshow(uv_2d_v)  ### 直接show 2D的長相
    return fig, ax

def wc_2d_plot(wc, figsize=(10, 10)):
    '''
    wc  ： float32, h, w, 3
    '''
    ### 2D 先把整張wc的值 整個弄到 0~1 之間 後 直接畫出來
    wc_2d_v = (wc - wc.min()) / (wc.max() - wc.min())

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(*figsize)
    ax.imshow(wc_2d_v)  ### 直接show 2D的長相
    return fig, ax


def wc_3d_plot(wc, mask, fewer_point=False, small_size=(200, 200), figsize=(10, 10)):
    '''
    wc  ： float32, h, w, 3
    mask： bool,    h, w   ，可以從 uv ch0 來得到，再用 np.astype 轉成 np.bool 即可！
    fewer_point：因為原始 448*448 plot出來會很lag，如果縮小一些plot的話會比較好一些
    '''
    if(mask.dtype != np.bool): mask = mask.astype(np.bool)  ### 防呆，怕沒轉成 np.bool

    if(fewer_point):
        mask = mask.astype(np.float32)  ### bool 不能resize，要轉乘
        wc = cv2.resize(wc, small_size)
        mask = cv2.resize(mask, small_size)
        mask = mask.astype(np.bool)

    wc_ch1 = wc[..., 0]
    wc_ch2 = wc[..., 1]
    wc_ch3 = wc[..., 2]



    ### 3D 直接畫出來囉～～
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(*figsize)
    ax = Axes3D(fig)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")  ### 設定 x,y,z軸顯示的字
    # plt.axis("off")     ### 純3D圖
    # ax.set_xticks([])   ### x軸無字
    # ax.set_yticks([])   ### y軸無字
    # ax.set_zticks([])   ### z軸無字
    # ax.set_zlim(-20, 30) ### 設定 z範圍

    # row, col = wc_ch3.shape[:2]  ### 隨便從ch1 或 ch2 或 ch3 抓 row, col 資訊， 好像沒用到
    ax.scatter(wc_ch1[ mask ], wc_ch2[ mask ], wc_ch3[ mask ],  ### 直接畫出來
                s=1,                        ### 點點的 大小
                # linewidths = 1,             ### 點點的 邊寬
                # edgecolors = "black"        ### 點點的 邊邊顏色
                c=np.arange( mask.sum()),   ### 彩色
    )
    # plt.show()
    return fig, ax
