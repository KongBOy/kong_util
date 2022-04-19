from matplotlib import projections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import pdb

from matplot_fig_ax_util import check_fig_ax_init

debug_dict = {}

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
    fig, ax = plt.subplots(1, 1, figsize=figsize)
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

def WM_3d_plot( WM,
                fewer_point=True,
                savefig=False, save_path=".",
                xmin=-0.08075158,  xmax=0.07755918,
                ymin=-0.13532962, ymax=0.1357405,
                zmin=0.0, zmax=0.039187048,
                fig=None, ax=None, ax_r=0, ax_c=None, ax_rows=1, ax_size=6):
    '''
    wc  ： float32, h, w, 3
    mask： bool,    h, w   ，可以從 uv ch0 來得到，再用 np.astype 轉成 np.bool 即可！
    fewer_point：因為原始 448*448 plot出來會很lag，如果縮小一些plot的話會比較好一些
    '''
    debug_dict["WM"] = WM
    h, w, c = WM.shape
    if(fewer_point):
        WM  = cv2.resize(WM, (w // 2, h // 2))  ### cv2.resize 是 先x 在y
        debug_dict["WM_resize"] = WM

    Wz = WM[..., 0]
    Wy = WM[..., 1]
    Wx = WM[..., 2]
    if(c == 4):
        M  = WM[..., 3]
        M  = np.where(M > 0.99, 1, 0)
    else:
        M = np.where(Wz > 0, 1, 0)

    M  = M.astype(np.bool)

    ### 3D 直接畫出來囉～～
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=1, ax_size=ax_size, tight_layout=True)
    # fig.set_size_inches((ax_size + 0.5, ax_size))  ### x + 50 pixel 讓z軸的自可以完整顯示
    ax[ax_c].remove()  ### 因為 是 3D子圖 要和 2D子圖 放同張figure， 所以要 ax2d.remove() 把原本的 2D子圖 隱藏起來(.remove())
    ax_cols = len(ax)  ### 定位出 ax3d 要加入 fig 的哪個的位置， 需要知道目前的 fig_rows/cols
    ax3d = fig.add_subplot(ax_rows, ax_cols, (ax_r * ax_cols) + ax_c + 1, projection="3d")
    ax[ax_c] = ax3d  ### 把 ax3d 取代 原本的 ax

    ax[ax_c] .set_xlabel("x")
    ax[ax_c] .set_ylabel("y")
    ax[ax_c] .set_zlabel("z")  ### 設定 x,y,z軸顯示的字

    ax[ax_c] .set_xlim(xmin, xmax)  ### 設定 x範圍
    ax[ax_c] .set_ylim(ymin, ymax)  ### 設定 y範圍
    ax[ax_c] .set_zlim(zmin, zmax)  ### 設定 z範圍

    ax[ax_c] .scatter(Wx[ M ], Wy[ M ], Wz[ M ],  ### 直接畫出來
                s=1,                         ### 點點的 大小
                c=np.arange( M.sum()),       ### 彩色
    )
    if(savefig):
        plt.savefig(save_path)
        # plt.show()
        plt.close()
    return fig, ax, ax_c


if(__name__ == "__main__"):
    import os
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import time
    from tqdm import tqdm
    import os
    # WM_dir = r"J:\kong_render_os_book_and_paper_all_have_dtd_hdr_mix_bg_512\2_wc_w_M_npy"
    WM_dir = r"C:\Users\CVML\Desktop\see001_manually\kong_render_os_book_and_paper_all_have_dtd_hdr_mix_bg_512\2_wc_w_M_npy"
    WM_file_names = os.listdir(WM_dir)

    # dst_dir = r"J:\kong_render_os_book_and_paper_all_have_dtd_hdr_mix_bg_512\2_wc_w_M_npy_3D_visual"

    dst_dir = r"C:\Users\CVML\Desktop\see001_manually\kong_render_os_book_and_paper_all_have_dtd_hdr_mix_bg_512\2_wc_w_M_npy_3D_visual"
    os.makedirs(dst_dir, exist_ok=True)
    for go_f, _ in enumerate(tqdm(os.listdir(WM_dir))):
        start_time = time.time()
        WM_path = f"{WM_dir}/{WM_file_names[go_f]}"
        WM = np.load(WM_path)
        WM_3d_plot(WM,
                   fewer_point=True,
                   savefig=True, save_path=f"{dst_dir}/%04i" % go_f,
                   xmin=-0.08075158,  xmax=0.07755918, ymin=-0.13532962, ymax=0.1357405, zmin=0.0, zmax=0.039187048)
        print("cost time:", time.time() - start_time)
        # break
