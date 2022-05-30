from matplotlib import projections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import pdb

from kong_util.matplot_fig_ax_util import check_fig_ax_init, change_into_3D_coord_ax

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


def wc_3d_plot(wc, mask, fewer_point=False, small_size=(200, 200), ax_size=5, ch0_min=None, ch0_max=None, ch1_min=None, ch1_max=None, ch2_min=None, ch2_max=None):
    def _draw_wc_3d(ax3d, azim=-60, elev=30):
        ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")  ### 設定 x,y,z軸顯示的字
        ### 設定 x範圍， 丟None相當於不設定
        ax3d.set_xlim(ch0_min, ch0_max)
        ax3d.set_ylim(ch1_min, ch1_max)  ### 設定 y範圍， 丟None相當於不設定
        ax3d.set_zlim(ch2_min, ch2_max)  ### 設定 z範圍， 丟None相當於不設定

        ax3d.scatter(X_w_M, Y_w_M, Z_w_M,  ### 直接畫出來
                    s=1,                        ### 點點的 大小
                    # linewidths = 1,             ### 點點的 邊寬
                    # edgecolors = "black"        ### 點點的 邊邊顏色
                    c=np.arange( mask.sum()),   ### 彩色
        )

        ### 畫出 z top/bot 平面
        if(None not in [ch0_min, ch0_max, ch1_min, ch1_max, ch2_min, ch2_max]):
            xdata = np.linspace(ch0_min, ch0_max, 100)
            ydata = np.linspace(ch1_min, ch1_max, 100)
            zdata = np.linspace(ch2_min, ch2_max, 100)

            X, Y  = np.meshgrid(xdata, ydata)
            z_min_plane =  ch2_min * np.ones(shape=(100, 100))
            z_max_plane =  ch2_max * np.ones(shape=(100, 100))
            ax3d.plot_surface(X, Y, z_min_plane, alpha=0.5)
            ax3d.plot_surface(X, Y, z_max_plane, alpha=0.5)



            zdata = np.linspace(Z_w_M.min(), Z_w_M.max(), 100)
            X, Z  = np.meshgrid(xdata, zdata)
            y_0_plane =  np.zeros(shape=(100, 100))
            ax3d.plot_surface(X, y_0_plane, Z, alpha=0.5)

        ### 把 view 轉到 指定的位置
        ax3d.azim = azim
        ax3d.elev = elev

    ### 看一下 used_ch軸 的使用率
    def _see_used_and_residual_ratio(see_which_ch = "z"):
        if(see_which_ch.lower() == "z"):
            see_ch_w_M = Z_w_M
            see_ch_total_max = ch2_max
            see_ch_total_min = ch2_min
        elif(see_which_ch.lower() == "y"):
            see_ch_w_M = Y_w_M
            see_ch_total_max = ch1_max
            see_ch_total_min = ch1_min
        elif(see_which_ch.lower() == "x"):
            see_ch_w_M = X_w_M
            see_ch_total_max = ch0_max
            see_ch_total_min = ch0_min

        see_ch_w_M_min   = see_ch_w_M.min()
        see_ch_w_M_max   = see_ch_w_M.max()
        see_ch_w_M_range = see_ch_w_M_max - see_ch_w_M_min
        total_see_ch_range = see_ch_total_max - see_ch_total_min

        see_ch_used_ratio  = see_ch_w_M_range / total_see_ch_range
        see_ch_w_M_top_res_ratio = abs(see_ch_total_max - see_ch_w_M_max) / total_see_ch_range
        see_ch_w_M_bot_res_ratio = abs(see_ch_total_min - see_ch_w_M_min) / total_see_ch_range
        see_ch_w_M_top_bot_ratio_res = abs(see_ch_w_M_top_res_ratio - see_ch_w_M_bot_res_ratio)

        see_ch_used_ratio_str            = f"{see_which_ch.upper()}_used_ratio:%.2f" % see_ch_used_ratio
        see_ch_w_M_top_res_ratio_str     = f"{see_which_ch.upper()}_Top_res:%.2f" % see_ch_w_M_top_res_ratio
        see_ch_w_M_bot_res_ratio_str     = f"{see_which_ch.upper()}_Bot_res:%.2f" % see_ch_w_M_bot_res_ratio
        see_ch_w_M_top_bot_ratio_res_str = f"{see_which_ch.upper()}_T_B_res:%.2f" % see_ch_w_M_top_bot_ratio_res

        return see_ch_used_ratio_str, see_ch_w_M_top_res_ratio_str, see_ch_w_M_bot_res_ratio_str, see_ch_w_M_top_bot_ratio_res_str

    '''
    wc  ： float32, h, w, 3, 請自己調成 ch0:x, ch1:y, ch2:z 再輸入近來， 會顯示的最正確
    mask： bool,    h, w   ，可以從 uv ch0 來得到，再用 np.astype 轉成 np.bool 即可！
    fewer_point：因為原始 448*448 plot出來會很lag，如果縮小一些plot的話會比較好一些
    '''
    ### 以下處理 計算部分
    if(mask.dtype != np.bool): mask = mask.astype(np.bool)  ### 防呆，怕沒轉成 np.bool

    if(fewer_point):
        mask = mask.astype(np.float32)  ### bool 不能resize，要轉乘
        wc = cv2.resize(wc, small_size)
        mask = cv2.resize(mask, small_size)
        mask = mask.astype(np.bool)

    row, col = wc.shape[:2]
    wc_ch0 = wc[..., 0]
    wc_ch1 = wc[..., 1]
    wc_ch2 = wc[..., 2]

    X_w_M = wc_ch0[ mask ]
    Y_w_M = wc_ch1[ mask ]
    Z_w_M = wc_ch2[ mask ]

    ##### 以下處理視覺化
    ### 3D 直接畫出來囉～～
    fig, ax, ax_c = check_fig_ax_init(fig=None, ax=None, ax_c=None, fig_rows=1, fig_cols=3, ax_size=ax_size, tight_layout=True)
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    # ax.remove()
    # ax = Axes3D(fig)
    ax3d_origin = change_into_3D_coord_ax(fig, ax, 0)
    ax3d_side_z = change_into_3D_coord_ax(fig, ax, 1)
    ax3d_top_xy = change_into_3D_coord_ax(fig, ax, 2)
    _draw_wc_3d(ax3d_origin, azim= -50, elev = 16)
    _draw_wc_3d(ax3d_side_z, azim= -90, elev =  0)
    _draw_wc_3d(ax3d_top_xy, azim=   0, elev = 90)

    ### 看一下 z軸 的使用率
    Z_used_ratio, Z_w_M_top_res_ratio, Z_w_M_bot_res_ratio, Z_w_M_top_bot_ratio_res = _see_used_and_residual_ratio("z")
    Y_used_ratio, Y_w_M_top_res_ratio, Y_w_M_bot_res_ratio, Y_w_M_top_bot_ratio_res = _see_used_and_residual_ratio("Y")
    X_used_ratio, X_w_M_top_res_ratio, X_w_M_bot_res_ratio, X_w_M_top_bot_ratio_res = _see_used_and_residual_ratio("X")
    fig.text(0.15, 0.95, "ord_wc")
    fig.text(0.45, 0.85, f"{Z_used_ratio}%\n{Z_w_M_top_res_ratio}%\n{Z_w_M_bot_res_ratio}%\n{Z_w_M_top_bot_ratio_res}%")
    fig.text(0.72, 0.85, f"{Y_used_ratio}%\n{Y_w_M_top_res_ratio}%\n{Y_w_M_bot_res_ratio}%\n{Y_w_M_top_bot_ratio_res}%")
    fig.text(0.85, 0.85, f"{X_used_ratio}%\n{X_w_M_top_res_ratio}%\n{X_w_M_bot_res_ratio}%\n{X_w_M_top_bot_ratio_res}%")


    # plt.axis("off")     ### 純3D圖
    # ax.set_xticks([])   ### x軸無字
    # ax.set_yticks([])   ### y軸無字
    # ax.set_zticks([])   ### z軸無字

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
