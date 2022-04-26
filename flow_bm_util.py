import numpy as np
import scipy.interpolate as spin
import cv2

from matplot_fig_ax_util import check_fig_ax_init, move_map_2D_arrow, img_scatter_visual
from util import get_xy_f_and_m


def check_flow_quality_then_I_w_F_to_R(dis_img, flow):
    '''
    dis_img: 0~255 uint8   (h, w, 3)
    flow   : 0~1   float32 (h, w, 3) mask/y/x

    目前的 check 無法 判別 fake_see 或 with C_with_Mgt 的狀況喔～～　因為 Mask 都是幾乎完美的呀 ~~
    '''
    h, w = flow.shape[:2]
    total_pix_amount = h * w
    valid_mask_pix_amount = (flow[..., 0] >= 0.90).astype(np.int).sum()
    # print("dis_img.shape:", dis_img.shape)
    # print("valid_mask_pix_amount:", valid_mask_pix_amount)
    # print("valid_mask_pix_amount / total_pix_amount:", valid_mask_pix_amount / total_pix_amount)
    if( valid_mask_pix_amount / total_pix_amount > 0.20):
        bm  = use_flow_to_get_bm(flow, flow_scale=h)
        rec = use_bm_to_rec_img (bm  , flow_scale=h, dis_img=dis_img)
        # print("here1~~~~~~~~~")
    else:
        dis_h, dis_w = dis_img.shape[:2]
        bm  = np.zeros(shape=(h, w, 2))
        rec = np.zeros(shape=(dis_h, dis_w, 3))
        # print("here2~~~~~~~~~")
    return bm, rec


def use_flow_to_get_bm(flow, flow_scale):
    '''
    input:
        flow: H, W, 3， ch1:Mask, ch2:y, ch3:x

    參考code：https://github.com/cvlab-stonybrook/doc3D-dataset/issues/2

    fl: the forward mapping in range [0,1]
    s: shape of the image required to sample from, defines the range of image coordinates in

    result: 值域 0~1
    '''
    flow = flow.copy()
    flow [..., 1] = 1 - flow[..., 1]  ### y 上下 flip， 因為blender 裡面 y軸朝上， 影像的 y軸朝下
    fl = flow  ### (540, 540, 3)
    # msk = fl[:, :, 0] > 0    ### (540, 540)  ### 原本的DewarpNet這樣寫，但是可能是他們處理得很乾淨吧所以可以這樣，我的有雜邊所以需要 >=1 喔！
    msk = fl[:, :, 0] >= 0.99  # 1  ### 幹真的差好多，這樣就對了，self.mask是(540, 540, 3)，msk是(540, 540)所以這行還是要有喔， 原版只有 >0
    # print("bm calculate flow mask.sum()", msk.sum())

    ### 以下 分完 sh, sw 以後再仔細思考，好像這個只用 s 的才是最正確的，所以sh, sw 都設一樣囉，這版griddata沒有配好
    # s  = self.flow_scale
    # fl_s   = fl * s
    # fl_s_m = fl_s[msk]        ### (143217, 3)
    # tx, ty = np.nonzero(msk)  ### tx (143217,), ty (143217,)
    # grid = np.meshgrid(np.linspace(1, s, s), np.linspace(1, s, s))  ### meshgrid( 先x, 再y)，return 也是 [先x, 再y]，shape為 [x(h540, w540), y(h540, w540)]
    # vy = spin.griddata(fl_s_m[:, 1:], ty, tuple(grid), method='nearest') / float(s)  ### vy(540, 540)
    # vx = spin.griddata(fl_s_m[:, 1:], tx, tuple(grid), method='nearest') / float(s)  ### vx(540, 540)
    # result = np.stack([vx, vy], axis=-1)  ### (540, 540, 2)


    # if(msk.sum() > 0):
    ### meshgrid 和 griddata 有匹配
    sh = flow_scale
    sw = flow_scale  ##340
    s2 = np.array([[[1, sh, sw]]])
    fl_s   = fl * s2
    fl_s_m = fl_s[msk][:, ::-1]   ### (143217, 3), mask, y, x，改成 x, y, mask，讓下面scipy.interpolate.griddata(第一個參數是 先x)比較好用
    ty, tx = np.nonzero(msk)      ### ty(143217,), tx(143217,)
    grid = np.meshgrid(np.linspace(1, sw, sw), np.linspace(1, sh, sh))  ### meshgrid(先x, 再y)，return 也是 [先x, 再y]，shape為 [x(h540, w340), y(h540, w340)]
    vy = spin.griddata(fl_s_m[:, :2], ty, tuple(grid), method='nearest') / float(sh)  ### vy(h540, w340)
    vx = spin.griddata(fl_s_m[:, :2], tx, tuple(grid), method='nearest') / float(sw)  ### vx(h540, w340)
    result = np.stack([vy, vx], axis=-1)  ### (540, 540, 2), ch1:y, ch2:x
    return result

# def use_bm_to_rec_img_old(bm, flow_scale, dis_img):
#     bm = np.around(bm * flow_scale)

#     sh = flow_scale
#     sw = flow_scale
#     # s2 = np.array([[[sh, sw]]])
#     # bm = np.around(self.bm * s2)

#     bm = bm.astype(np.int32)
#     bm = np.clip(bm, 0, flow_scale - 1)
#     # print("bm.max()", bm.max())
#     # print("bm.min()", bm.min())
#     result = dis_img[bm[..., 0], bm[..., 1], :]  ### 根據 bm 去 dis_img 把 圖片攤平
#     result = cv2.resize(result, (sh, sw))  ### 覺得如果根據blender的運作原理，應該還是bm回 方形， 再 resize成自己要的形狀 比較符合 blender的運作
#     return result


def use_bm_to_rec_img(bm, flow_scale, dis_img):
    '''
    bm: (h, w, 2)
        訓練時設定的h, w
        ch0 是 y, ch1 是 x

    dis_img: (H, W, 3)
        原始拍照的大小
    '''
    # bm = np.around(bm * flow_scale)

    sh = dis_img.shape[0]
    sw = dis_img.shape[1]
    bm = cv2.blur(bm, (3, 3))
    bm = cv2.resize(bm, (sw, sh))
    s2 = np.array([[[sh, sw]]])
    bm = bm * s2
    # bm = np.around(bm * s2)

    bm = bm.astype(np.int32)
    bm[0] = np.clip(bm[0], 0, sh - 1)
    bm[1] = np.clip(bm[0], 0, sw - 1)
    # print("bm.max()", bm.max())
    # print("bm.min()", bm.min())
    result = dis_img[bm[..., 0], bm[..., 1], :]  ### 根據 bm 去 dis_img 把 圖片攤平
    # result = cv2.resize(result, (sh, sw))  ### 覺得如果根據blender的運作原理，應該還是bm回 方形， 再 resize成自己要的形狀 比較符合 blender的運作
    return result


def bm_arrow_visual(bm,
                    x_min=-1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00,
                    jump_r=1, jump_c=1,
                    bm_arrow_alpha     = 0.50, bm_arrow_c=None, bm_arrow_cmap     = "hsv",
                    bm_after_dot_alpha = 0.35, bm_after_dot_c=None, bm_after_dot_cmap = "hsv",
                    fig=None, ax=None, ax_c=None, tight_layout=False):
    '''
    先假設 bm 值域在 -1~1
    '''
    fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=1, fig_cols=1, ax_size=8, tight_layout=tight_layout)
    ##################################################################################################################
    bm_h, bm_w = bm.shape[:2]
    start_xy_f, start_xy_m = get_xy_f_and_m(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=bm_w, h_res=bm_h)  ### 拿到map的shape：(..., 2), f 是 flatten 的意思

    see_inv_move_map_m = bm - start_xy_m    ### 計算 move_map = dst - start， 如果有nan 減完 仍為 nan ， dis_coord_big 的話 see_inv_coord_f 填滿滿
    fig, ax, ax_c = move_map_2D_arrow(see_inv_move_map_m, start_xy_m=start_xy_m,
        fig_title="bm visual",
        jump_r=jump_r, jump_c=jump_c,
        arrow_C=bm_arrow_c, arrow_alpha=bm_arrow_alpha, arrow_cmap=bm_arrow_cmap,  ### 視覺化 boundary內 新bm 的 dis_coord/move_map/移動後的dis_coord
        show_before_move_coord=False,
        show_after_move_coord =True, after_C=bm_after_dot_c, after_alpha=bm_after_dot_alpha, after_cmap=bm_after_dot_cmap,
        fig=fig, ax=ax, ax_c=ax_c)
    ax_c += 1
    return fig, ax, ax_c



def dis_bm_rec_visual( dis_img, bm, rec_img, img_smaller=0.5,
                       x_min= -1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00,
                       jump_r=1, jump_c=1,
                       dis_alpha=1.0, dis_dot_s=2,
                       bm_arrow_alpha     = 1.0, bm_arrow_c     = None, bm_arrow_cmap="hsv",
                       bm_after_dot_alpha = 1.0, bm_after_dot_c = None, bm_after_dot_cmap="hsv",
                       fig=None, ax=None, ax_c=None, tight_layout=False):
    '''
    先假設 bm 值域在 -1~1
    '''
    fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=1, fig_cols=3, ax_size=8, tight_layout=tight_layout)
    ##################################################################################################################
    ax[0].set_title("dis_img")
    ax[0].imshow(dis_img)

    ##################################################################################################################
    bm_h, bm_w = bm.shape[:2]
    dis_h, dis_w = dis_img.shape[:2]
    start_xy_f, start_xy_m = get_xy_f_and_m(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=bm_w, h_res=bm_h)  ### 拿到map的shape：(..., 2), f 是 flatten 的意思
    fig, ax, ax_c = img_scatter_visual(cv2.resize(dis_img, ( int(dis_w * img_smaller), int(dis_h * img_smaller)) ),
                                       x_min=x_min, x_max=x_max, y_min= y_min, y_max= y_max,
                                       alpha=dis_alpha, s=dis_dot_s,
                                       fig=fig, ax=ax, ax_c=1, tight_layout=tight_layout)

    fig, ax, ax_c = bm_arrow_visual(bm,
                                    x_min=x_min, x_max=x_max, y_min= y_min, y_max= y_max,
                                    jump_r=jump_r, jump_c=jump_r,
                                    bm_arrow_alpha     = bm_arrow_alpha,     bm_arrow_c    =np.arange(bm_h * bm_w).reshape(bm_h, bm_w), bm_arrow_cmap=bm_arrow_cmap,
                                    bm_after_dot_alpha = bm_after_dot_alpha, bm_after_dot_c=np.arange(bm_h * bm_w).reshape(bm_h, bm_w), bm_after_dot_cmap=bm_after_dot_cmap,
                                    fig=fig, ax=ax, ax_c=1, tight_layout=tight_layout)
    ##################################################################################################################
    ax[2].set_title("rec_img")
    ax[2].imshow(rec_img)
    ax_c = 3
    return fig, ax, ax_c
    # see_inv_move_map_m = bm - start_xy_m    ### 計算 move_map = dst - start， 如果有nan 減完 仍為 nan ， dis_coord_big 的話 see_inv_coord_f 填滿滿
    # fig, ax, ax_c = move_map_2D_arrow(see_inv_move_map_m, start_xy_m=start_xy_m,
    #     arrow_C=None, arrow_cmap="gray",  ### 視覺化 boundary內 新bm 的 dis_coord/move_map/移動後的dis_coord
    #     fig_title="bm visual",
    #     jump_r=jump_r, jump_c=jump_c,
    #     arrow_alpha=1.0,
    #     show_before_move_coord=False, before_alpha=0.8,
    #     show_after_move_coord =True, after_alpha=0.10, after_C="red",
    #     fig=fig, ax=ax, ax_c=1)

if(__name__ == "__main__"):
    import os
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    dis_dir = r"J:\kong_render_os_book_and_paper_all_have_dtd_hdr_mix_bg_512\0_image"
    dis_file_names = os.listdir(dis_dir)

    uv_dir = r"J:\kong_render_os_book_and_paper_all_have_dtd_hdr_mix_bg_512\1_uv_npy"
    uv_file_names = os.listdir(uv_dir)
    for go_f, _ in enumerate(os.listdir(uv_dir)):
        dis_path = f"{dis_dir}/{dis_file_names[go_f]}"
        dis_img = cv2.imread(dis_path)

        uv_path  = f"{uv_dir}/{uv_file_names[go_f]}"
        uv = np.load(uv_path)
        h, w, c = uv.shape
        m = uv[..., 0]
        y = uv[..., 1]
        x = uv[..., 2]
        bm  = use_flow_to_get_bm(uv, flow_scale=h)
        bm_y = bm[..., 0]
        bm_x = bm[..., 1]

        ### 測試網路上找的 tf2 來做bm的程式碼 能不能還原dis_img， 結果是成功的！
        # from flow_bm_tf_try import bilinear_sampler
        # tf_dis_img = dis_img[np.newaxis, ...].astype(np.float32)
        # tf_x       = x      [np.newaxis, ...]
        # tf_y       = y      [np.newaxis, ...]
        # tf_bm_x    = bm_x   [np.newaxis, ...]
        # tf_bm_y    = bm_y   [np.newaxis, ...]
        # result = bilinear_sampler(tf_dis_img, tf_bm_x, tf_bm_y)
        # # result = bilinear_sampler(tf_dis_img, tf_x, tf_y)

        # print(result.shape)
        # print(result.numpy().max())
        # print(result.numpy().min())
        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        # ax[0].imshow(m)
        # ax[1].imshow(dis_img)
        # ax[2].imshow(result[0].numpy().astype(np.uint8))
        # fig.tight_layout()
        # plt.show()

        ### 測試 取 mask 要用 2d 還 3d 的寫法，結論是看狀況
        #   做 bm 或 tight_crop 的時候用 2d就好， 因為 mask 會丟給 np.nonzero， 這樣 3d 多一個用不到的 z_channel
        #   給 dis_img 或 wc 套用時 用 3d， 因為 要做 相乘 這樣子 shape 才對應的到 才可以 broadcast 
        # mask2d = uv[..., 0]   > 0.99  ### shape = (512, 512)
        # mask3d = uv[..., 0:1] > 0.99  ### shape = (512, 512, 1)
        ### bm中使用 nonzero：
        # data2d = np.nonzero(mask2d)   ### 會有 xy  的結果
        # data3d = np.nonzero(mask3d)   ### 會有 xyz 的結果， 多一個z_channel 且 值全為0
        ### dis_img, wc 只取 mask 部分：
        # img = dis_img * mask2d  ### 無法會出錯
        # img = dis_img * mask3d  ### 正確


        ### 測試 dis_img, uv(flow) 一起 crop, resize回原本大小後， uv(flow) 做出來的bm能不能直接套用， 我預測是可以， 測試結果也確定沒問題！ 
        # 只是會變得有點務， 所以 Doc3D 才從 448 -> 256 以避免放大變霧吧我猜
        # 所以我覺得也許我也該弄個 1024 的 Dataset， crop 後 再 resize 到 512 也許就沒問題了～
        # 比較麻煩的是我能用 Doc3D 的 448 套用到我的model 嗎
        ord_mask2d = uv[..., 0]   > 0.99
        ord_y, ord_x = np.nonzero(ord_mask2d)
        ord_y_min = min(ord_y)
        ord_y_max = max(ord_y)
        ord_x_min = min(ord_x)
        ord_x_max = max(ord_x)

        crop_dis_img = dis_img[ord_y_min : ord_y_max + 1, ord_x_min : ord_x_max + 1]
        crop_uv      = uv     [ord_y_min : ord_y_max + 1, ord_x_min : ord_x_max + 1]

        crop_resize_dis_img = cv2.resize(crop_dis_img, (w, h))
        crop_resize_uv      = cv2.resize(crop_uv     , (w, h))
        crop_resize_bm  = use_flow_to_get_bm(crop_resize_uv, flow_scale=h)
        crop_resize_rec = use_bm_to_rec_img (crop_resize_bm, flow_scale=h, dis_img=crop_resize_dis_img)

        rec             = use_bm_to_rec_img (bm            , flow_scale=h, dis_img=dis_img            )

        fig_base_size = 8
        col_imgs_amount = 6
        fig, ax = plt.subplots(nrows=1, ncols=col_imgs_amount, figsize=(fig_base_size * col_imgs_amount, fig_base_size))
        ax[0].imshow(ord_mask2d)
        ax[1].imshow(dis_img)
        ax[2].imshow(crop_dis_img)
        ax[2].imshow(crop_resize_dis_img)
        ax[3].imshow(crop_resize_rec)
        ax[4].imshow(rec)
        fig.tight_layout()
        plt.show()
        print("finish")
