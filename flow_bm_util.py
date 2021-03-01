import numpy as np
import scipy.interpolate as spin
import cv2
def use_flow_to_get_bm(flow, flow_scale):
    '''
    fl: the forward mapping in range [0,1]
    s: shape of the image required to sample from, defines the range of image coordinates in
    '''
    fl = flow.copy()     ### (540, 540, 3)
    # msk = fl[:, :, 0] > 0    ### (540, 540)  ### 原本的DewarpNet這樣寫，但是可能是他們處理得很乾淨吧所以可以這樣，我的有雜邊所以需要 >=1 喔！
    msk = fl[:, :, 0] >= 0.99  # 1  ### 幹真的差好多，這樣就對了，self.mask是(540, 540, 3)，msk是(540, 540)所以這行還是要有喔
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


def use_bm_to_rec_img(bm, flow_scale, dis_img):
    bm = np.around(bm * flow_scale)

    sh = flow_scale
    sw = flow_scale
    # s2 = np.array([[[sh, sw]]])
    # bm = np.around(self.bm * s2)

    bm = bm.astype(np.int32)
    bm = np.clip(bm, 0, flow_scale - 1)
    # print("bm.max()", bm.max())
    # print("bm.min()", bm.min())
    result = dis_img[bm[..., 0], bm[..., 1], :]  ### 根據 bm 去 dis_img 把 圖片攤平
    result = cv2.resize(result, (sh, sw))  ### 覺得如果根據blender的運作原理，應該還是bm回 方形， 再 resize成自己要的形狀 比較符合 blender的運作
    return result
