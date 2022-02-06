from re import L
import numpy as np
import matplotlib.pyplot as plt

def get_receptive_filed_feature_length(kernel_size, strides, layer, ord_len):
    '''
        L1 的 receptive_filed =  kernel_size
        L2 的 receptive_filed =  kernel_size * 2 + 1
        L3 的 receptive_filed = (kernel_size * 2 + 1) * 2 + 1
            * 2 + 1 這個動作 做 layer -1 次
            這個 1 是 kernel_size // strides  得到的
    '''
    receptive_field_feature_length = ord_len
    for _ in range(layer): receptive_field_feature_length = (receptive_field_feature_length - kernel_size) // strides + 1
    return receptive_field_feature_length

def get_receptive_filed_length(kernel_size, strides, layer):
    '''
        L1 的 receptive_filed =  kernel_size
        L2 的 receptive_filed =  kernel_size * 2 + 1
        L3 的 receptive_filed = (kernel_size * 2 + 1) * 2 + 1
            * 2 + 1 這個動作 做 layer -1 次
            這個 1 是 kernel_size // strides  得到的
    '''
    receptive_field_length = kernel_size
    for _ in range(layer - 1): receptive_field_length = receptive_field_length * strides + kernel_size // strides
    return receptive_field_length

def get_receptive_filed_step(strides, layer):
    return strides ** layer

def get_receptive_field_mask(kernel_size, strides, layer, img_shape, Mask, vmin=0, print_msg=False):
    if  (len(Mask.shape) == 4 ): Mask = Mask[0]
    elif(len(Mask.shape) == 2 ): Mask = Mask[..., np.newaxis]
    h, w = img_shape[:2]

    receptive_filed_mask = np.ones(shape=(h, w, 1), dtype=np.float32) * vmin

    receptive_filed_length = get_receptive_filed_length(kernel_size=kernel_size, strides=strides, layer=layer)
    receptive_filed_step   = get_receptive_filed_step  (               strides=strides, layer=layer)

    y_index, x_index, z_index = np.nonzero(Mask)  ### 多了z 是因為 Mask 的 shape 為 (h, w, c)， 多了c 的部分喔！
    if(print_msg):
        print("x_index:", x_index)
        print("y_index:", y_index)
        print("z_index:", z_index)

    x_start = receptive_filed_step * x_index
    x_end   = receptive_filed_step * x_index + receptive_filed_length
    y_start = receptive_filed_step * y_index
    y_end   = receptive_filed_step * y_index + receptive_filed_length
    if(print_msg):
        print("x_start:", x_start)
        print("x_end  :", x_end)
        print("y_start:", y_start)
        print("y_end  :", y_end)

    for go_i, _ in enumerate(x_index):
        receptive_filed_mask[y_start[go_i] : y_end[go_i], x_start[go_i]:x_end[go_i]] = 1.

    if(print_msg):
        plt.imshow(receptive_filed_mask)
        plt.show()

    return receptive_filed_mask

def tf_M_resize_then_erosion_by_kong(Mask, resize_h, resize_w):
    import tensorflow as tf
    ''' Mask shape 要為 BHWC 喔 '''
    kernel = tf.ones((3, 3, 1))
    Mask = tf.image.resize(Mask, (resize_h, resize_w), method=tf.image.ResizeMethod.BILINEAR)
    if(Mask.shape[1] == 3 and Mask.shape[2] == 3 ):
        Mask = Mask * tf.constant( [[[ 0 ], [ 0 ], [ 0 ]],
                                    [[ 0 ], [ 1 ], [ 0 ]],
                                    [[ 0 ], [ 0 ], [ 0 ]]], dtype=tf.float32)
    else:
        Mask = tf.nn.erosion2d(Mask, filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)) + 1
    return Mask


if(__name__ == "__main__"):
    kernel_size = 4
    strides = 2
    layer  = 5

    import os
    import numpy
    import cv2
    import tensorflow as tf

    W_w_M_dir        = "J:/kong_render_os_book_and_paper_all_have_dtd_hdr_mix_bg_512/2_wc_w_M_npy"
    W_w_M_visual_dir = "J:/kong_render_os_book_and_paper_all_have_dtd_hdr_mix_bg_512/2_wc_visual"

    W_w_M_file_names        = os.listdir(W_w_M_dir)
    W_w_M_visual_file_names = os.listdir(W_w_M_visual_dir)
    for go_name, _ in enumerate(W_w_M_file_names):
        W_w_M_path        = f"{W_w_M_dir}/{W_w_M_file_names[go_name]}"
        W_w_M_visual_path = f"{W_w_M_visual_dir}/{W_w_M_visual_file_names[go_name]}"
        W_w_M        = np .load  (W_w_M_path)
        W_w_M_visual = cv2.imread(W_w_M_visual_path, 0)
        h, w = W_w_M_visual.shape[:2]

        # print("W_w_M.dtype", W_w_M.dtype)
        # print("W_w_M_visual.dtype", W_w_M_visual.dtype)

        # W = W_w_M[..., 0:3]
        M_ord = W_w_M[..., 3:4]

        receptive_filed_feature_length = get_receptive_filed_feature_length(kernel_size=kernel_size, strides=strides, layer=layer, ord_len=h)
        e_kernel = tf.ones((3, 3, 1))
        M = M_ord[tf.newaxis, ...]
        M_reisze = tf.image.resize(M, size=(receptive_filed_feature_length, receptive_filed_feature_length))
        M_erosion = tf.nn.erosion2d(M_reisze, filters=e_kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)) + 1
        M_erosion = M_erosion[0]

        # img = np.ones(shape=(512, 512, 3))
        receptive_filed_mask = get_receptive_field_mask(kernel_size=kernel_size, strides=strides, layer=layer, img_shape=W_w_M_visual.shape, Mask=M_erosion, vmin=0.5)
        # receptive_filed_mask = get_receptive_field_mask(kernel_size=kernel_size, strides=strides, layer=layer, img_shape=W_w_M_visual.shape, Mask=M_reisze[0], vmin=0.5)
        W_w_M_visual_w_receptive_field = W_w_M_visual * receptive_filed_mask

        canvas_size = 5
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5 * canvas_size, canvas_size))
        ax[0].imshow(W_w_M_visual)
        ax[1].imshow(M_ord)
        ax[2].imshow(M_erosion)
        ax[3].imshow(receptive_filed_mask)
        ax[4].imshow(W_w_M_visual, alpha=receptive_filed_mask[..., 0])
        fig.tight_layout()
        plt.show()
