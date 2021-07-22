### (128G 碩三下)H:\0 學校 碩三上\15篇吧\15_code_try
from build_dataset_combine import Check_dir_exist_and_build
import matplotlib.pyplot as plt
import numpy as np

def coordinate_dot(x_min, x_max, y_min, y_max, w_res, h_res, y_flip=False, dst_dir="."):
    '''
    y_flip=True 代表 原點在左上角，image coordinate 會用到
    '''
    Check_dir_exist_and_build(dst_dir)  ### 建立 dst_dir

    x = np.tile(np.reshape(np.linspace(x_min, x_max, w_res), [1, w_res]), [h_res, 1])
    y = np.tile(np.reshape(np.linspace(y_min, y_max, h_res), [h_res, 1]), [1, w_res])
    if(y_flip): y = y_max - y
    # y_t = np.expand_dims(y, axis=-1)
    # x_t = np.expand_dims(x, axis=-1)
    # meshgrid = np.concatenate([y_t, x_t], axis=-1)
    # print("meshgrid.shape", meshgrid.shape)


    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout()
    fig.set_size_inches(8, 8.1)
    ax.scatter(x, y, s=1)
    ax.set_xticks( x[0, ::2] )
    ax.set_yticks( y[::2, 0] )
    # ax.set_xlabel('X LABEL')
    # ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ### https://blog.csdn.net/qq_39429714/article/details/95988703
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.invert_yaxis()
    save_string = f"coord_x={x_min}~{x_max}, y={y_min}~{y_max}, w_res={w_res}, h_res={h_res}"
    plt.savefig(dst_dir + "/" + save_string)
    plt.savefig(dst_dir + "/" + f"{save_string}_透明", transparent=True)
    # plt.show()


    '''
    和上面的效果幾乎差不多，本來是希望能夠畫出漂亮的 x,y 軸才用下面的 axisartist，
    但後來發現 y軸方向 改不了！ 所以幾乎就沒用了！
    不過也是個還不錯的例子就保留下來囉！
    '''
    import mpl_toolkits.axisartist as AA
    fig2 = plt.figure()
    axAA = AA.Subplot(fig2, 111)
    fig2.add_axes(axAA)
    fig2.set_size_inches(8, 8)
    axAA.scatter(x, y, s=1)
    fig.tight_layout()
    axAA.set_xticks( x[0, ::2] )
    axAA.set_yticks( y[::2, 0] )
    axAA.set_ylim(y_max, y_min)  ### 用ylim設反 也沒辦法讓箭頭 往下 QAQ
    ### https://blog.csdn.net/A_Z666666/article/details/80400858
    ### https://blog.csdn.net/COCO56/article/details/100173824
    axAA.axis["top"   ].set_axisline_style("-|>", size = 2.5)
    axAA.axis["left"  ].set_axisline_style("-|>", size = 2.5)
    axAA.axis["left"  ].set_ticklabel_direction("-")
    axAA.axis["bottom"].set_visible(False)
    axAA.axis["right" ].set_visible(False)
    axAA.axis["top"   ].major_ticklabels.set_visible(True)
    axAA.axis["bottom"].major_ticklabels.set_visible(False)
    # axAA.invert_yaxis()
    # print(dir(axAA.axis["left"  ]))
    # print(dir(axAA))
    # print(axAA.yaxis_inverted())
    plt.show()


coordinate_dot(x_min=0, x_max=20, y_min=0, y_max=20, w_res=21, h_res=21, y_flip=True, dst_dir="grid_util_result")
coordinate_dot(x_min=0, x_max= 1, y_min=0, y_max= 1, w_res=21, h_res=21, y_flip=True, dst_dir="grid_util_result")
