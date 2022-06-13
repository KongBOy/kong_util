#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
code_dir = "\\".join(code_exe_path_element[:-1])
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])  ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    code_dir:", code_dir)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################

### ...\碩一下 多媒體助教 密碼107summer\碩一下 多媒體助教\teach_filter_merge_ok\teach_filter_merge_ok
from kong_util.build_dataset_combine import Check_dir_exist_and_build
import numpy as np
import matplotlib.pyplot as plt
import time

def _decide_color_list_example(img_1ch,
                  fig_size = 10,
                  out_file_name = "map",
                  filter_type   = "",
                  doing_type    = ""):
    """
    建一個符合格式的color_list：顏色需要用 rgb channel 且 數值要0~1
    """
    img_height = img_1ch.shape[0]
    img_width  = img_1ch.shape[1]

    value = img_1ch.copy()  #np.ones( shape=(img_height, img_width), dtype = np.uint8)*100

    ### 根據 filter_type 決定顏色 怎麼畫
    color_list = value.copy()
    ### 如果是filter，背景顏色用白色
    if(filter_type == "kernel_3"):
        color_list.fill(255)
        value = value.astype(np.dtype("<U3"))
        value.fill("1/9")
    elif(filter_type == "kernel_3_one"):
        color_list.fill(255)
        value = value.astype(np.str)
        value.fill("1")
    elif(filter_type == "kernel_5"):
        color_list.fill(255)
        # value = value.astype(np.str)
        value = value.astype(np.dtype("<U4"))  ### 這樣才可以存 4個byte char
        value.fill("1/25")
        # print(value)
    elif(filter_type == "kernel_5_one"):
        color_list.fill(255)
        value = value.astype(np.str)
        value.fill("1")
    elif("prewitt" in filter_type):
        color_list[ color_list > 0 ] = 255
        color_list[ color_list == 0] = 128
        color_list[ color_list < 0]  =  30

    elif("sobel" in filter_type):
        color_list[ color_list == -2] =  10
        color_list[ color_list == -1] =  40
        color_list[ color_list ==  0] = 150
        color_list[ color_list ==  1] = 210
        color_list[ color_list ==  2] = 255

    if("prewitt" in doing_type or "sobel" in doing_type):
        color_list[ color_list > 255] = 255


    ### 顏色需要用 rgb channel 且 數值要0~1，所以建一個符合格式的color_list
    color_list = color_list.astype(np.float64) / 255
    color_list = color_list.reshape(img_height, img_width, 1)
    color_list = np.tile(color_list, (1, 1, 3))
    return color_list


def Image_to_grid(img_1ch,
                  fig_size      = 10,
                  text_align    = "left",  ### 設定 格子內 文字對齊方式，預設放左邊
                  color_list    = None,
                  empty_grid    = False,
                  out_file_name = "map",
                  dst_dir       = ".",
                  print_msg     = False,
                  ):
    """
    img_1ch       ： shape 為 (h, w)，如果 (h, w, 1) 也是可以， 不過 表格裡面的字可能會多一層 [] 喔
    fig_size      ： 就 fig_size
    text_align    ： 表格裡的字要靠哪邊
    color_list    ： 表格每一格的顏色， shape 為 (h, w, 3)， 值域為 0~1， color_list 如果為None 以下預設用 全白色
    empty_grid    ： 表格裡要不要放東西， True 就 empty 的 table ， False 就放 img_1ch 的 值
    out_file_name ： 想存的 檔名
    dst_dir       ： 想存的 dst_dir
    print_msg     ： show_time
    """
    Check_dir_exist_and_build(dst_dir)  ### 建立 dst_dir

    ord_grid_draw_time_start = time.time()
    img_height = img_1ch.shape[0]
    img_width  = img_1ch.shape[1]

    ### 格子裡面的字
    if(empty_grid): value = None            ### 表格裡不要有任何字
    else:           value = img_1ch.copy()  ### 表格裡的字 為 img_1ch 的數值

    ### 格子預設用白色，顏色需要用 rgb channel 且 數值要0~1，所以建一個符合格式的color_list
    if(color_list is None): color_list = np.ones((img_height, img_width, 3))


    ### 設定 新建畫布 及 設定畫布大小，取得現在的畫布
    fig = plt.figure(figsize=( fig_size, fig_size))
    ax = fig.gca()

    ### 畫出表格
    the_table = ax.table(cellText    = value,       ### cellText   是 表格內的字 shape(height,width)
                         cellColours = color_list,  ### color_list 是 每格的顏色 shape(height,width,3)
                         loc         ='center',
                         cellLoc     = text_align   ### cellLoc 是 格子內字的對齊方式
                        )

    ### 如果表格太小，會不會自動調整字的大小
    the_table.auto_set_font_size(False)

    ### 設定表格高度
    cell_img_height = 1 / (img_width)
    for pos, cell in the_table.get_celld().items():
        # print("color_list[pos]",color_list[pos])
        if(color_list[pos[0], pos[1], 0] < 0.196078):  # 如果背景顏色太深(灰階50)，字體用亮灰色
            cell._text.set_color('#AEAEAE')  ### https://stackoverflow.com/questions/58817940/how-to-change-cells-text-color-in-table

        cell.set_height(cell_img_height)

    ### 設定 xy軸 不要顯示字
    plt.xticks([])
    plt.yticks([])

    ### 設定 表格圖旁的白框 靠近 表格圖
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # fig.tight_layout()
    plt.savefig( dst_dir + "/" + out_file_name + ".png")
    # plt.show()
    # print("Drawing grid finished!")
    plt.close()


    ord_grid_draw_time_end = time.time()
    if(print_msg): print("ord_grid_draw cost time:", ord_grid_draw_time_end - ord_grid_draw_time_start)


if(__name__ == "__main__"):
    y_flip = False
    h_res = 21
    w_res = 21

    ### meshgrid
    x_min = 0
    x_max = 20
    y_min = 0
    y_max = 20
    x = np.tile(np.reshape(np.linspace(x_min, x_max, w_res), [1, w_res]), [h_res, 1])
    y = np.tile(np.reshape(np.linspace(y_min, y_max, h_res), [h_res, 1]), [1, w_res])
    mask = np.ones(shape=x.shape)
    if(y_flip): y = y_max - y
    y_t = np.expand_dims(y, axis=-1)
    y_t_gray = np.tile(y_t, (1, 1, 3)) / y_t.max()
    x_t = np.expand_dims(x, axis=-1)
    x_t_gray = np.tile(x_t, (1, 1, 3)) / x_t.max()
    meshgrid = np.concatenate([x_t, y_t], axis=-1)

    coord_color_blue = np.dstack( (x / x.max(), y / y.max(), mask) )
    coord_color_red  = np.dstack( (mask, y / y.max(), x / x.max()) )

    ### 轉 np.uint8 是為了 讓格子裡的字沒有小數點喔
    # Image_to_grid(x    .astype(np.uint8), fig_size= 5.8, out_file_name="map_x")
    # Image_to_grid(y    .astype(np.uint8), fig_size= 5.8, out_file_name="map_y")
    # Image_to_grid(mask .astype(np.uint8), fig_size= 5.8, out_file_name="map_mask")
    # Image_to_grid(y    .astype(np.uint8), fig_size= 5.8, out_file_name="map_y_gray", color_list=y_t_gray)
    # Image_to_grid(x    .astype(np.uint8), fig_size= 5.8, out_file_name="map_x_gray", color_list=x_t_gray)

    ### 這裡因為 empty_grid 為 True， img_1ch 只要 shape有對就好，直要丟 x, y 都沒差喔
    Image_to_grid(x     .astype(np.uint8), fig_size= 5.8, out_file_name="map_color_blue", color_list=coord_color_blue, empty_grid=True, dst_dir="grid_util_result")
    Image_to_grid(x     .astype(np.uint8), fig_size= 5.8, out_file_name="map_color_red" , color_list=coord_color_red , empty_grid=True, dst_dir="grid_util_result")
