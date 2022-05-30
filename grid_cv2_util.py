### (128G 碩三下)H:\0 學校 碩三上\15篇吧\15_code_try
from kong_util.build_dataset_combine import Check_dir_exist_and_build
import cv2
import numpy as np

def white_square_grid( grid_width=30, grid_amount=21, edge_width=2, abs_width=None, dst_dir=".", save_name="grid2"):
    """
    這裡 只畫 正方形grid 且 邊框黑色 喔！長相大概是：
        {edge_width + grid_width} + {} + ... 共 grid_amount 個 + edge_width

    grid_width  ： 小格子白色部分多寬
    grid_amount ： 小格子數
    edge_width  ： 小格子邊邊寬度
    abs_width   ： 整張圖的 寬度 希望要多寬，如果 abs_width - edge_width 除grid_amount 除的盡 才會剛好喔
    """
    Check_dir_exist_and_build(dst_dir)  ### 建立 dst_dir

    if(abs_width is not None):
        grid_width = (abs_width - edge_width) // grid_amount - edge_width
    grid = np.ones(shape = (grid_width, grid_width)) * 255
    grid = np.pad(grid, pad_width=( (edge_width, 0), (edge_width, 0) ))
    grid = np.tile(grid, (grid_amount, grid_amount))
    grid = np.pad(grid, pad_width=( (0, edge_width), (0, edge_width) ))
    cv2.imwrite(dst_dir + "/" + save_name + ".jpg", grid)


if(__name__ == "__main__"):
    white_square_grid(grid_width=30, grid_amount=21, edge_width=2, dst_dir="grid_util_result", save_name="grid2")
