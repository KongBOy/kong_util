import sys
sys.path.append("C:/Users/TKU/Desktop/kong_model2/kong_util")
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from util import get_dir_certain_file_name
import os

import bpy
import shutil


def get_dir_blends_and_extract_texture_image_file_name(page_ord_dir="K:/500G_transform_data/0 data_dir/datasets/type7_cut_os_book/produce_straight/1_page_num_ok",
                                                       blender_ord_dir="K:/kong_render_os_book_no_bg_768",
                                                       dst_dir="K:/kong_render_os_book_no_bg_768/0_image_ord"):
    """
    執行的時候要在cmd裡面打指令：blender --background --python blender_util.py

    page_ord_dir   ：放 blender texture 的來源，比如 os_book 直 的 資料夾
    blender_ord_dir：放 .blend 存的地方
    dst_dir：你的 texture 想要存哪裡
    """

    Check_dir_exist_and_build_new_dir(dst_dir)  ### 建立 放結果的資料夾

    blender_file_names = get_dir_certain_file_name(blender_ord_dir, certain_word=".blend")  ### 抓出 blender_ord_dir 裡面的 所有.blend 的 file_names
    for i, blender_file_name in enumerate(blender_file_names):
        bpy.ops.wm.open_mainfile(filepath=f"{blender_ord_dir}/{blender_file_name}")  ### 在 blender 內讀出 .blend 檔
        data = bpy.data.materials[0].node_tree.nodes[2].image.filepath  ### 抓出 texture node 裡面的 filepath，這裡要自己去對應 try_do_all_291.py 裡面的 step4_page_texture_1_image_and_uv_material 裡的 ShaderNodeTexImage 建立順序喔！
        file_name = data.split("\\")[-1]  ### 從filepath 裡 擷取出 file_name

        ord_path = f"{page_ord_dir}/{file_name}"             ### file_name 要配上 blender texture 的來源 位址喔！比如：K:/500G_transform_data/0 data_dir/datasets/type7_cut_os_book/produce_straight/1_page_num_ok/0696.jpg
        dst_path = f"{dst_dir}/%04i-{file_name}" % (i + 1)   ### 目的地想存哪，也可以自己設計 名字喔！            比如：K:/kong_render_os_book_no_bg_768/0_image_ord/0978-0696.jpg
        shutil.copy(ord_path, dst_path)
        print(f"{ord_path} copyt to {dst_path} finish")

    print("get_dir_blends_and_extract_texture_image_file_name finish")


if(__name__ == "__main__"):
    get_dir_blends_and_extract_texture_image_file_name(page_ord_dir="K:/500G_transform_data/0 data_dir/datasets/type7_cut_os_book/produce_straight/1_page_num_ok",
                                                       blender_ord_dir="K:/kong_render_os_book_no_bg_768",
                                                       dst_dir="K:/kong_render_os_book_no_bg_768/0_image_ord")
