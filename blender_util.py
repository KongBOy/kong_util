import sys
# sys.path.append("C:/Users/TKU/Desktop/kong_model2/kong_util")
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from util import get_dir_certain_file_name
import os

import bpy
import shutil


def get_dir_blends_and_extract_texture_image_file_name(blender_ord_dir, dst_dir, print_msg=False):
    """
    執行的時候要在cmd裡面打指令：blender --background --python blender_util.py

    blender_ord_dir：放 .blend 存的地方
    dst_dir：你的 texture 想要存哪裡

    如果有遇到 "整理資料夾" 可能會讓 blender 裡面的 filepath 跟 目前的資料夾對不上， 那可能就需要 這個 page_ord_paths囉！
    覺得有遇到再寫，
    因為我應該要從源頭解決這問題，
    就是在 Blender 裡面多 Render 出 ord_page 就好了！
        page_ord_paths   ：放 blender texture 的來源paths
    """

    Check_dir_exist_and_build_new_dir(dst_dir)  ### 建立 放結果的資料夾

    blender_file_names = get_dir_certain_file_name(blender_ord_dir, certain_word=".blend")  ### 抓出 blender_ord_dir 裡面的 所有.blend 的 file_names
    for blender_file_name in blender_file_names:
        blender_file_path = f"{blender_ord_dir}/{blender_file_name}"           ### 定位出blender檔的path
        bpy.ops.wm.open_mainfile(filepath=blender_file_path)                   ### 在 Blender 內讀出 .blend 檔

        ord_path = bpy.data.materials[0].node_tree.nodes[2].image.filepath     ### 抓出 texture node 裡面的 filepath，這裡要自己去對應 try_do_all_291.py 裡面的 step4_page_texture_1_image_and_uv_material 裡的 ShaderNodeTexImage 建立順序喔！
        ord_file_name = ord_path.split("\\")[-1]                               ### 從filepath 裡 擷取出 file_name
        blender_file_id = int(blender_file_name.split("_")[-1].split(".")[0])  ### 分離出 blender_file 名字內的編號
        dst_path = "%s/%06i_%s" % (dst_dir, blender_file_id, ord_file_name)    ### 定位出 dst_path (根據 blender_file_id 和 ord_file_name， 把兩者關聯起來)
        shutil.copy(ord_path, dst_path)                                        ### 從 ord 複製到 dst
        if(print_msg): print(f"{ord_path}--copy->{dst_path} ok")

    print("get_dir_blends_and_extract_texture_image_file_name finish")


if(__name__ == "__main__"):
    # get_dir_blends_and_extract_texture_image_file_name(blender_ord_dir="K:/kong_render_os_book_no_bg_768",
    #                                                    dst_dir="K:/kong_render_os_book_no_bg_768/0_image_ord")
    get_dir_blends_and_extract_texture_image_file_name(blender_ord_dir="J:/kong_render_os_book_have_bg_512",
                                                       dst_dir="J:/kong_render_os_book_have_bg_512/0_image_ord")

### cv2
### tqdm
### matplotlib
