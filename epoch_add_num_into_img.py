import cv2
import sys
sys.path.append("..")

from step0_access_path import access_path
from kong_util.util import get_dir_certain_file_names
from kong_util.build_dataset_combine import Check_dir_exist_and_build


def epoch_add_num_into_img(ord_dir, dst_dir):
    Check_dir_exist_and_build(dst_dir)
    file_names = get_dir_certain_file_names(ord_dir, ".png")
    for file_name in file_names:
        epoch_string = file_name.split("-")[0].split("_")[1]
        img = cv2.imread( ord_dir + "/" + file_name )
        cv2.putText(img, epoch_string, (10, int(img.shape[0] / 2) ), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(dst_dir + "/" + file_name, img)
        print(dst_dir + "/" + file_name, "add num finish")


if(__name__ == "__main__"):
    access_dir = access_path + "result"

    ord_dir = access_dir + "/" + "wei_book_tf1_db_20200408-225902_model5_rect2"
    dst_dir = ord_dir    + "/" + "epoch_add_num"
    epoch_add_num_into_img(ord_dir, dst_dir)

    # ord_dir = access_dir + "/" + "wei_book_tf1_db_20200410-025655_model6_mrf_rect2"
    # dst_dir = ord_dir    + "/" + "epoch_add_num"
    # epoch_add_num(ord_dir, dst_dir)
