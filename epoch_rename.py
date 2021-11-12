import shutil
import sys
sys.path.append("..")
from step0_access_path import access_path
from util import get_dir_certain_file_names, get_dir_dir_names

### 應該用不到了，因為已經改寫好 訓練過程epoch產生的名字囉！
def Rename_epoch(ord_dir):
    file_names = get_dir_certain_file_names(ord_dir, ".png")
    for file_name in file_names:
        old_epoch_string = file_name.split("-")[0].split("_")[1]
        new_epoch_string = "%04i" % int(old_epoch_string)
        ord_path = ord_dir + "/" + "epoch_%s-result.png" % old_epoch_string
        ren_path = ord_dir + "/" + "epoch_%s-result.png" % new_epoch_string
        print("old", ord_path)
        print("new", ren_path)
        shutil.move(ord_path, ren_path)


if(__name__ == "__main__"):
    ord_dirs = get_dir_dir_names(access_path + "result")
    for dir_name in ord_dirs:
        Rename_epoch(access_path + "result" + "/" + dir_name)
    # print(ord_dirs)
