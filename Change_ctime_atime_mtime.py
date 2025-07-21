### 大致架構從這來的，但 win32file 的 CreateFile參數有錯, SetFileTime參數不完整：https://www.796t.com/article.php?id=107325
### 參數正確，參考這邊的參數：http://fygul.blogspot.com/2017/11/python-file-creation-time.html
from win32file import CreateFile, SetFileTime, GetFileTime, CloseHandle
from win32file import GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING
import pywintypes  # 可以忽視這個 Time 報錯（執行程式還是沒問題的）

import os ### os抓的time是 自1970/01/01 00:00:00 以來到現在的總秒數：https://vimsky.com/zh-tw/examples/usage/python-os-path-getctime-method.html
import time
import shutil

def modifyFileTime(file_path, os_ctime, os_atime, os_mtime, print_msg=False):
    """
    用來修改任意檔案的相關時間屬性，時間格式：YYYY-MM-DD HH:MM:SS 例如：2019-02-02 00:01:02
      file_path: 檔案路徑名
      os_ctime: createTime: 建立時間
      os_atime: modifyTime: 修改時間
      os_mtime: accessTime: 訪問時間
    """
    fh = CreateFile(file_path, GENERIC_WRITE, 0 , None, OPEN_EXISTING, 0, None)
    win_ctime = pywintypes.Time(int(os_ctime))
    win_atime = pywintypes.Time(int(os_atime))
    win_mtime = pywintypes.Time(int(os_mtime))
    if(print_msg):
        print("win_ctime", win_ctime)
        print("win_atime", win_atime)
        print("win_mtime", win_mtime)


    SetFileTime(fh, win_ctime, win_atime, win_mtime)
    CloseHandle(fh)

def Copy_DirFiles_and_AddDate_to_file_name(ord_dir, dst_dir=None, print_msg=False):

    if(dst_dir is None):
        # dir_name = ord_dir.split("\\")[-1]
        # dst_dir = dir_name + "_copy_to_there"
        dst_dir = ord_dir + "_copy_to_there"
        os.makedirs(dst_dir, exist_ok=True)

    file_names = os.listdir(ord_dir)
    for file_name in file_names:
        ord_file_path = f"{ord_dir}/{file_name}"    ### 定位 ord_file_path
        os_ctime = os.path.getctime(ord_file_path)  ### 抓出 os 的 ctime： 上次 create 的時間 (建立時間)
        os_mtime = os.path.getmtime(ord_file_path)  ### 抓出 os 的 mtime： 上次 modify 的時間 (修改時間)
        os_atime = os.path.getatime(ord_file_path)  ### 抓出 os 的 atime： 上次 access 的時間

        ctimestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(os_ctime))  ### 轉換成我想要得 時間格式
        atimestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(os_atime))  ### 轉換成我想要得 時間格式
        mtimestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(os_mtime))  ### 轉換成我想要得 時間格式
        dst_file_path = f"{dst_dir}/{ctimestamp}-{file_name}"  ### 定位 dst_file_path
        shutil.copy(ord_file_path, dst_file_path)  ### 複製！ ord_file_path --> dst_file_path
        if(print_msg): print("copy:", ord_file_path, "-->", dst_file_path, "finish")

        modifyFileTime(dst_file_path, os_ctime, os_atime, os_mtime, print_msg=True)

if __name__ == '__main__':
    # 需要自己配置
    cTime = 1626577390.0  # 建立時間
    mTime = 1626577390.0  # 修改時間
    aTime = 1626577390.0  # 訪問時間
    fName = "C:/Users/TKU/Desktop/arange iphone record/test.txt"  # 檔案路徑，檔案存在才能成功（可以寫絕對路徑，也可以寫相對路徑）

    offset = (0, 1, 2)  # 偏移的秒數（不知道幹啥的）

    # 呼叫函式修改檔案建立時間，並判斷是否修改成功
    modifyFileTime(fName, cTime, mTime, aTime, print_msg=True)
