import os
import shutil
# operating_system = "windows" ### 有用到linux在把他補完

### 來源資料夾
src_dir = "util_code"

### 決定 目的地
user_name = "TKU"

env_name = "base"
# env_name = "pytorch"
# env_name = "tensorflow200rc2"


#######################################################################################################################################
if(env_name=="base"): dst_path = "C:/Users"+"/"+user_name+"/Anaconda3/Lib" ### 例如："C:/Users/TKU/Anaconda3/Lib"
else                : dst_path = "C:/Users"+"/"+user_name+"/Anaconda3/envs"+"/"+ env_name+"/"  "Lib" ### 例如："C:/Users/TKU/Anaconda3/envs/tensorflow200rc2/Lib"

### 抓出要copy的.py
file_names = [file_name for file_name in os.listdir(src_dir) if ".py" in file_name.lower()]

### copy到 目的地
for file_name in file_names:
    print(src_dir + "/" + file_name, ", to", dst_path+"/"+file_name)
    shutil.copy(src_dir + "/" + file_name, dst_path+"/"+file_name)
