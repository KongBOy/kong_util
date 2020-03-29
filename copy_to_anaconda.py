import os
import shutil
# operating_system = "windows" ### 有用到linux在把他補完

### 決定目的地
user_name = "TKU"

# env_name = "base"
# env_name = "pytorch"
env_name = "tensorflow200rc2"


#######################################################################################################################################
if(env_name=="base"): copy_dst_path = "C:/Users"+"/"+user_name+"/Anaconda3/Lib" ### 例如："C:/Users/TKU/Anaconda3/Lib"
else                : copy_dst_path = "C:/Users"+"/"+user_name+"/Anaconda3/envs"+"/"+ env_name+"/"  "Lib" ### 例如："C:/Users/TKU/Anaconda3/envs/tensorflow200rc2/Lib"

### 本層目錄抓出要copy的.py
file_names = [file_name for file_name in os.listdir("copy_code") if ".py" in file_name.lower()]
for file_name in file_names:
    print("copy_code/"+file_name, ", to", copy_dst_path+"/"+file_name)
    shutil.copy("copy_code/"+file_name, copy_dst_path+"/"+file_name)

