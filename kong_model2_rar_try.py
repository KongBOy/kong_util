from multiprocessing import Process
import time
import os
from kong_util.util import Visit_sub_dir_include_self_and_get_dir_paths
import shutil
from tqdm import tqdm

start_time = time.time()

def compress_train_code(src_dir):
    print("compress_train_code, src_dir=", src_dir)
    dir_containor = []
    Visit_sub_dir_include_self_and_get_dir_paths(src_dir = src_dir, dir_containor=dir_containor)

    train_code_dirs = [dir_name for dir_name in dir_containor if "train_code" in dir_name.split("\\")[-1]]
    for go_dir, train_code_dir in enumerate(tqdm(train_code_dirs)):
        print("processing_dir:", train_code_dir)
        dir_start_time = time.time()
        # print(go_dir, train_code_dir)
        train_code_dir_name = train_code_dir.split("\\")[-1]
        os.chdir(train_code_dir)
        os.chdir("..")
        print(os.listdir())
        os.system(f"winrar a -m5 -r {train_code_dir_name}.rar {train_code_dir_name}")

        try:
            shutil.rmtree(train_code_dir_name)
        except:
            print("processing_dir:", train_code_dir)
            with open(r"C:\Users\HP820G1\Desktop\winrar_python\delete_error.txt", "a") as f:
                f.write("processing_dir: " + train_code_dir + "\n")

        print(os.listdir())
        print(f"{go_dir}: {train_code_dir_name} finished cost_time = ", time.time() - dir_start_time)
    print("all dir cost_time = ", time.time() - start_time)

if __name__ == "__main__":

    # src_dir = r"E:\d\r\7\d\Ablation2_Pyr_ch016_ep010"
    # compress_train_code(src_dir)

    # src_dir = r"E:\d\r\7\d\Ablation4"
    # Process(target = compress_train_code, args = (src_dir, )).start()

    # src_dir = r"E:\d\r\7\d\Ablation4_2blk_ch016_ep010_wiwoM_Chbg"
    # Process(target = compress_train_code, args = (src_dir, )).start()

    # src_dir = r"E:\d\r\7\d\Ablation4_ch016_ep003"
    # Process(target = compress_train_code, args = (src_dir, )).start()

    # src_dir = r"E:\d\r\7\d\Ablation4_ch016_ep003_7"
    # Process(target = compress_train_code, args = (src_dir, )).start()

    # src_dir = r"E:\d\r\7\d\Ablation4_ch016_ep003_7_10"
    # Process(target = compress_train_code, args = (src_dir, )).start()

    # src_dir = r"E:\d\r\7\d\Ablation4_ch016_ep010"
    # Process(target = compress_train_code, args = (src_dir, )).start()

    # src_dir = r"E:\d\r\7\d\Ablation4_ch016_ep010_relu"
    # Process(target = compress_train_code, args = (src_dir, )).start()

    # src_dir = r"E:\d\r\7\d\Ablation4_good_ch016_ep010_wiwoM_Chbg"
    # Process(target = compress_train_code, args = (src_dir, )).start()


    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\compare_with_before"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_0side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_1side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_2side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_3side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_4side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_5side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_0s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_1s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_2s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_3s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_4s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_5s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_0side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_1side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_2side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_3side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_4side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_5side"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_0s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_1s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_2s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_3s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_4s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_0s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_1s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_2s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_3s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_0s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_1s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_2s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_3s"
    # Process(target = compress_train_code, args = (src_dir, )).start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_4s"
    # Process(target = compress_train_code, args = (src_dir, )).start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_4s"
    # Process(target = compress_train_code, args = (src_dir, )).start()


    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_2s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_2s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_3s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_3s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_3s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_4s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_4s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_4s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_4s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_4s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start(); p.join()



    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_4s\bce_s001_tv_s0p1_L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_4s\bce_s001_tv_s0p1_L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L8"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_1side\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_1side\bce_s001_tv_s0p1_L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_1side\bce_s001_tv_s0p1_L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_2side\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_2side\bce_s001_tv_s0p1_L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_2side\bce_s001_tv_s0p1_L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_3side\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_3side\bce_s001_tv_s0p1_L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_3side\bce_s001_tv_s0p1_L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_4side\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_4side\bce_s001_tv_s0p1_L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_5side\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_1s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_1s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_1s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_1s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_1s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_0s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_0s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_0s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_0s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_0s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_1s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_1s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_1s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_1s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_1s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start(); p.join()




    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_0s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_0s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_0s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_0s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_0s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_1s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_1s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_1s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_1s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_1s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_0s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_0s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_0s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_0s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_0s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_1s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_1s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_1s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_1s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_1s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop255_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop255_pad60_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k15_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k15_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k25_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k25_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k35_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k35_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start(); p.join()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop255_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop255_pad60_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k15_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k15_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k25_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k25_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k35_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k35_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\sob_s001_erose_M\pyr_Tcrop256_p20_j15\pyr_0s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Mae_s001\pyr_Tcrop255_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Mae_s001\pyr_Tcrop255_pad60_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start(); p.join()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k05_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k05_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k15_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k15_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k25_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k25_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k35_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k35_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Mae_s001\pyr_Tcrop255_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Mae_s001\pyr_Tcrop255_pad60_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k05_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k05_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k15_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k15_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k25_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k25_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k35_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k35_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k15_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k15_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k25_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k25_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k35_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k35_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start(); p.join()


    # src_dir = r"F:\data_dir\result\Exps_7_v3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()


    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_0s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_0s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_0s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_0s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_1s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_1s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_1s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_3s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_0s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_0s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_0s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_0s\L6"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_1s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_1s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_1s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_2s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_2s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_2s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_3s\L3"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_3s\L4"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_3s\L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\8\I\pyramid_4side\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\8\I\pyramid_4side\bce_s001_tv_s0p1_L7"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
    src_dir = r"F:\data_dir\result\8\I\pyramid_5side\bce_s001_tv_s0p1_L5"; p = Process(target = compress_train_code, args = (src_dir, )); p.start()
