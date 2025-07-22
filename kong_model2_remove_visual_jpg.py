from multiprocessing import Process
import os
from tqdm import tqdm
import time


def remove_visual_jpg(src_dir):
    start_time = time.time()
    print("src_dir:", src_dir)

    dir_paths = [src_dir + "\\" + dir_name for dir_name in os.listdir(src_dir) if "ana" not in dir_name ]
    for dir_path in tqdm(dir_paths):
        print("dir_path:", dir_path)
        see_dir_paths = []
        for go_see in range(11):
            see_dir_path = dir_path + "\\see_%03i" % go_see
            if(go_see == 0 or go_see == 5 or go_see == 6 or go_see == 7):
                continue
            if  (go_see <= 4 ):
                see_dir_path += "-real"
            elif(go_see <= 8 ):
                see_dir_path += "-train"
            elif(go_see <= 10 ):
                see_dir_path += "-test"
            see_dir_paths.append(see_dir_path)

        for see_dir_path in tqdm(see_dir_paths):
            file_names = os.listdir(see_dir_path)

            for file_name in tqdm(file_names):
                if("visual.jpg" in file_name or "epoch_0" in file_name):
                    visual_jpg_file_path = see_dir_path + "\\" + file_name
                    os.remove(visual_jpg_file_path)

    print("cost_time:", time.time() - start_time)


''' src_dir 要進到 pyr_xs 內 的 Lx 才夠深喔'''
# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep003_7__Remove_see_xxx_visual_jpg\I_w_M_to_W_pyr\pyr_3s\L5"
# remove_visual_jpg(src_dir)
# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep003_7__Remove_see_xxx_visual_jpg\W_w_M_to_C_pyr\pyr_2s\L5"
# remove_visual_jpg(src_dir)

# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep003_7_10__Remove_see_xxx_visual_jpg\I_w_M_to_W_pyr\pyr_3s\L5"
# remove_visual_jpg(src_dir)
# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep003_7_10__Remove_see_xxx_visual_jpg\W_w_M_to_C_pyr\pyr_2s\L5"
# remove_visual_jpg(src_dir)

# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep010__Remove_see_xxx_visual_jpg\I_w_M_to_W_pyr\pyr_3s\L5"
# remove_visual_jpg(src_dir)
# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep010__Remove_see_xxx_visual_jpg\W_w_M_to_C_pyr\pyr_2s\L5"
# remove_visual_jpg(src_dir)

# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep010_relu__Remove_see_xxx_visual_jpg\I_w_M_to_W_pyr\pyr_2s\L5"
# remove_visual_jpg(src_dir)
# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep010_relu__Remove_see_xxx_visual_jpg\I_w_M_to_W_pyr\pyr_3s\L5"
# remove_visual_jpg(src_dir)
# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep010_relu__Remove_see_xxx_visual_jpg\W_w_M_to_C_pyr\pyr_2s\L5"
# remove_visual_jpg(src_dir)

### 這個大
# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep003__Remove_see_xxx_visual_jpg\I_w_M_to_W_pyr\pyr_2s\L5"
# remove_visual_jpg(src_dir)
# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep003__Remove_see_xxx_visual_jpg\I_w_M_to_W_pyr\pyr_3s\L5"
# remove_visual_jpg(src_dir)
# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep003__Remove_see_xxx_visual_jpg\W_w_M_to_C_pyr\pyr_2s\L5"
# remove_visual_jpg(src_dir)
# src_dir = r"F:\d\r\7\d\Ablation4_ch016_ep003__Remove_see_xxx_visual_jpg\W_w_M_to_C_pyr\pyr_3s\L5"
# remove_visual_jpg(src_dir)


if __name__ == "__main__":
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_1side\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_1side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_1side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_2side\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_2side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_2side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_3side\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_3side\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_3side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_3side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_4side\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_4side\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_4side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_5side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_0s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_0s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_0s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_0s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_1s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_1s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_1s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_1s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_2s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_2s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_2s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_2s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_3s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_3s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_3s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_4s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_4s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_4s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_5s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_size256\pyr_5s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_0side\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_0side\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_0side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_0side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_0side\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_1side\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_1side\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_1side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_1side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_2side\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_2side\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_2side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_2side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_3side\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_3side\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_3side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_4side\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_4side\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_4side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_5side\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256\pyramid_5side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_2s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_2s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_2s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_2s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_2s\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_3s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_3s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_3s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad20_jit15\pyr_4s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_0s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_0s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_0s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_0s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_1s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_1s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_1s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_2s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_2s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_2s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_3s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_3s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_3s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_4s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_4s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60\pyr_4s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_to_M_Gk3_no_pad\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()




    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_0s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_1s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_2s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_2s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_3s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_3s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_3s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_4s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad20_jit15\pyr_4s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_4s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_4s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_C_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_4s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_0s\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_1s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_2s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_3s\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_4s\bce_s001_tv_s0p1_L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\I_w_M_to_W_focus\pyramid_tight_crop_size256_pad60_jit15\pyr_4s\bce_s001_tv_s0p1_L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_0s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_0s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_0s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_0s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_1s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_1s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_1s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_3s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad60_jit15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad60_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad60_jit15\pyr_3s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad60_jit15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus\pyr_Tcrop256_pad60_jit15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_0side\bce_s001_tv_s0p1_L8"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_1side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_1side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_1side\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_2side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_2side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_2side\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_3side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_3side\bce_s001_tv_s0p1_L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_3side\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_4side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_4side\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_8_v3\I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad\pyramid_5side\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_1s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_1s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_1s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_1s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad20_jit15\pyr_1s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_0s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_0s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_0s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_0s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_0s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_1s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_1s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_1s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_1s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\mae_s001\pyr_Tcrop256_pad60_jit15\pyr_1s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s001_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s010_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus\Sob_s100_erose_M\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_0s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_0s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_0s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_0s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_0s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_1s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_1s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_1s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_1s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad20_jit15\pyr_1s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_0s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_0s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_0s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_0s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_0s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_1s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_1s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_1s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_1s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_to_Wz_focus_BN\pyr_Tcrop256_pad60_jit15\pyr_1s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop255_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop255_pad60_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k15_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k15_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k25_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k25_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k35_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k35_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_only\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop255_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop255_pad60_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_0s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_2s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Mae_s001\pyr_Tcrop256_pad20_jit15\pyr_3s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_2s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM\pyr_Tcrop256_p20_j15\pyr_3s\L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k15_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k15_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k25_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k25_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k35_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k35_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroM\Sob_Wxy\sob_s001_erose_M\pyr_Tcrop256_p20_j15\pyr_0s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Mae_s001\pyr_Tcrop255_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Mae_s001\pyr_Tcrop255_pad60_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k05_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k05_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k15_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k15_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k25_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k25_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k35_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k5_EroMore\Sob_k35_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Mae_s001\pyr_Tcrop255_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Mae_s001\pyr_Tcrop255_pad60_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k05_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k05_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k15_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k15_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k25_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k25_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k35_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\IN_Sob_k15_EroM\Sob_k35_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k05_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k15_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k25_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k35_s001_EroM_Mae_s001\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k15_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k15_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k25_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k25_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k35_s001_EroM\pyr_Tcrop255_p20_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\doc3d\Wyx_w_M_w_Sob_to_Wz_focus\Sob_Wxy\Sob_k35_s001_EroM\pyr_Tcrop255_p60_j15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start(); p.join()

    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_0s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_0s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_0s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_0s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_1s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_1s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_1s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_3s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad20_jit15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_0s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_0s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_0s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_0s\L6"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_1s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_1s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_1s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_2s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_2s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_2s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_3s\L3"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_3s\L4"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\Exps_7_v3\W_w_Mgt_to_Cx_Cy_focus_raring\pyr_Tcrop256_pad60_jit15\pyr_3s\L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    

    # src_dir = r"F:\data_dir\result\8\I\4\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    # src_dir = r"F:\data_dir\result\8\I\4\bce_s001_tv_s0p1_L7"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()
    src_dir = r"F:\data_dir\result\8\I\5\bce_s001_tv_s0p1_L5"; p = Process(target = remove_visual_jpg, args = (src_dir, )); p.start()