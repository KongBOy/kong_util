### 參考網頁： https://towardsdatascience.com/basics-of-gifs-with-pythons-matplotlib-54dd544b6f30
import imageio
import os
from build_dataset_combine import Check_dir_exist_and_build


def Build_gif_from_dir(ord_dir=".", dst_dir=".", gif_name="myGif.gif"):
    Check_dir_exist_and_build(dst_dir)

    file_names = os.listdir(ord_dir)
    # build gif
    with imageio.get_writer( dst_dir + "/" + gif_name + '.gif', mode='I') as writer:
        for file_name in file_names:
            image = imageio.imread(ord_dir + "/" + file_name)
            writer.append_data(image)
