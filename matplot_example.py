import matplotlib.pyplot as plt
import numpy as np

def example_subplots_combine():
    ### https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html

    fig, axs = plt.subplots(ncols=3, nrows=3)
    gs = axs[1, 2].get_gridspec()  ### 取得 整張圖的 grid範圍規格表 我猜

    ### 發現其實 ax[不管在哪邊].get_gridspec()都是一樣的呀~~
    ### 所以應該是隨便指定 ax 的某個位置，取得 整張圖的 grid範圍規格表 我猜
    for go_r, r_ax in enumerate(axs):
        for go_c, c_ax in enumerate(r_ax):
            print(f"gs == gs[{go_r}, {go_c}]:", gs == axs[go_r, go_c].get_gridspec())
            print(axs[go_r, go_c].get_gridspec()[go_r, go_c])
    print(axs[0, 0].get_gridspec()[1:, -1])

    # remove the underlying axes
    ### 這裡remove 的 用途是讓 這些圖 不要顯示出來，才不會蓋過新圖
    for ax in axs[1:, -1]:
        ### https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.axes.Axes.remove.html
        ### ax.remove() 不是真的刪掉 ax，是讓 ax 不顯示出來！
        ax.remove()

    ### 在 grid範圍規格表 指定的位置 畫上新圖
    axbig = fig.add_subplot(gs[1:, -1])
    axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5), xycoords='axes fraction', va='center')

    fig.tight_layout()

    fig2, axs2 = plt.subplots(ncols=3, nrows=3)
    plt.show()


def example_subplots_adjust():
    x = np.linspace(-3, 3, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = 1 / (1 + np.exp(-x))
    y4 = np.exp(x)

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].plot(x, y1)
    ax[0, 1].plot(x, y2)
    ax[0, 2].plot(x, y2)
    ax[1, 0].plot(x, y3)
    ax[1, 1].plot(x, y4)
    ax[1, 2].plot(x, y4)

    ax[0, 0].set_title("Sine function")
    ax[0, 1].set_title("Cosine function")
    ax[0, 2].set_title("Cosine function")
    ax[1, 0].set_title("Sigmoid function")
    ax[1, 1].set_title("Exponential function")
    ax[1, 2].set_title("Exponential function")

    # fig.tight_layout()  ### 補充一下 tight_layout 加這裡沒有用，因為這裡tight完，下面就又改成用 adjust 的方式排版囉！
    plt.subplots_adjust(left=0.01,         ### 左邊界 到 圖的距離 是 幾%
                        bottom=0.01,       ### 下邊界 到 圖的距離 是 幾%
                        right=1.0 - 0.01,  ### 右邊界 到 圖的距離 是 幾%
                        top=1.0 - 0.1,     ### 上邊界 到 圖的距離 是 幾%
                        wspace=0.01,       ### 圖左右之間的 寬
                        hspace=0.01        ### 圖左右之間的 高
                        )
    plt.show()

def example_subplots_change_axes_size_ratio():
    ### https://stackoverflow.com/questions/53521778/matplotlib-set-subplot-axis-size-iteratively
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import cv2

    this_py_path = "C:/Users/TKU/Desktop/kong_model2/kong_util"
    img1 = cv2.imread(f"{this_py_path}/img_data/0a-in_img.jpg")
    img2 = cv2.imread(f"{this_py_path}/img_data/0b-gt_a_gt_flow.jpg")
    img3 = cv2.imread(f"{this_py_path}/img_data/epoch_0000_a_flow_visual.jpg")

    ### step1 先架構好 整張圖的骨架
    # create figure
    f, ax = plt.subplots(3, 1, figsize=(10, 10))

    ### step2 把圖都畫上去
    # plot some data
    # ax[0].plot([1, 2, 3])
    # ax[1].plot([1, 0, 1])
    # ax[2].plot([1, 2, 20])
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[2].imshow(img3)

    ### step3 重新規劃一下 各個圖 要顯示的 大小比例
    # adjust subplot sizes
    gs = GridSpec(3, 1, height_ratios=[5, 2, 1])
    print("gs:", gs)
    for i in range(3):
        print(f"gs[{i}]:", gs[i])  ### 可以看到 目前使用的規格的範圍
        print(f"gs[{i}].get_position(f):", gs[i].get_position(f))  ### 可以看到 目前取用的這個的規格的範圍 對應到 f 上 框出的box是在哪裡
        ax[i].set_position(gs[i].get_position(f))  ### 根據目前的圖(f)， 重新規劃一下 各個圖 要顯示的 大小比例

    plt.show()

def example_subplots_ax_and_axes():
    from matplotlib import pyplot as plt
    ### 沒有 dim
    fig, ax = plt.subplots(nrows=1, ncols=1)
    print(ax)  ### AxesSubplot(0.125,0.11;0.775x0.77)

    ### dim=1，不管 直的 或 橫的 都是如此， 不會說 直的 就變 dim=2！
    fig, ax = plt.subplots(nrows=2, ncols=1)
    print(ax)  ### [<AxesSubplot:> <AxesSubplot:>]
    fig, ax = plt.subplots(nrows=1, ncols=2)
    print(ax)  ### [<AxesSubplot:> <AxesSubplot:>]

    ### dim=2
    fig, ax = plt.subplots(nrows=2, ncols=2)
    print(ax)
    ### [[<AxesSubplot:> <AxesSubplot:>]
    ###  [<AxesSubplot:> <AxesSubplot:>]]


def example_pure_img():
    ### https://gist.github.com/zhou13/b4ee8e815aee83e88df5b865896aaf5a
    from matplotlib import pyplot as plt
    import cv2
    this_py_path = "C:/Users/TKU/Desktop/kong_model2/kong_util"
    img1 = cv2.imread(f"{this_py_path}/img_data/0a-in_img.jpg")

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(7.68, 7.68)

    ### 關閉 軸
    ax.axis('off')


    ax.imshow(img1)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  ### 可參考上面 subplots_adjust 的例子

    plt.show()
    # plt.savefig(f"filename.png")  ### 如何存

def example_close_axes():
    ### 關閉 軸 的 三個方法
    fig, ax = plt.subplots(nrows=4, ncols=1)
    ### 原始 有軸的拿來對比
    ax[0].plot(range(10), 'k-')

    #######################################################
    ### 方法1
    ax[1].plot(range(10), 'b-')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    #######################################################
    ### 方法2： 連框框都拿掉
    ax[2].plot(range(10), 'r-')
    ax[2].axis('off')
    #######################################################
    ### 方法3
    ax[3].plot(range(10), 'g-')
    ax[3].xaxis.set_major_locator(plt.NullLocator())
    ax[3].yaxis.set_major_locator(plt.NullLocator())

    ### 圖很緊密的靠在一起
    fig.tight_layout()
    plt.show()


if(__name__ == "__main__"):
    pass
