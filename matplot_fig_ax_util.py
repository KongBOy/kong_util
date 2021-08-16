from util import get_dir_certain_file_name, method1
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import sys
LOSS_YLIM = 2.0

def get_cmap(color_amount, cmap_name='hsv'):
    '''Returns a function that maps each index in 0, 1,.. . N-1 to a distinct
    RGB color.
    '''
    color_norm = colors.Normalize(vmin=0, vmax=color_amount - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap_name)

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

############################################################################################################
############################################################################################################
class Matplot_ax_util():
    @staticmethod
    def Draw_ax_loss_during_train( ax, logs_read_dir, cur_epoch, epochs , ylim=LOSS_YLIM ):  ### logs_read_dir 不能改丟 result_obj喔！因為See裡面沒有Result喔！
        x_epoch = np.arange(cur_epoch + 1)  ### x座標畫多少，畫到目前訓練的 cur_epoch，+1是為了index轉數量喔

        logs_file_names = get_dir_certain_file_name(logs_read_dir, "npy")  ### 去logs_dir 抓 當時訓練時存的 loss.npy
        for loss_i, logs_file_name in enumerate(logs_file_names):
            y_loss_array = np.load( logs_read_dir + "/" + logs_file_name)  ### 去logs_dir 抓 當時訓練時存的 loss.npy
            loss_name = logs_file_name.split(".")[0]
            Matplot_ax_util._Draw_ax_loss(ax, cur_epoch, loss_name, loss_i, x_array=x_epoch, y_array=y_loss_array, xlim=epochs, ylim=ylim)


    @staticmethod
    ### 注意這會給 see, result, c_results 用喔！ 所以多 result的情況也要考慮，所以才要傳 min_epochs，
    ### 且因為有給see用，logs_dir 不能改丟 result_obj喔！因為See裡面沒有Result喔！
    def Draw_ax_loss_after_train( ax, logs_read_dir, cur_epoch, min_epochs , ylim=LOSS_YLIM ):
        x_epoch = np.arange(min_epochs)  ### x座標畫多少
        logs_file_names = get_dir_certain_file_name(logs_read_dir, "npy")  ### 去logs_dir 抓 當時訓練時存的 loss.npy
        for loss_i, logs_file_name in enumerate(logs_file_names):
            y_loss_array = np.load( logs_read_dir + "/" + logs_file_name)  ### 把loss讀出來
            loss_amount = len(y_loss_array)                           ### 訓練的當下存了多少個loss
            if( (min_epochs - 1) == loss_amount):                     ### 如果現在result剛好是訓練最少次的result，要注意有可能訓練時中斷在存loss前，造成 epochs數 比 loss數 多一個喔！這樣畫圖會出錯！
                y_loss_array = np.append(y_loss_array, y_loss_array[-1])  ### 把loss array 最後補一個自己的尾巴
            y_loss_array_used = y_loss_array[:min_epochs]             ### 補完後，別忘了考慮多result的情況，result裡挑最少量的的loss數量 來show
            loss_name = logs_file_name.split(".")[0]

            # print("len(x_epoch)", len(x_epoch))
            # print("len(y_loss_array_used)", len(y_loss_array_used))
            Matplot_ax_util._Draw_ax_loss(ax, cur_epoch, loss_name, loss_i, x_array=x_epoch, y_array=y_loss_array_used, xlim=min_epochs, ylim=ylim)

    @staticmethod
    def _Draw_ax_loss(ax, cur_epoch, loss_name, loss_i, x_array, y_array, xlim, ylim=LOSS_YLIM, x_label="epoch loss avg", y_label="epoch_num"):
        cmap = get_cmap(8)  ### 隨便一個比6多的數字，嘗試後8的顏色分布不錯！
        plt.sca(ax)  ### plt指向目前的 小畫布 這是為了設定 xylim 和 xylabel
        plt.ylim(0, ylim); plt.ylabel( x_label )
        plt.xlim(0, xlim); plt.xlabel( y_label )

        ### 畫線
        ax.plot(x_array, y_array, c=cmap(loss_i), label=loss_name)
        ### 畫點
        ax.scatter(cur_epoch, y_array[cur_epoch], color=cmap(loss_i))
        ### 點旁邊註記值
        ax.annotate( text="%.3f" % y_array[cur_epoch],    ### 顯示的文字
                     xy=(cur_epoch, y_array[cur_epoch]),  ### 要標註的目標點
                     xytext=( 0 , 10 * loss_i),         ### 顯示的文字放哪裡
                     textcoords='offset points',         ### 目前東西放哪裡的坐標系用什麼
                     arrowprops=dict(arrowstyle="->",    ### 畫箭頭的資訊
                                    connectionstyle= "arc3",
                                    color = cmap(loss_i),))
        ax.legend(loc='best')


class Matplot_fig_util(Matplot_ax_util):
    @staticmethod
    def Save_fig(dst_dir, epoch, epoch_name="epoch"):
        """
        存的長相是：dst_dir/{epoch_name}={epoch}.png
        存完會自動關閉 fig
        """
        plt.savefig(dst_dir + "/" + "%s=%04i" % (epoch_name, epoch) )
        plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！

class Matplot_util(Matplot_fig_util): pass
##########################################################################################################################################################
##########################################################################################################################################################



class Matplot_single_row_imgs(Matplot_util):
    def __init__(self, imgs, img_titles, fig_title, pure_img=False, bgr2rgb=False, add_loss=False):
        self.imgs       = imgs  ### imgs是個list，裡面放的圖片可能不一樣大喔
        self.img_titles = img_titles
        self.fig_title  = fig_title
        self.pure_img   = pure_img
        self.bgr2rgb    = bgr2rgb

        self.add_loss   = add_loss

        self.fig_row_amount   = 1
        self.fig_col_amount   = len(self.imgs)
        self.ax_titles_amount = len(self.img_titles)
        self._step0_a_build_check()


        self.canvas_height = None
        self.canvas_width  = None
        self.canvas_1_ax_h = None
        self.canvas_1_ax_w = None
        self.fig = None
        self.ax  = None
        self._step0_b_set_canvas_hw_and_build()

        self.first_time_row_col_finish = False
        self.merged_ax_list = []
        self.merged_gs_list = []

    def _step0_a_build_check(self):
        #### 防呆 ####################################################
        ### 正常來說 一個 title 對應 一張圖
        if( self.ax_titles_amount < self.fig_col_amount):  ### 如果 title數 比 影像數多，那就用 空title來補
            for _ in range(self.fig_col_amount - self.ax_titles_amount):
                self.img_titles.append("")

        elif(self.ax_titles_amount > self.fig_col_amount):
            print("title 太多了，沒有圖可以對應")
            return

        if(self.fig_col_amount == 0):
            print("沒圖可show喔！")
            return

    def _step0_b_get_one_row_canvas_height(self):
        height_list = []    ### imgs是個list，裡面放的圖片可能不一樣大喔
        for img in self.imgs: height_list.append(img.shape[0])
        if(self.pure_img): return  (max(height_list) / 100 + 0.0) * 1.00  ### 純影像 沒有任何其他東西
        else:              return  (max(height_list) / 100 + 0.0) * 1.15  ### 1.15 就慢慢試出來的囉～因為除了圖以外 還會有旁邊軸的標籤 和 margin也會被算進圖的大小裡， 所以要算比原圖大一點 才能讓show出的影像跟原始影像差不多大

    def _step0_b_get_one_row_canvas_width(self):
        width = 0
        for img in self.imgs: width += img.shape[1]
        if(self.pure_img): return  (width / 100 + 0) * 1.00  ### 純影像 沒有任何其他東西
        else:              return  (width / 100 + 0) * 1.15  ### 1.15 就慢慢試出來的囉～因為除了圖以外 還會有旁邊軸的標籤 和 margin也會被算進圖的大小裡， 所以要算比原圖大一點 才能讓show出的影像跟原始影像差不多大 col=1時
        # if  (self.fig_col_amount == 1): return  (width // 100 + 0) * 1.15  ### 1.1 就慢慢試出來的囉～因為除了圖以外 還會有旁邊軸的標籤 和 margin也會被算進圖的大小裡， 所以要算比原圖大一點 才能讓show出的影像跟原始影像差不多大 col=1時
        # elif(self.fig_col_amount == 2): return  (width // 100 + 0) * 1.15  ### 1.1 就慢慢試出來的囉～因為除了圖以外 還會有旁邊軸的標籤 和 margin也會被算進圖的大小裡， 所以要算比原圖大一點 才能讓show出的影像跟原始影像差不多大 col=2時
        # elif(self.fig_col_amount == 3): return  (width // 100 + 0) * 1.15  ### 1.1 就慢慢試出來的囉～因為除了圖以外 還會有旁邊軸的標籤 和 margin也會被算進圖的大小裡， 所以要算比原圖大一點 才能讓show出的影像跟原始影像差不多大 col=3時
        # elif(self.fig_col_amount == 4): return  (width // 100 + 0) * 1.15  ### 1.1 就慢慢試出來的囉～因為除了圖以外 還會有旁邊軸的標籤 和 margin也會被算進圖的大小裡， 所以要算比原圖大一點 才能讓show出的影像跟原始影像差不多大 col=4時
        # elif(self.fig_col_amount == 5): return  (width // 100 + 0) * 1.15  ### 1.1 就慢慢試出來的囉～因為除了圖以外 還會有旁邊軸的標籤 和 margin也會被算進圖的大小裡， 所以要算比原圖大一點 才能讓show出的影像跟原始影像差不多大 col=5時
        # elif(self.fig_col_amount == 6): return  (width // 100 + 0) * 1.15  ### 1.1 就慢慢試出來的囉～因為除了圖以外 還會有旁邊軸的標籤 和 margin也會被算進圖的大小裡， 所以要算比原圖大一點 才能讓show出的影像跟原始影像差不多大 col=6時
        # elif(self.fig_col_amount == 7): return  (width // 100 + 0) * 1.15  ### 1.1 就慢慢試出來的囉～因為除了圖以外 還會有旁邊軸的標籤 和 margin也會被算進圖的大小裡， 所以要算比原圖大一點 才能讓show出的影像跟原始影像差不多大 col=7時
        # elif(self.fig_col_amount >  7): return  (width // 100 + 0) * 1.15  ### 1.1 就慢慢試出來的囉～因為除了圖以外 還會有旁邊軸的標籤 和 margin也會被算進圖的大小裡， 所以要算比原圖大一點 才能讓show出的影像跟原始影像差不多大 col=7時

    def _step0_b_set_canvas_hw_and_build(self):
        ### 設定canvas的大小
        self.canvas_height = self._step0_b_get_one_row_canvas_height()
        self.canvas_width  = self._step0_b_get_one_row_canvas_width()
        self.canvas_1_ax_h = self.canvas_height / self.fig_row_amount
        self.canvas_1_ax_w = self.canvas_width  / self.fig_col_amount
        if(self.add_loss):   ### 多一些空間來畫loss
            self.fig_row_amount += 1  ### 多一row來畫loss
            self.canvas_height   += self.canvas_1_ax_h    ### 慢慢試囉～
            # self.canvas_width  -= 1.5 * self.fig_col_amount  ### 慢慢試囉～
        # print("canvas_height",   self.canvas_height)
        # print("canvas_width ",   self.canvas_width)
        # print("canvas_1_ax_h ",  self.canvas_1_ax_h)
        # print("canvas_1_ax_w ",  self.canvas_1_ax_w)
        # print("fig_row_amount", self.fig_row_amount)
        # print("fig_col_amount", self.fig_col_amount)

        ### 建立canvas出來
        self.fig, self.ax = plt.subplots(nrows=self.fig_row_amount, ncols=self.fig_col_amount)
        self.fig.set_size_inches(self.canvas_width, self.canvas_height)  ### 設定 畫布大小

    def step1_add_row_col(self, add_where="", merge=True, grid_ratio=1):
        if(add_where == "add_row"):
            self.fig_row_amount += 1
            self.canvas_height   += self.canvas_1_ax_h * grid_ratio
        elif(add_where == "add_col"):
            self.fig_col_amount += 1
            self.canvas_width   += self.canvas_1_ax_w * grid_ratio

        fig_new, ax_new = plt.subplots(nrows=self.fig_row_amount, ncols=self.fig_col_amount)  ### 新畫一張 加一個row的 新大圖
        fig_new.set_size_inches(self.canvas_width, self.canvas_height)            ### 設定 新大圖 的 畫布大小
        gs_new = ax_new[0, 0].get_gridspec()                                      ### 取得 新大圖 的 grid範圍規格表

        if(merge):  ### 如果新增的 row/col 是想合併的樣式
            if  (add_where == "add_row"):
                merge_grid_gs =         gs_new[-1, :]                             ### 取得 新大圖 的 最後一row整個 的 grid範圍資訊
                for ax_new_final_row in ax_new[-1, :]: ax_new_final_row.remove()  ### 新大圖 的最後一row 取消顯示
            elif(add_where == "add_col"):
                merge_grid_gs =         gs_new[:, -1]                             ### 取得 新大圖 的 最後一col整個 的 grid範圍資訊
                for ax_new_final_col in ax_new[:, -1]: ax_new_final_col.remove()  ### 新大圖 的最後一col 取消顯示

            ax_new_merge = fig_new.add_subplot(merge_grid_gs)                     ### 根據 最後一row/col整個 的 grid範圍資訊 貼上一張新 subplots

            self.merged_ax_list.append(ax_new_merge)                              ### 把 ax merge資訊存起來
            self.merged_gs_list.append(merge_grid_gs)                             ### 把 gs merge範圍存起來

        ### 因為 fig_new 是 新的subplots圖，第二次以上做 merge 的話，第二次以前的結果 都會被新subplots圖蓋掉，所以要對 新subplots圖 把 之前的 merge過的動作都重做一次喔！
        if(self.first_time_row_col_finish):                        ### 等於1 是第一次執行，不需同步，第一次以後才需要同步囉
            for go_pass, pass_gs in enumerate(self.merged_gs_list[:-1]):  ### 走訪到 最新的 merged_gs 以前
                # print("here~~~~~~~~~~~~~~~")
                self.merged_ax_list[go_pass] = self._syn_with_pass_merged_grid(pass_gs, fig_new, ax_new, gs_new)

        plt.close(self.fig)                                                   ### 把舊圖關掉
        self.fig = fig_new
        self.ax  = ax_new
        if(merge): self.first_time_row_col_finish = True

    def _syn_with_pass_merged_grid(self, pass_gs, fig_new, ax_new, gs_new):
        """
        pass_gs： 之前的 merge 範圍資訊
        fig_new： 目前的 新 subplots圖 的 fig
        ax_new：  目前的 新 subplots圖 的 ax
        gs_new：  目前的 新 subplots圖 的 grid範圍資訊
        """
        ### 把過去的 merge 的 上下左右 抓出來
        t = pass_gs.rowspan.start; d = pass_gs.rowspan.stop
        l = pass_gs.colspan.start; r = pass_gs.colspan.stop

        ### 過去 已merge的ax位置 相對應於 新ax的哪裡，相當於pass_update的概念
        pass_gs_update = gs_new[t:d, l:r]
        # print("ax_new", ax_new)
        # print("pass_gs", pass_gs)
        ### 新大圖上 把 之前合併的grid 取消顯示
        for go_r in pass_gs.rowspan:
            for go_c in pass_gs.colspan:
                # print(go_r, go_c, "remove()")
                ax_new[go_r, go_c].remove()

        ### 把 過去已merge 的地方 根據 對應的 新ax 補起來
        pass_ax_update = fig_new.add_subplot(pass_gs_update)
        return pass_ax_update


    def _step3_draw(self, used_ax):
        '''
        used_ax 只接受 1r1c 和 1r多c 喔， 不接受 多r多c( 會取第1row)
        '''
        ### 這就是手動微調 text的位置囉ˊ口ˋ
        ### (0.5 / self.canvas_height) 的意思是 我想留 50px 左右 給上方
        self.fig.text(x=0.5, y= 1 - (0.5 / self.canvas_height), s=self.fig_title, fontsize=28, c=(0., 0., 0., 1.),  horizontalalignment='center',)

        for go_img, img in enumerate(self.imgs):
            if(self.bgr2rgb): img = img[..., ::-1]  ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
            if(self.fig_col_amount == 1):  used_ax = [used_ax]  ### 把 (1r1c 的) ax 包成 (1r多c 的) ax[...]， 能確定是 1r1c 因為能來到這個 method 就代表 是 1r1c 或 1r多c， 而又 如果 fig_col_amount==1 就代表一錠是 1r1c 的 case 囉！

            ### 因為上面有包成 ax[...]，以下統一用 ax[...] 的方式來處理囉！ 就不用 多寫一個if/else來區分 ax/ax[...] 不同的操作方式了！
            if(not self.pure_img):
                used_ax[go_img].imshow(img)  ### 小畫布 畫上影像，別忘記要bgr -> rgb喔！
                used_ax[go_img].set_title( self.img_titles[go_img], fontsize=16 )  ### 小畫布上的 title
                used_ax[go_img].set_yticks( (0, img.shape[0]) )   ### 設定 y軸 顯示的字，tuple是要顯示的數字， 目前是顯示 0 和 h
                used_ax[go_img].set_xticks( (0, img.shape[1]) )   ### 設定 x軸 顯示的字，tuple是要顯示的數字
            else:  ### 目前的 pure_img 只有給 SIFT_d 來用， 所以就先針對他 來 設計要怎麼 show 圖囉！
                ax_img = used_ax[go_img].imshow(img, vmin=0, vmax=50)
                used_ax[go_img].set_yticks(())  ### 設定 y軸 不顯示字
                used_ax[go_img].set_xticks(())  ### 設定 x軸 不顯示字
                cax = plt.axes([0.85, 0.1, 0.010, 0.2])  ### 左下角x, 左下角y, w是整張圖的幾%, h是整張圖的幾%
                self.fig.colorbar(ax_img, ax=used_ax[go_img], cax=cax)

    def Draw_img(self):  ### 呼叫 _step3_draw 畫圖
        ###############################################################
        ### 注意 _draw_single_row_imgs 的 ax 只能丟 一row，所以才寫這if/else
        # if(not self.add_loss): used_ax = self.ax
        # elif(self.add_loss):   used_ax = self.ax[0]  ### 只能丟第一row喔！因為_draw_single_row_imgs 裡面的操作方式 是 一row的方式，丟兩row ax維度會出問題！
        used_ax = self.ax  ### 先假設 為 1r1c
        if(type(self.ax) == type(np.array(1))):            ### 如果是np.array的形式，就不是1r1c， 而有可能是 1r多c 或 多r多c
            if(  self.ax.ndim == 1): used_ax = self.ax     ### 1r多c，沒問題！
            elif(self.ax.ndim  > 1): used_ax = self.ax[0]  ### 多r多c，只能丟第一row喔！因為_draw_single_row_imgs 裡面的操作方式 是 一row的方式，丟兩row ax維度會出問題！ 所以取第0row，就變 1r多c了， 也代表了圖繪從左上角開始畫喔！

        self._step3_draw(used_ax)  ### 只接受 1r1c 和 1r多c 喔， 不接受 多r多c
        ###############################################################
        ### 想畫得更漂亮一點，兩種還是有些一咪咪差距喔~
        if(not self.pure_img):
            if(not self.add_loss): self.fig.tight_layout(rect=[0, 0, 1, 0.93])
            else:                  self.fig.tight_layout(rect=[0, 0.006, 1, 0.95])
        elif(self.pure_img):       self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ###############################################################
        ### Draw_img完，不一定要馬上Draw_loss喔！像是train的時候 就是分開的 1.see(Draw_img), 2.train, 3.loss(Draw_loss)


##########################################################################################################################################################
##########################################################################################################################################################
class Matplot_multi_row_imgs(Matplot_util):
    def __init__(self, rows_cols_imgs, rows_cols_titles, fig_title, bgr2rgb=True, add_loss=False):
        self.r_c_imgs = rows_cols_imgs
        self.r_c_titles = rows_cols_titles
        self.fig_title = fig_title
        self.bgr2rgb = bgr2rgb
        self.add_loss = add_loss

        self.fig_row_amount   = len(self.r_c_imgs)
        self.fig_col_amount   = len(self.r_c_imgs[0])
        self.ax_titles_amount = len(self.r_c_imgs[0])
        self._step1_build_check()

        self.canvas_height     = None
        self.canvas_width      = None
        self.fig = None
        self.ax  = None
        self._step2_set_canvas_hw_and_build()

    def _step1_build_check(self):
        #### 防呆 ####################################################
        if( self.ax_titles_amount < self.fig_col_amount):
            for row_titles in self.r_c_titles:
                for _ in range(self.fig_col_amount - self.ax_titles_amount):
                    row_titles.append("")
        elif(self.ax_titles_amount > self.fig_col_amount):
            print("title 太多了，沒有圖可以對應")
            return

        if(self.fig_col_amount == 0):
            print("沒圖可show喔！")
            return

        if(len(self.r_c_imgs) == 1):
            print("本function 不能處理 single_row_imgs喔，因為matplot在row只有1時的維度跟1以上時不同！麻煩呼叫相對應處理single_row的function！")

    def _get_row_col_canvas_height(self):
        height = 0
        for row_imgs in self.r_c_imgs: height += row_imgs[0].shape[0]
        return (height // 100 + 0) * 1.2  ### 慢慢試囉～ +1.5是要給title 和 matplot邊界margin喔

    def _get_row_col_canvas_width(self):
        width = 0
        for col_imgs in self.r_c_imgs[0]: width += col_imgs.shape[1]
        return (width // 100 + 1) * 1.2  ### 慢慢試囉～

    def _step2_set_canvas_hw_and_build(self):
        ###########################################################
        ### 設定canvas的大小
        self.canvas_height = self._get_row_col_canvas_height()
        self.canvas_width  = self._get_row_col_canvas_width ()
        if(self.add_loss):   ### 多一些空間來畫loss
            self.fig_row_amount += 1  ### 多一row來畫loss
            self.canvas_height += 3.0  ### 慢慢試囉～
            self.canvas_width  -= 0.55 * self.fig_col_amount  ### 慢慢試囉～
            self.canvas_height *= 1.1  #1.2最好，但有點佔記憶體  ### 慢慢試囉～
            self.canvas_width  *= 1.1  #1.2最好，但有點佔記憶體  ### 慢慢試囉～
        # print("canvas_height",canvas_height)
        # print("canvas_width",canvas_width)
        # print("fig_row_amount", fig_row_amount)

        ### 建立canvas出來
        self.fig, self.ax = plt.subplots(nrows=self.fig_row_amount, ncols=self.fig_col_amount)
        self.fig.set_size_inches(self.canvas_width, self.canvas_height)  ### 設定 畫布大小

    def _step3_draw(self):
        '''
        從左上角 ax 開始畫圖，一張圖對一個title這樣子畫
        '''
        ### 這就是手動微調 text的位置囉ˊ口ˋ
        self.fig.text(x=0.5, y=0.95, s=self.fig_title, fontsize=20, c=(0., 0., 0., 1.),  horizontalalignment='center',)

        for go_row, row_imgs in enumerate(self.r_c_imgs):
            for go_col, col_img in enumerate(row_imgs):
                if(self.bgr2rgb): col_img = col_img[..., ::-1]  ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
                if(self.fig_col_amount > 1):
                    self.ax[go_row, go_col].imshow(col_img)  ### 小畫布 畫上影像，別忘記要bgr -> rgb喔！
                    if  (len(self.r_c_titles) > 1): self.ax[go_row, go_col].set_title( self.r_c_titles[go_row][go_col], fontsize=16 )  ### 小畫布　標上小標題
                    elif(len(self.r_c_titles) == 1 and go_row == 0): self.ax[go_row, go_col].set_title( self.r_c_titles[go_row][go_col], fontsize=16 )  ### 小畫布　標上小標題

                    plt.sca(self.ax[go_row, go_col])  ### plt指向目前的 小畫布 這是為了設定 yticks和xticks
                    plt.yticks( (0, col_img.shape[0]), (0, col_img.shape[0]) )   ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                    plt.xticks( (0, col_img.shape[1]), ("", col_img.shape[1]) )  ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                else:  ### 要多這if/else是因為，col_imgs_amount == 1時，ax[]只會有一維！用二維的寫法會出錯！所以才獨立出來寫喔～
                    self.ax[go_row].imshow(col_img)  ### 小畫布 畫上影像
                    if  (len(self.r_c_titles) > 1 ): self.ax[go_row].set_title( self.r_c_titles[go_row][go_col], fontsize=16 )  ### 小畫布　標上小標題
                    elif(len(self.r_c_titles) == 1 and go_row == 0): self.ax[go_row].set_title( self.r_c_titles[go_row][go_col], fontsize=16 )  ### 小畫布　標上小標題
                    plt.yticks( (0, col_img.shape[0]), (0, col_img.shape[0]) )   ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                    plt.xticks( (0, col_img.shape[1]), ("", col_img.shape[1]) )  ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字

    def Draw_img(self):
        self._step3_draw()
        if(not self.add_loss): self.fig.tight_layout(rect=[0, 0, 1, 0.95])  ### 待嘗試喔！
        else:                  self.fig.tight_layout(rect=[0, 0.0035, 1, 0.95])  ### 待嘗試喔！
        ###############################################################
        ### Draw_img完，不一定要馬上Draw_loss喔！但 multi的好像可以馬上Draw_loss~ 不過想想還是general一點分開做好了~~






def draw_loss_util(fig, ax, logs_read_dir, epoch, epochs ):
    x_epoch = np.arange(epochs)

    logs_file_names = get_dir_certain_file_name(logs_read_dir, "npy")
    y_loss_array = np.load( logs_read_dir + "/" + logs_file_names[0])

    plt.sca(ax)  ### plt指向目前的 小畫布 這是為了設定 xylim 和 xylabel
    plt.ylim(0, LOSS_YLIM)   ; plt.ylabel(logs_file_names[0])
    plt.xlim(0,  epochs)     ; plt.xlabel("epoch_num")
    ax.plot(x_epoch, y_loss_array)
    ax.scatter(epoch, y_loss_array[epoch], c="red")
    return fig, ax


### 偷偷畫一個透明方形 來達到 xlim, ylim 的效果
XLIM = 1.0
YLIM = 1.0

def change_into_3D_coord_ax(fig, ax, ax_c, ax_r=0, ax_rows=1, fig_title=None,
            xlabel="x", ylabel="y", zlabel="z",
            y_flip=False, y_coord=None, tight_layout=True):
    '''
    注意！ 如果有多row， 只能丟 1 row 近來喔
    fig   ： 要加入 3D 圖的 fig
    ax    ： 因為要隱藏 2D的子圖，所以需要這個，fig.axes[...].remove() 刪除好像就真的整個刪掉了會出事 不要用比較好
    ax_i  ： 想加 3D圖 到 哪張fig的哪張子圖
    y_fip ：
    '''

    ax[ax_c].remove()  ### 因為 是 3D子圖 要和 2D子圖 放同張figure， 所以要 ax[ax_c].remove() 把原本的 2D子圖 隱藏起來(.remove())
    ax_cols = len(ax)  ### 定位出 ax3d 要加入 fig 的哪個的位置， 需要知道目前的 fig_rows/cols
    ax3d = fig.add_subplot(ax_rows, ax_cols, (ax_r * ax_cols) + ax_c + 1, projection="3d")  ### +1 是因為 add_subplot( 括號裡面index 都是從 1 開始！)
    ax[ax_c] = ax3d  ### 把 ax3d 取代 原本的 ax

    ### 設定 xyz_label
    ax3d.set_xlabel(xlabel)
    ax3d.set_ylabel(ylabel)
    # ax3d.set_zlabel(zlabel)  ### z軸很常客製化的顯示自己要的字， 所以最後決定拿掉囉不設定"z"了

    ### 設定title
    if(fig_title is not None): ax3d.set_title(fig_title)
    ### y軸顛倒
    if(y_flip):
        if( y_coord is None):
            print("change_into_3D_coord_ax_combine 如果 y_flip 設定True， 需要 y_coord 參數， 才能知道 y_max/min 來顛倒y軸， \
                目前偵測到沒給 y_coord 因此停止程式， 麻煩去給一下囉～")
            sys.exit()
        else:
            y_max = y_coord.max()
            y_min = y_coord.min()
            ax3d.set_ylim(y_max, y_min)
    if(tight_layout): fig.tight_layout()

    ### 偷偷畫一個透明方形 來達到 xlim, ylim 的效果
    # verts = np.array(  [[(-XLIM, -YLIM, 0),
    #                     (-XLIM,  YLIM, 0),
    #                     ( XLIM,  YLIM, 0),
    #                     ( XLIM, -YLIM, 0),
    #                 ]] )
    # ax3d.add_collection3d(Poly3DCollection(verts, alpha=1))  ### Poly3DCollection 偷偷畫一個透明方形 來達到 xlim, ylim 的效果  ### 雖然有點作用，但大小還是不一樣
    # ax3d.set_xlim(-XLIM,  XLIM)  ### 雖然有點作用，但大小還是不一樣
    # ax3d.set_ylim( YLIM, -YLIM)  ### 雖然有點作用，但大小還是不一樣
    return ax3d

def mesh3D_scatter_and_z0_plane(x_m, y_m, z_m, fig_title,
        xlabel="x", ylabel="y", zlabel="z",
        cmap="viridis",
        y_flip=False, scatter_alpha=1.0, plane_alpha=0.5, tight_layout=False,
        fig=None, ax=None, ax_c=None, ax_r=0):
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=1, ax_size=5, tight_layout=True)
    ##########################################################################################################
    h_res, w_res = x_m.shape[:2]
    ax3d = change_into_3D_coord_ax(fig, ax, ax_c, fig_title=fig_title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, y_flip=y_flip, y_coord=y_m[..., 1], tight_layout=tight_layout)
    ax3d.set_title(fig_title)
    ax3d.scatter(x_m, y_m, z_m , c = np.arange(h_res * w_res), s=1, cmap=cmap, alpha=scatter_alpha)
    ax3d = draw_3D_xy_plane_by_mesh_f(ax3d, x_m=x_m, y_m=y_m, z=0, alpha=plane_alpha )
    ax_c += 1

    return fig, ax, ax_c, ax3d


def change_into_img_2D_coord_ax(ax):
    '''
    image 的 coordinate 概念：
        x軸 往右 增加
        y軸 往下 增加

    最後不 return 其實也可以， 但為了外面寫 ax[...] 的美觀統一性， 所以這邊就return 一下囉
    '''
    if(not ax.yaxis_inverted()): ax.invert_yaxis()  ### 整張圖上下顛倒(如果還沒顛倒過的話)
    ax.spines['right'].set_color('None')
    ax.spines['top']  .set_color('None')
    ax.xaxis.set_ticks_position('bottom')         # 設定bottom 為 x軸
    ax.yaxis.set_ticks_position('left')           # 設定left   為 y軸
    ax.spines['bottom'].set_position(('data', 0))  # 設定bottom x軸 位置(要丟tuple)
    ax.spines['left']  .set_position(('data', 0))  # 設定left   y軸 位置(要丟tuple)
    ax.add_patch( patches.Rectangle( (-XLIM, -YLIM), 2 * XLIM, 2 * YLIM, alpha=0))   ### 偷偷畫一個透明方形 來達到 xlim, ylim 的效果
    return ax


def check_fig_ax_init(fig=None, ax=None, ax_c=None, ax_r=0, fig_rows=1, fig_cols=5, ax_size=7, tight_layout=False):
    """
    檢查 fig/ax 是否 同時都存在， 如果同時都存在OK， 同時都不存在 就 建立新subplots， 一者不在代表有錯誤，應該是忘記傳另一方近來， 就停止程式去修正囉
    fig_cols/ ax_size/ tight_layout 只有在 fig/ax is None 要建立新subplots 時才會用到
    """
    if   (fig is None and ax is None):
        fig, ax = plt.subplots(nrows=fig_rows, ncols=fig_cols)
        fig.set_size_inches(ax_size * fig_cols, ax_size * fig_rows)
        if(tight_layout): fig.tight_layout()
        ax_c = 0
    elif(fig is not None and ax is None):
        print("有fig， 但忘記傳 ax 近來囉！ 停止程式，麻煩去把ax 傳進來 或者 fig/ax 都不要傳進來 讓 funtion 自己建立新subplots")
        sys.exit()
    elif(fig is None and ax is not None):
        print("有ax， 但忘記傳 fig 近來囉！ 停止程式，麻煩去把fig 傳進來 或者 fig/ax 都不要傳進來 讓 funtion 自己建立新subplots")
        sys.exit()
    elif((fig is not None and ax is not None)):
        if(ax_c is None):
            print("有fig/ax， 但忘記傳 ax_c 近來囉！ 停止程式，麻煩去把ax_c 傳進來 或者 fig/ax 都不要傳進來 讓 funtion 自己建立新subplots")
            sys.exit()
    ### 怕有 fig_rows == 1 and fig_cols==1 的情況， ax 不是 np.array， 遇到這情況就要把他轉乘 np.array喔！
    if(type(ax) != type(np.array(1))): ax = np.array([ax])

    ### 走到這裡 可以確認 fig/ax/ax_c 都有東西，就回傳回去囉
    return fig, ax, ax_c

def draw_3D_xy_plane_by_mesh_f(ax3d, x_m, y_m, z=0, alpha=0.5):
    '''
    ax3       d： 要是 projection=3d 的那種 ax3d
    x_m   ： 沒錯就是要分開寫， 這樣 mesh_x, mesh_y 分別是不同東西時(比如：d_mesh, alpha_mesh)才比較好處理
    z          ： xy平面 z 的高度
    alpha      ： 透明度

    最後不 return 其實也可以， 但為了外面寫 ax3d 的美觀統一性， 所以這邊就return 一下囉
    '''
    row, col = x_m.shape[:2]
    ax3d.plot_surface(x_m,
                      y_m,
                      z * np.ones(shape=(row, col)),
                      alpha=0.5)
    return ax3d




def coord_f_2D_scatter(coord_f, h_res, w_res, fig_title=None, fig=None, ax=None, ax_c=None, ax_r=0, tight_layout=False):
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=1, ax_size=5, tight_layout=tight_layout)
    ##########################################################################################################
    if(fig_title is not None): ax[ax_c].set_title(fig_title)  ### 設定title
    ax[ax_c] = change_into_img_2D_coord_ax(ax[ax_c])          ### 構圖 變成 2D img coord 的形式
    ax[ax_c].scatter(coord_f[:, 0], coord_f[:, 1], c = np.arange(h_res * w_res), s=1, cmap="hsv")  ### 用scatter 視覺化
    ax_c += 1
    return fig, ax, ax_c

def coord_m_2D_scatter(coord_m,
                       fig_title=None,
                       fig_C=None, fig_cmap="hsv", fig_alpha=1,
                       fig=None, ax=None, ax_c=None, ax_r=0, tight_layout=False):
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=1, ax_size=5, tight_layout=tight_layout)
    h_res, w_res = coord_m.shape[:2]
    ##########################################################################################################
    if(fig_title is not None): ax[ax_c].set_title(fig_title)  ### 設定title
    if(fig_C is None): fig_C = np.arange(h_res * w_res)
    ##########################################################################################################
    ax[ax_c] = change_into_img_2D_coord_ax(ax[ax_c])          ### 構圖 變成 2D img coord 的形式
    ### 真正要做的事情開始
    ax[ax_c].scatter(coord_m[..., 0], coord_m[..., 1], c = fig_C, s=1, cmap=fig_cmap, alpha=fig_alpha)  ### 用scatter 視覺化
    ax_c += 1
    return fig, ax, ax_c


def get_jump_index(number, jump_step):
    step = number - 1  ### 步數 比 格數 少1， 比如有10格， 第一格走到第10個只需要9步
    jump_amount = step // jump_step
    if(step % jump_step == 0): return [ go_j * jump_step for go_j in range(jump_amount + 1)  ]           ### 如果  整除，直接用
    else                     : return [ go_j * jump_step for go_j in range(jump_amount + 1)  ] + [step]  ### 如果不整除，加最後的index

def apply_jump_index(data, row_ids_m, col_ids_m):
    if(check_is_numpy_and_is_map_form_or_exit(data)):
        return data[row_ids_m, col_ids_m]

def check_is_numpy_and_is_map_form_or_exit(data, print_msg=False):  ### 防呆
    if(print_msg): print("執行 check_is_numpy_and_is_map_form_or_exit")

    if(type(data) == type(np.array(1))): pass
    else: sys.exit(f"資料型態不是 numpy array， 目前型態為 {type(data)}， 已停止程式， 請修正為 numpy array")

    if  (data.ndim == 3): pass
    elif(data.ndim == 2 and data.shape[1] > 3 ): pass  ### data.shape[1] < 3 可能是 (h * w, 3) 之類的忘記reshape了
    else: sys.exit(f"確認為 numpy array， 但不是 map_form！ 目前shape為{data.shape}， 已停止程式， 請修正為 (h, w) 或 (h, w, c) 之類的樣子")

    return True


def move_map_1D_value(move_map_m, move_x, move_y,
                      fig_title=None,
                      fig=None, ax=None, ax_c=None, ax_r=0, tight_layout=False):
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=1, ax_size=5, tight_layout=tight_layout)
    if(fig_title is not None): ax[ax_c].set_title(fig_title)  ### 設定title
    h_res, w_res = move_map_m.shape[:2]
    ##########################################################################################################
    ax[ax_c].set_title(fig_title)
    ax[ax_c] = change_into_img_2D_coord_ax(ax[ax_c])
    ax[ax_c].scatter(move_map_m[..., 0], move_map_m[..., 1], c = np.arange(h_res * w_res), s=1, cmap="hsv")
    ax[ax_c].arrow(0, 0, move_x, move_y, color="black", length_includes_head=True, width = 0.001,  head_width=0.01, alpha=0.2)  ### 移動向量，然後箭頭化的方式是(x,y,dx,dy)！ 不是(x1,y1,x2,y2)！head_width+1是怕col太小除完變0
    ax_c += 1

    return fig, ax, ax_c

def move_map_2D_arrow(move_map_m, start_xy_m,
        fig_title=None,
        jump_r=7, jump_c=7,
        arrow_C=None, arrow_cmap="viridis", arrow_alpha=1.0,
        boundary_value=0, boundary_C="orange", boundary_linewidth=3, boundary_fill=False,
        show_before_move_coord=False, before_alpha=0.1, before_C=None, before_cmap="hsv",
        show_after_move_coord=False,  after_alpha=1.0,   after_C=None,  after_cmap="hsv",

        fig=None, ax=None, ax_c=None, ax_r=0, tight_layout=False):
    '''
    move_map_m： shape為(h_res, w_res, 2) 移動的向量 move
    start_xy_m： shape為(h_res, w_res, 2) 移動的起始 coord
    jump  ： 一次跳幾格 show點
    C/cmap： str直接指定顏色 或 shape為(h_res, w_res, 1) 單純把 點 編號 後 讓matplot自動幫你填 cmap 的顏色
    color ： shape為(h_res, w_res, 3 或 4(含alpha透明度))， 值域 0.~1.， 每一點的顏色都可以自己指定， 記得在丟進去matplot 時要把 color reshape 成 (-1, 3) 或 (-1, 4) 喔！
    alpha ： 透明度
    '''
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=1, ax_size=5, tight_layout=tight_layout)
    ##########################################################################################################
    if(fig_title is not None): ax[ax_c].set_title(fig_title)  ### 設定title
    ax[ax_c] = change_into_img_2D_coord_ax(ax[ax_c])

    ### 看要不要把 移動前的座標 畫出來
    if(show_before_move_coord):
        fig, ax, ax_c = coord_m_2D_scatter(start_xy_m,
                            fig_alpha=before_alpha, fig_C=before_C, fig_cmap=before_cmap,
                            fig=fig, ax=ax, ax_c=ax_c, ax_r=ax_r)
        ax_c -= 1


    ### 處理 jump 的 index
    h_res, w_res = start_xy_m.shape[:2]
    row_ids = get_jump_index(h_res, jump_r)
    col_ids = get_jump_index(w_res, jump_c)
    row_ids_m, col_ids_m = np.meshgrid(col_ids, row_ids)

    ### 處理 jump 的 index， 顏色 抓 jump 後的 顏色
    if(arrow_C is None): arrow_C = np.zeros(shape=(h_res, w_res))
    arrow_C = apply_jump_index(arrow_C, row_ids_m, col_ids_m)

    ##########################################################################################################
    ### 真正要做的事情開始
    ax[ax_c].quiver(start_xy_m      [row_ids_m, col_ids_m, 0],
                    start_xy_m      [row_ids_m, col_ids_m, 1],
                    move_map_m[row_ids_m, col_ids_m, 0],
                    move_map_m[row_ids_m, col_ids_m, 1],
                    arrow_C, cmap=arrow_cmap, alpha=arrow_alpha,                ### arrow_C 的 關鍵字參數名字我猜不到 也查不大到， 但是發現只要放 第5個參數 就對囉！ 且無法丟None， 且當指定color時， 會以 arrow_C的顏色為主 忽略color， 所以目前無法用 color， 只能用 cmap調顏色
                    angles='xy', scale_units='xy', scale=1, pivot="tail",  ### 設定 quiver 參考 xy座標 來畫圖，scale 指定1 才不會用到 auto-scale
                   )

    ### 看要不要把 移動後的座標 也畫出來
    if(show_after_move_coord):
        dis_coord_m = move_map_m + start_xy_m
        fig, ax, ax_c = coord_m_2D_scatter(dis_coord_m,
                                        fig_alpha=after_alpha, fig_C=after_C, fig_cmap=after_cmap,
                                        fig=fig, ax=ax, ax_c=ax_c, ax_r=ax_r)
        ax_c -= 1

    ### 看要不在 移動後的結果 框出一個 valid 區域， 已paper17 是用 pytorch 的 grid_sample， valid 區域維 -1~1
    if(boundary_value != 0):
        ax[ax_c].add_patch( patches.Rectangle( (-boundary_value, -boundary_value), 2 * boundary_value, 2 * boundary_value, edgecolor=boundary_C , fill=boundary_fill, linewidth=boundary_linewidth))

    ax_c += 1
    return fig, ax, ax_c


def move_map_3D_scatter(move_map_m, start_xy_m,
                        fig_title=None,
                        zticklabels=(),
                        jump_r=1, jump_c=1,
                        boundary_value=0, boundary_C="orange", boundary_height=0.5, boundary_linewidth=3, boundary_fill=False, boundary_alpha=1,
                        before_C=None, before_cmap="hsv", before_color=None, before_alpha=0.1, before_s=1, before_height=0,
                        after_C=None,  after_cmap="hsv",  after_color=None,  after_alpha=1.0,  after_s=1,  after_height=0.5,
                        fig=None, ax=None, ax_c=None, ax_r=0, ax_rows=1, tight_layout=False):
    '''
    move_map_m： shape為(h_res, w_res, 2) 移動的向量 move
    start_xy_m： shape為(h_res, w_res, 2) 移動的起始 coord
    zticklabels： z軸上要標得字
    jump  ： 一次跳幾格 show點
    C/cmap： str直接指定顏色 或 shape為(h_res, w_res, 1) 單純把 點 編號 後 讓matplot自動幫你填 cmap 的顏色
    color ： shape為(h_res, w_res, 3 或 4(含alpha透明度))， 值域 0.~1.， 每一點的顏色都可以自己指定， 記得在丟進去matplot 時要把 color reshape 成 (-1, 3) 或 (-1, 4) 喔！
    alpha ： 透明度
    height： 在 z 的高度
    '''
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=1, ax_size=5, tight_layout=tight_layout)
    if(fig_title is not None): ax[ax_c].set_title(fig_title)  ### 設定title
    ax3d = change_into_3D_coord_ax(fig, ax, ax_c, ax_r, ax_rows, y_flip=True, y_coord=start_xy_m[..., 1], tight_layout=tight_layout)
    ##########################################################################################################
    ### 處理 jump 的 index
    h_res, w_res = start_xy_m.shape[:2]
    row_ids = get_jump_index(h_res, jump_r)
    col_ids = get_jump_index(w_res, jump_c)
    row_ids_m, col_ids_m = np.meshgrid(col_ids, row_ids)
    ### 處理 jump 的 index， 顏色 抓 jump 後的 顏色
    if(before_C is not None and type(before_C) != type("str")): before_C = apply_jump_index(before_C, row_ids_m, col_ids_m)
    if(after_C  is not None and type(after_C)  != type("str")): after_C  = apply_jump_index(after_C , row_ids_m, col_ids_m)
    if(before_color is not None):
        before_color = apply_jump_index(before_color, row_ids_m, col_ids_m)
        before_color = before_color.reshape(-1, 3)
    if(after_color  is not None):
        after_color  = apply_jump_index(after_color , row_ids_m, col_ids_m)
        after_color  = after_color.reshape(-1, 3)


    ax3d.set_title(fig_title)
    y_max = start_xy_m[..., 1].max()
    y_min = start_xy_m[..., 1].min()
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_ylim(y_max, y_min)
    ax3d.set_zlim(0, 1)
    ax3d.set_zticklabels(zticklabels)

    ##########################################################################################################
    ### 真正要做的事情開始
    ax3d.scatter(start_xy_m[row_ids_m, col_ids_m, 0]                                      , start_xy_m[row_ids_m, col_ids_m, 1]                                      , before_height, c=before_C, s=before_s, cmap=before_cmap, alpha=before_alpha, color=before_color)
    ax3d.scatter(start_xy_m[row_ids_m, col_ids_m, 0] + move_map_m[row_ids_m, col_ids_m, 0], start_xy_m[row_ids_m, col_ids_m, 1] + move_map_m[row_ids_m, col_ids_m, 1], after_height , c=after_C , s=after_s,  cmap=after_cmap, alpha=after_alpha)
    ##########################################################################################################
    if(boundary_value != 0):
        verts = np.array(  [[(-boundary_value, -boundary_value, boundary_height),
                             (-boundary_value,  boundary_value, boundary_height),
                             ( boundary_value,  boundary_value, boundary_height),
                             ( boundary_value, -boundary_value, boundary_height),
                             (-boundary_value, -boundary_value, boundary_height),  ### Poly不需要這個， 但多這個也沒差， 不過Line 必須要這個才能圍成方形， 所以就加這行囉！
                            ]] )
        if(boundary_fill): ax3d.add_collection3d(Poly3DCollection(verts, linewidth=boundary_linewidth, facecolor=boundary_C, alpha=boundary_alpha))  ### Poly3DCollection 目前找不到方法 內部空白， 所以乾脆用畫線的方式
        else             : ax3d.add_collection3d(Line3DCollection(verts, linewidth=boundary_linewidth, color=boundary_C, alpha=boundary_alpha))  ### Poly3DCollection 目前找不到方法 內部空白， 所以乾脆用畫線的方式
    ax_c += 1

    return fig, ax, ax_c, ax3d


def move_map_2D_moving_visual(move_map_m, start_xy_m, fig_title="",
                              fig=None, ax=None, ax_c=None, ax_r=0, tight_layout=False):
    '''
    start_xy_m  ： 單純視覺化用的座標而已
    move_map_m  ： 單純視覺化出來而已
    fig/ax/ax_c ： default 為 None， 代表要 建立新subplots
                   若 不是 None，在 fig上 畫上此function裡產生的圖
    '''
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=4, ax_size=5, tight_layout=tight_layout)
    ##########################################################################################################
    fig, ax, ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m, fig_title=":".join([fig_title, "move_map"]), jump_r=4, jump_c=4,
                             show_before_move_coord=False, before_alpha=0.2,
                             show_after_move_coord=False, after_alpha=0.8,
                             fig=fig, ax=ax, ax_c=ax_c, ax_r=ax_r)

    fig, ax, ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m, fig_title=":".join([fig_title, "move_map+coord==bm"]), jump_r=4, jump_c=4,
                             show_before_move_coord=True, before_alpha=0.2,
                             show_after_move_coord=False, after_alpha=0.8,
                             fig=fig, ax=ax, ax_c=ax_c, ax_r=ax_r)

    ax[ax_c].set_title(":".join([fig_title, "bm visual"]))
    dis_coord_m = move_map_m + start_xy_m
    dis_coord_method1_visual = method1(x=dis_coord_m[..., 0], y=dis_coord_m[..., 1], mask_ch=0)
    ax[ax_c].imshow(dis_coord_method1_visual)
    ax_c += 1

    fig, ax, ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m, fig_title=":".join([fig_title, "bm moving"]), jump_r=4, jump_c=4,
                             show_before_move_coord=True, before_alpha=0.2,
                             show_after_move_coord=True, after_alpha=0.8,
                             fig=fig, ax=ax, ax_c=ax_c, ax_r=ax_r)

    ### 畫 2D scatter 移動後的 coord
    fig, ax, ax_c = coord_m_2D_scatter(dis_coord_m, fig_title=":".join([fig_title, "dis_coord appearance is fm"]), fig=fig, ax=ax, ax_c=ax_c)


    # ### 3D箭頭視覺化 效果不好： 分不同平面 畫箭頭 發現 箭頭中間的線太密看不到平面ˊ口ˋ
    # 但這 3D_quiver例子 又捨不得刪掉， 先留著好了
    # jump_r = 8
    # jump_c = 7
    # ax3d = change_into_3D_coord_ax(fig, ax, ax_c, y_flip=True, y_coord=start_xy_m[..., 1], tight_layout=True)
    # ax3d.set_title("step5.move_map 3D_visual fold")
    # ax3d.set_zlim(0, 1)
    # ax3d.quiver(start_xy_m[::jump_r, ::jump_c, 0],
    #             start_xy_m[::jump_r, ::jump_c, 1],
    #             0,
    #             move_map_m[::jump_r, ::jump_c, 0],
    #            -move_map_m[::jump_r, ::jump_c, 1],  ### quiver 好像就算 整張圖y已經反轉了， 但畫出來的y沒有跟著轉， 所以只好自己加 "-" 來反轉囉！
    #             0.5,
    #             alpha=0.5)
    # ax_c += 1


    ### 畫 3D scatter 分不同平面 沒有箭頭的中間段 效果比較好
    # jump_r = 2
    # jump_c = 2
    # ax3d = change_into_3D_coord_ax(fig, ax, ax_c, y_flip=True, y_coord=start_xy_m[..., 1], tight_layout=True)
    # ax3d.set_title(f"step5.move_map_{dis_type} 3D_scatter_visual")
    # ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_ylim(y_max, y_min)
    # ax3d.set_zlim(0, 1)
    # ax3d.scatter(start_xy_m[::jump_r, ::jump_c, 0]                                    , start_xy_m[::jump_r, ::jump_c, 1]                                    , 0   , c = np.arange(np.ceil(h_res / jump_r) * np.ceil(w_res / jump_c)), s=1, cmap="hsv", alpha=0.3)
    # ax3d.scatter(start_xy_m[::jump_r, ::jump_c, 0] + move_map_m[::jump_r, ::jump_c, 0], start_xy_m[::jump_r, ::jump_c, 1] + move_map_m[::jump_r, ::jump_c, 1], 0.5 , c = np.arange(np.ceil(h_res / jump_r) * np.ceil(w_res / jump_c)), s=3, cmap="hsv")
    # ax_c += 1
    return fig, ax, ax_c

def img_scatter_visual(img,
                    x_min  = -1.00, x_max  = +1.00, y_min  = -1.00, y_max  = +1.00,
                    alpha = 1., s = 1,
                    jump_r=1, jump_c=1,
                    fig=None, ax=None, ax_c=None, tight_layout=False):
    '''
    先假設 座標 值域在 -1~1
    '''
    from util import get_xy_f_and_m
    fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=1, fig_cols=1, ax_size=10, tight_layout=tight_layout)
    ##################################################################################################################

    h_res, w_res = img.shape[:2]
    jump_r = jump_r
    jump_c = jump_c
    start_xy_f, start_xy_m = get_xy_f_and_m(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res)  ### 拿到map的shape：(..., 2), f 是 flatten 的意思
    ax[ax_c].scatter(start_xy_m[..., 0], start_xy_m[..., 1], s = s, color=img.reshape(-1, 3) / 255, alpha=alpha)
    ax_c += 1
    return fig, ax, ax_c


if(__name__ == "__main__"):
    # fig, ax = plt.subplots(nrows=3, ncols=3)
    # gs = ax[0, 0].get_gridspec()  ### 取得 整張圖的 grid範圍規格表 我猜

    # fig_new, ax_new = plt.subplots(nrows=4, ncols=3)
    # gs_new = ax_new[0, 0].get_gridspec()  ### 取得 整張圖的 grid範圍規格表 我猜

    # for ax_new_final_row in ax_new[-1]: ax_new_final_row.remove()

    # ax_new_row = fig_new.add_subplot(gs_new[-1, :])
    # print(ax_new[0, 0].get_gridspec())
    # print(ax_new_row.grid())
    # print(gs_new[-1, :])
    # plt.close(fig)

    # plt.show()
    ############################################################################################################################
    import cv2
    this_py_path = "C:/Users/TKU/Desktop/kong_model2/kong_util"
    img1 = cv2.imread(f"{this_py_path}/img_data/0a-in_img.jpg")
    img2 = cv2.imread(f"{this_py_path}/img_data/0b-gt_a_gt_flow.jpg")
    img3 = cv2.imread(f"{this_py_path}/img_data/epoch_0000_a_flow_visual.jpg")

    single_row_imgs = Matplot_single_row_imgs(imgs       = [img1, img2, img3],
                                              img_titles = ["in_img", "pred", "GT"],
                                              fig_title  = "epoch0000",
                                              bgr2rgb    = True,
                                              add_loss   = False)

    # single_row_imgs.step1_add_row_col(add_where="add_row", merge=False)
    # single_row_imgs.step1_add_row_col(add_where="add_row", merge=False)

    # single_row_imgs.step1_add_row_col(add_where="add_row", merge=True)
    # single_row_imgs.step1_add_row_col(add_where="add_row", merge=False)

    single_row_imgs.step1_add_row_col(add_where="add_row", merge=True)
    print("2 finish")
    single_row_imgs.step1_add_row_col(add_where="add_row", merge=False)
    print("3 finish")
    single_row_imgs.step1_add_row_col(add_where="add_row", merge=True)
    print("4 finish")
    single_row_imgs.step1_add_row_col(add_where="add_col", merge=True)
    print("5 finish")
    single_row_imgs.Draw_img()
    single_row_imgs.merged_ax_list[0].imshow(img1)
    single_row_imgs.merged_ax_list[1].imshow(img1)
    single_row_imgs.merged_ax_list[2].imshow(img1)
    single_row_imgs.ax[2, 0].imshow(img1)
    single_row_imgs.ax[2, 1].imshow(img1)
    single_row_imgs.ax[2, 2].imshow(img1)
    single_row_imgs.ax[2, 3].imshow(img1)
    plt.show()

    ############################################################################################################################
    # subplots_combine_example()
    # fig, ax = plt.subplots(1,2)
    # import numpy as np
    # print(type(ax))
    # print(ax.shape)
    # print(type(np.array(1)))
