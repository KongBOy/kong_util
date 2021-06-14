from util import get_dir_certain_file_name
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


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

def example_subplots_combine():
    ### https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
    import matplotlib.pyplot as plt

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

def example_subplots_adjust_axes_size_ratio():
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
    fig, ax = plt.subplots(nrows=2, ncols=1)
    print(ax)  ### [<AxesSubplot:> <AxesSubplot:>]
    fig, ax = plt.subplots(nrows=1, ncols=2)
    print(ax)  ### [<AxesSubplot:> <AxesSubplot:>]
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

    ### 關閉 軸 的 三個方法
    ### 方法1
    ax.set_xticks([])
    ax.set_yticks([])
    ### 方法2
    # plt.sca(ax)
    # plt.axis('off')
    ### 方法3
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())

    ax.imshow(img1)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    plt.show()

    # plt.savefig(f"filename.png")  ### 如何存


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
