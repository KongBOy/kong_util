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
    def __init__(self, imgs, img_titles, fig_title, bgr2rgb=False, add_loss=False):
        self.imgs       = imgs  ### imgs是個list，裡面放的圖片可能不一樣大喔
        self.img_titles = img_titles
        self.fig_title  = fig_title
        self.bgr2rgb    = bgr2rgb
        self.add_loss   = add_loss

        self.row_imgs_amount   = 1
        self.col_imgs_amount   = len(self.imgs)
        self.col_titles_amount = len(self.img_titles)
        self._step1_build_check()


        self.canvas_height     = None
        self.canvas_width      = None
        self.fig = None
        self.ax  = None
        self._step2_set_canvas_hw_and_build()


    def _get_one_row_canvas_height(self):
        height_list = []    ### imgs是個list，裡面放的圖片可能不一樣大喔
        for img in self.imgs: height_list.append(img.shape[0])
        return  (max(height_list) // 100 + 1.0) * 1.0 + 1.5  ### 慢慢試囉～ +1.5是要給title 和 matplot邊界margin喔

    def _get_one_row_canvas_width(self):
        width = 0
        for img in self.imgs: width += img.shape[1]

        if  (self.col_imgs_amount == 3): return  (width // 100 + 0) * 1.0 + 5.7   ### 慢慢試囉～ col=3時
        elif(self.col_imgs_amount == 4): return  (width // 100 + 0) * 1.0 + 6.8   ### 慢慢試囉～ col=4時
        elif(self.col_imgs_amount == 5): return  (width // 100 + 0) * 1.0 + 8.5   ### 慢慢試囉～ col=5時
        elif(self.col_imgs_amount == 6): return  (width // 100 + 0) * 1.0 + 10.5  ### 慢慢試囉～ col=6時
        elif(self.col_imgs_amount == 7): return  (width // 100 + 0) * 1.0 + 11.5  ### 慢慢試囉～ col=7時
        elif(self.col_imgs_amount >  7): return  (width // 100 + 0) * 1.0 + 11.5   ### 慢慢試囉～ col=7時，沒有試過用猜的，因為覺得用不到ˊ口ˋ用到再來試


    def _step1_build_check(self):
        #### 防呆 ####################################################
        ### 正常來說 一個 title 對應 一張圖
        if( self.col_titles_amount < self.col_imgs_amount):  ### 如果 title數 比 影像數多，那就用 空title來補
            for _ in range(self.col_imgs_amount - self.col_titles_amount):
                self.img_titles.append("")

        elif(self.col_titles_amount > self.col_imgs_amount):
            print("title 太多了，沒有圖可以對應")
            return

        if(self.col_imgs_amount == 0):
            print("沒圖可show喔！")
            return

    def _step2_set_canvas_hw_and_build(self):
        ### 設定canvas的大小
        self.canvas_height = self._get_one_row_canvas_height()
        self.canvas_width  = self._get_one_row_canvas_width()
        if(self.add_loss):   ### 多一些空間來畫loss
            self.row_imgs_amount += 1  ### 多一row來畫loss
            self.canvas_height += 3    ### 慢慢試囉～
            self.canvas_width  -= 1.5 * self.col_imgs_amount  ### 慢慢試囉～
        # print("canvas_height",canvas_height)
        # print("canvas_width",canvas_width)
        # print("row_imgs_amount", row_imgs_amount)
        # print("col_imgs_amount", col_imgs_amount)

        ### 建立canvas出來
        self.fig, self.ax = plt.subplots(nrows=self.row_imgs_amount, ncols=self.col_imgs_amount)
        self.fig.set_size_inches(self.canvas_width, self.canvas_height)  ### 設定 畫布大小


    def _step3_draw(self, used_ax):
        ### 這就是手動微調 text的位置囉ˊ口ˋ
        self.fig.text(x=0.5, y=0.945, s=self.fig_title, fontsize=20, c=(0., 0., 0., 1.),  horizontalalignment='center',)

        for go_img, img in enumerate(self.imgs):
            if(self.bgr2rgb): img = img[..., ::-1]  ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
            if(self.col_imgs_amount > 1):
                used_ax[go_img].imshow(img)  ### 小畫布 畫上影像，別忘記要bgr -> rgb喔！
                used_ax[go_img].set_title( self.img_titles[go_img], fontsize=16 )  ### 小畫布上的 title

                plt.sca(used_ax[go_img])  ### plt指向目前的 小畫布 這是為了設定 yticks和xticks
                plt.yticks( (0, img.shape[0]), (0, img.shape[0]) )   ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                plt.xticks( (0, img.shape[1]), ("", img.shape[1]) )  ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
            else:
                used_ax.imshow(img)  ### 小畫布 畫上影像
                used_ax.set_title( self.img_titles[go_img], fontsize=16 )  ### 小畫布上的 title

                plt.yticks( (0, img.shape[0]), (0, img.shape[0]) )   ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                plt.xticks( (0, img.shape[1]), ("", img.shape[1]) )  ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字

    def Draw_img(self):  ### 呼叫 _step3_draw 畫圖
        ###############################################################
        ### 注意 _draw_single_row_imgs 的 ax 只能丟 一row，所以才寫這if/else
        if(not self.add_loss): used_ax = self.ax
        elif(self.add_loss):   used_ax = self.ax[0]  ### 只能丟第一row喔！因為_draw_single_row_imgs 裡面的操作方式 是 一row的方式，丟兩row ax維度會出問題！
        self._step3_draw(used_ax)
        ###############################################################
        ### 想畫得更漂亮一點，兩種還是有些一咪咪差距喔~
        if(not self.add_loss): self.fig.tight_layout(rect=[0, 0, 1, 0.93])
        else:                  self.fig.tight_layout(rect=[0, 0.006, 1, 0.95])
        ###############################################################
        ### Draw_img完，不一定要馬上Draw_loss喔！像是train的時候 就是分開的 1.see(Draw_img), 2.train, 3.loss(Draw_loss)


"""
### 已經 包成class 了

### imgs是個list，裡面放的圖片可能不一樣大喔
def _get_one_row_canvas_height(imgs):
    height_list = []
    for img in imgs: height_list.append(img.shape[0])
    # return  (max(height_list) // 100+2.0) * 0.8  ### 沒有弄得很精準，+1好了
    return  (max(height_list) // 100 + 1.0) * 1.0 + 1.5  ### 慢慢試囉～ +1.5是要給title 和 matplot邊界margin喔

def _get_one_row_canvas_width(imgs):
    width = 0
    for img in imgs: width += img.shape[1]

    if  (len(imgs) == 3): return  (width // 100 + 0) * 1.0 + 5.7   ### 慢慢試囉～ col=3時
    elif(len(imgs) == 4): return  (width // 100 + 0) * 1.0 + 6.8   ### 慢慢試囉～ col=4時
    elif(len(imgs) == 5): return  (width // 100 + 0) * 1.0 + 8.5   ### 慢慢試囉～ col=5時
    elif(len(imgs) == 6): return  (width // 100 + 0) * 1.0 + 10.5  ### 慢慢試囉～ col=6時
    elif(len(imgs) == 7): return  (width // 100 + 0) * 1.0 + 11.5  ### 慢慢試囉～ col=7時
    elif(len(imgs)  > 7): return  (width // 100 + 0) * 1.0 + 11.5   ### 慢慢試囉～ col=7時，沒有試過用猜的，因為覺得用不到ˊ口ˋ用到再來試

### single_row 的處理方式 還是跟 multi_row 有些許不同，所以不能因為時做出 multi後取代single喔！ 比如 ax[] 的維度、取長寬比之類的～
def _draw_single_row_imgs(fig, ax, col_imgs_amount, canvas_height, canvas_width, img_titles, imgs, fig_title="epoch = 1005", bgr2rgb=True):
    ### 這就是手動微調 text的位置囉ˊ口ˋ
    fig.text(x=0.5, y=0.945, s=fig_title, fontsize=20, c=(0., 0., 0., 1.),  horizontalalignment='center',)

    for go_img, img in enumerate(imgs):
        if(bgr2rgb): img[..., ::-1]  ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
        if(col_imgs_amount > 1):
            ax[go_img].imshow(img)  ### 小畫布 畫上影像，別忘記要bgr -> rgb喔！
            ax[go_img].set_title( img_titles[go_img], fontsize=16 )  ### 小畫布上的 title

            plt.sca(ax[go_img])  ### plt指向目前的 小畫布 這是為了設定 yticks和xticks
            plt.yticks( (0, img.shape[0]), (0, img.shape[0]) )   ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
            plt.xticks( (0, img.shape[1]), ("", img.shape[1]) )  ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
        else:
            ax.imshow(img)  ### 小畫布 畫上影像
            ax.set_title( img_titles[go_img], fontsize=16 )  ### 小畫布上的 title

            plt.yticks( (0, img.shape[0]), (0, img.shape[0]) )   ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
            plt.xticks( (0, img.shape[1]), ("", img.shape[1]) )  ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字


def matplot_visual_single_row_imgs(img_titles, imgs, fig_title="epoch = 1005", bgr2rgb=True, add_loss=False):
    col_titles_amount = len(img_titles)
    row_imgs_amount = 1
    col_imgs_amount = len(imgs)

    #### 防呆 ####################################################
    if( col_titles_amount < col_imgs_amount):
        for _ in range(col_imgs_amount - col_titles_amount):
            img_titles.append("")
    elif(col_titles_amount > col_imgs_amount):
        print("title 太多了，沒有圖可以對應")
        return

    if(col_imgs_amount == 0):
        print("沒圖可show喔！")
        return
    ###########################################################
    ### 設定canvas的大小
    canvas_height = _get_one_row_canvas_height(imgs)
    canvas_width  = _get_one_row_canvas_width(imgs)
    if(add_loss):   ### 多一些空間來畫loss
        row_imgs_amount += 1  ### 多一row來畫loss
        canvas_height += 3    ### 慢慢試囉～
        canvas_width  -= 1.5 * col_imgs_amount  ### 慢慢試囉～
    # print("canvas_height",canvas_height)
    # print("canvas_width",canvas_width)
    # print("row_imgs_amount", row_imgs_amount)
    # print("col_imgs_amount", col_imgs_amount)

    ### 建立canvas出來
    fig, ax = plt.subplots(nrows=row_imgs_amount, ncols=col_imgs_amount)
    fig.set_size_inches(canvas_width, canvas_height)  ### 設定 畫布大小
    ###############################################################
    ### 注意 _draw_single_row_imgs 的 ax 只能丟 一row，所以才寫這if/else
    if(not add_loss): used_ax = ax
    elif(add_loss):   used_ax = ax[0]  ### 只能丟第一row喔！因為_draw_single_row_imgs 裡面的操作方式 是 一row的方式，丟兩row ax維度會出問題！
    _draw_single_row_imgs(fig, used_ax, col_imgs_amount, canvas_height, canvas_width, img_titles, imgs, fig_title, bgr2rgb)
    ###############################################################
    ### 想畫得更漂亮一點，兩種還是有些一咪咪差距喔~
    if(not add_loss): fig.tight_layout(rect=[0, 0, 1, 0.93])
    else:             fig.tight_layout(rect=[0, 0.006, 1, 0.95])
    ###############################################################
    ### 統一不存，因為可能還要給別人後續處理，這裡只負責畫圖喔！
    # plt.savefig(dst_dir+"/"+file_name)
    # plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！
    return fig, ax
"""
##########################################################################################################################################################
##########################################################################################################################################################
class Matplot_multi_row_imgs(Matplot_util):
    def __init__(self, rows_cols_imgs, rows_cols_titles, fig_title, bgr2rgb=True, add_loss=False):
        self.r_c_imgs = rows_cols_imgs
        self.r_c_titles = rows_cols_titles
        self.fig_title = fig_title
        self.bgr2rgb = bgr2rgb
        self.add_loss = add_loss

        self.row_imgs_amount   = len(self.r_c_imgs)
        self.col_imgs_amount   = len(self.r_c_imgs[0])
        self.col_titles_amount = len(self.r_c_imgs[0])
        self._step1_build_check()

        self.canvas_height     = None
        self.canvas_width      = None
        self.fig = None
        self.ax  = None
        self._step2_set_canvas_hw_and_build()

    def _step1_build_check(self):
        #### 防呆 ####################################################
        if( self.col_titles_amount < self.col_imgs_amount):
            for row_titles in self.r_c_titles:
                for _ in range(self.col_imgs_amount - self.col_titles_amount):
                    row_titles.append("")
        elif(self.col_titles_amount > self.col_imgs_amount):
            print("title 太多了，沒有圖可以對應")
            return

        if(self.col_imgs_amount == 0):
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
            self.row_imgs_amount += 1  ### 多一row來畫loss
            self.canvas_height += 3.0  ### 慢慢試囉～
            self.canvas_width  -= 0.55 * self.col_imgs_amount  ### 慢慢試囉～
            self.canvas_height *= 1.1  #1.2最好，但有點佔記憶體  ### 慢慢試囉～
            self.canvas_width  *= 1.1  #1.2最好，但有點佔記憶體  ### 慢慢試囉～
        # print("canvas_height",canvas_height)
        # print("canvas_width",canvas_width)
        # print("row_imgs_amount", row_imgs_amount)

        ### 建立canvas出來
        self.fig, self.ax = plt.subplots(nrows=self.row_imgs_amount, ncols=self.col_imgs_amount)
        self.fig.set_size_inches(self.canvas_width, self.canvas_height)  ### 設定 畫布大小

    def _step3_draw(self):
        ### 這就是手動微調 text的位置囉ˊ口ˋ
        self.fig.text(x=0.5, y=0.95, s=self.fig_title, fontsize=20, c=(0., 0., 0., 1.),  horizontalalignment='center',)

        for go_row, row_imgs in enumerate(self.r_c_imgs):
            for go_col, col_img in enumerate(row_imgs):
                if(self.bgr2rgb): col_img = col_img[..., ::-1]  ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
                if(self.col_imgs_amount > 1):
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




"""
### 已經 包成class 了

def _get_row_col_canvas_height(r_c_imgs):
    height = 0
    for row_imgs in r_c_imgs: height += row_imgs[0].shape[0]
    return (height // 100 + 0) * 1.2  ### 慢慢試囉～ +1.5是要給title 和 matplot邊界margin喔

def _get_row_col_canvas_width(r_c_imgs):
    width = 0
    for col_imgs in r_c_imgs[0]: width += col_imgs.shape[1]
    return (width // 100 + 1) * 1.2  ### 慢慢試囉～


def _draw_multi_row_imgs(fig, ax, row_imgs_amount, col_imgs_amount, canvas_height, canvas_width, rows_cols_titles, rows_cols_imgs, fig_title="epoch = 1005", bgr2rgb=True):
    ### 這就是手動微調 text的位置囉ˊ口ˋ
    fig.text(x=0.5, y=0.95, s=fig_title, fontsize=20, c=(0., 0., 0., 1.),  horizontalalignment='center',)
    # if  (col_imgs_amount <  3):fig.text(x=0.5, y=0.92, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
    # elif(col_imgs_amount == 3):fig.text(x=0.5, y=0.91, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
    # elif(col_imgs_amount >  3):
    #     if  (row_imgs_amount <  3):fig.text(x=0.5, y=0.915, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
    #     elif(row_imgs_amount == 3):fig.text(x=0.5, y=0.90 , s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
    #     elif(row_imgs_amount >  3):fig.text(x=0.5, y=0.897, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',) ### 再往下覺得用不到就沒有試囉ˊ口ˋ有用到再來微調八~~

    for go_row, row_imgs in enumerate(rows_cols_imgs):
        for go_col, col_img in enumerate(row_imgs):
            if(bgr2rgb): col_img[..., ::-1]  ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
            if(col_imgs_amount > 1):
                ax[go_row, go_col].imshow(col_img)  ### 小畫布 畫上影像，別忘記要bgr -> rgb喔！
                if  (len(rows_cols_titles) > 1): ax[go_row, go_col].set_title( rows_cols_titles[go_row][go_col], fontsize=16 )  ### 小畫布　標上小標題
                elif(len(rows_cols_titles) == 1 and go_row == 0): ax[go_row, go_col].set_title( rows_cols_titles[go_row][go_col], fontsize=16 )  ### 小畫布　標上小標題

                plt.sca(ax[go_row, go_col])  ### plt指向目前的 小畫布 這是為了設定 yticks和xticks
                plt.yticks( (0, col_img.shape[0]), (0, col_img.shape[0]) )   ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                plt.xticks( (0, col_img.shape[1]), ("", col_img.shape[1]) )  ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
            else:  ### 要多這if/else是因為，col_imgs_amount == 1時，ax[]只會有一維！用二維的寫法會出錯！所以才獨立出來寫喔～
                ax[go_row].imshow(col_img)  ### 小畫布 畫上影像
                if  (len(rows_cols_titles) > 1): ax[go_row].set_title( rows_cols_titles[go_row][go_col], fontsize=16 )  ### 小畫布　標上小標題
                elif(len(rows_cols_titles) == 1 and go_row == 0): ax[go_row].set_title( rows_cols_titles[go_row][go_col], fontsize=16 )  ### 小畫布　標上小標題
                plt.yticks( (0, col_img.shape[0]), (0, col_img.shape[0]) )   ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                plt.xticks( (0, col_img.shape[1]), ("", col_img.shape[1]) )  ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字


def matplot_visual_multi_row_imgs(rows_cols_titles, rows_cols_imgs, fig_title="epoch=1005", bgr2rgb=True, add_loss=False):
    col_titles_amount = len(rows_cols_titles[0])
    row_imgs_amount   = len(rows_cols_imgs)
    col_imgs_amount   = len(rows_cols_imgs[0])

    #### 防呆 ####################################################
    if( col_titles_amount < col_imgs_amount):
        for row_titles in rows_cols_titles:
            for _ in range(col_imgs_amount - col_titles_amount):
                row_titles.append("")
    elif(col_titles_amount > col_imgs_amount):
        print("title 太多了，沒有圖可以對應")
        return

    if(col_imgs_amount == 0):
        print("沒圖可show喔！")
        return

    if(len(rows_cols_imgs) == 1):
        print("本function 不能處理 single_row_imgs喔，因為matplot在row只有1時的維度跟1以上時不同！麻煩呼叫相對應處理single_row的function！")
    ###########################################################
    ### 設定canvas的大小
    canvas_height = _get_row_col_canvas_height(rows_cols_imgs)
    canvas_width  = _get_row_col_canvas_width (rows_cols_imgs)
    if(add_loss):   ### 多一些空間來畫loss
        row_imgs_amount += 1  ### 多一row來畫loss
        canvas_height += 3.0  ### 慢慢試囉～
        canvas_width  -= 0.55 * col_imgs_amount  ### 慢慢試囉～
        canvas_height *= 1.2  ### 慢慢試囉～
        canvas_width  *= 1.2  ### 慢慢試囉～
    # print("canvas_height",canvas_height)
    # print("canvas_width",canvas_width)
    # print("row_imgs_amount", row_imgs_amount)

    ### 建立canvas出來
    fig, ax = plt.subplots(nrows=row_imgs_amount, ncols=col_imgs_amount)
    fig.set_size_inches(canvas_width, canvas_height)  ### 設定 畫布大小
    ###############################################################
    _draw_multi_row_imgs(fig, ax, row_imgs_amount, col_imgs_amount, canvas_height, canvas_width, rows_cols_titles, rows_cols_imgs, fig_title, bgr2rgb)
    ###############################################################
    ### 想畫得更漂亮一點，兩種還是有些一咪咪差距喔~
    if(not add_loss): fig.tight_layout(rect=[0, 0, 1, 0.95])       ### 待嘗試喔！
    else:             fig.tight_layout(rect=[0, 0.0035, 1, 0.95])  ### 待嘗試喔！
    ###############################################################
    ### 統一不存，因為可能還要給別人後續處理，這裡只負責畫圖喔！
    # plt.show()
    # plt.savefig(dst_dir+"/"+file_name)
    # plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！
    return fig, ax
"""


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
