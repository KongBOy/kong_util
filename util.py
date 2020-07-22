import numpy as np  
import cv2 
import os

from tqdm import tqdm
LOSS_YLIM = 2.0

def get_xy_map(row, col):
    x = np.arange(col)
    x = np.tile(x,(row,1))
    
#    y = np.arange(row-1, -1, -1) ### 就是這裡要改一下拉！不要抄網路的，網路的是用scatter的方式來看(左下角(0,0)，x往右增加，y往上增加)
    y = np.arange(row) ### 改成這樣子 就是用image的方式來處理囉！(左上角(0,0)，x往右增加，y往上增加)
    y = np.tile(y,(col,1)).T
    return x, y

def Check_img_filename(file_name):
    file_name = file_name.lower()
    if(".bmp" in file_name or ".jpg" in file_name or ".jpeg" in file_name or ".png" in file_name ):return True
    else: return False

def get_dir_img_file_names(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if Check_img_filename(file_name)]
    return file_names

def get_dir_certain_file_name(ord_dir, certain_word):
    file_names = [file_name for file_name in os.listdir(ord_dir) if (certain_word in file_name)]
    return file_names

def get_dir_dir_name(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if os.path.isdir(ord_dir+"/"+file_name) ]
    return file_names

def get_dir_certain_dir_name(ord_dir, certain_word):
    file_names = [file_name for file_name in os.listdir(ord_dir) if ((certain_word in file_name) and os.path.isdir(ord_dir+"/"+file_name)) ]
    return file_names


def get_dir_certain_img(ord_dir, certain_word, float_return =True):
    file_names = [file_name for file_name in os.listdir(ord_dir) if Check_img_filename(file_name) and (certain_word in file_name) ]
    img_list = []
    for file_name in file_names:
        img_list.append( cv2.imread(ord_dir + "/" + file_name) )
    if(float_return): img_list = np.array(img_list, dtype=np.float32)
    else:             img_list = np.array(img_list, dtype=np.uint8)
    return img_list

def get_dir_certain_move(ord_dir, certain_word):
    file_names = [file_name for file_name in os.listdir(ord_dir) if (".npy" in file_name) and (certain_word in file_name)]
    move_map_list = []
    for file_name in file_names:
        move_map_list.append( np.load(ord_dir + "/" + file_name) )
    move_map_list = np.array(move_map_list, dtype=np.float32)
    return move_map_list

def get_dir_img(ord_dir, float_return =False):
    file_names = [file_name for file_name in os.listdir(ord_dir) if Check_img_filename(file_name) ]
    img_list = []
    for file_name in tqdm(file_names):
        img_list.append( cv2.imread(ord_dir + "/" + file_name) )
    if(float_return): img_list = np.array(img_list, dtype=np.float32)
    else:             img_list = np.array(img_list, dtype=np.uint8)
    return img_list


def get_dir_move(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if ".npy" in file_name]
    move_map_list = []
    for file_name in file_names:
        move_map_list.append( np.load(ord_dir + "/" + file_name) )
    move_map_list = np.array(move_map_list, dtype=np.float32)
    return move_map_list
    
def get_dir_exr(ord_dir, rgb=False): ### 不要 float_return = True 之類的，因為他存的時候不一定用float32喔！rgb可以轉，已用網站生成的結果比較確認過囉～https://www.onlineconvert.com/exr-to-mat
    file_names = get_dir_certain_file_name(ord_dir, ".exr")

    imgs = []
    for file_name in file_names:
        img = cv2.imread(ord_dir + "/" + file_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED) ### 這行就可以了！
        if(rgb): img = img[...,::-1]    
        imgs.append(img)
        
    ### 不要轉dtype，因為不確定exr存的是啥米型態！
    # imgs = np.array(imgs, dtype=np.uint8) 
    # if(float_return): imgs = np.array(imgs, dtype=np.float32)
    return np.array(imgs)




def get_dir_mat(ord_dir, key):
    from hdf5storage import loadmat
    from util import get_dir_exr

    file_names = get_dir_certain_file_name(ord_dir, ".mat")
    imgs = []
    for file_name in file_names:
        mat = loadmat(ord_dir+"/"+file_name)
        imgs.append(mat[key])
    return np.array(imgs)
    

def get_db_amount(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if Check_img_filename(file_name) or (".npy" in file_name) ]
    return len(file_names)

##########################################################
def apply_move_map_boundary_mask(move_maps):
    boundary_width = 20 
    _, row, col = move_maps.shape[:3]
    move_maps[:, boundary_width:row-boundary_width,boundary_width:col-boundary_width,:] = 0
    return move_maps

def get_max_db_move_xy_from_numpy(move_maps): ### 注意這裡的 max/min 是找位移最大，不管正負號！ 跟 normalize 用的max/min 不一樣喔！ 
    move_maps = abs(move_maps)
    print("move_maps.shape",move_maps.shape)
    # move_maps = apply_move_map_boundary_mask(move_maps) ### 目前的dataset還是沒有只看邊邊，有空再用它來產生db，雖然實驗過有沒有用差不多(因為1019位移邊邊很大)
    max_move_x = move_maps[:,:,:,0].max()
    max_move_y = move_maps[:,:,:,1].max()
    return max_move_x, max_move_y

def get_max_db_move_xy_from_dir(ord_dir):
    move_maps = get_dir_move(ord_dir)
    return get_max_db_move_xy_from_numpy(move_maps)

def get_max_db_move_xy_from_certain_move(ord_dir, certain_word):
    move_maps = get_dir_certain_move(ord_dir, certain_word)
    return get_max_db_move_xy_from_numpy(move_maps)


def get_max_db_move_xy(db_dir="datasets", db_name="1_unet_page_h=384,w=256"):
    move_map_train_path = db_dir + "/" + db_name + "/" + "train/move_maps" 
    move_map_test_path  = db_dir + "/" + db_name + "/" + "test/move_maps" 
    train_move_maps = get_dir_move(move_map_train_path) # (1800, 384, 256, 2)
    test_move_maps  = get_dir_move(move_map_test_path)  # (200, 384, 256, 2)
    db_move_maps = np.concatenate((train_move_maps, test_move_maps), axis=0) # (2000, 384, 256, 2)

    max_move_x = db_move_maps[:,:,:,0].max()
    max_move_y = db_move_maps[:,:,:,1].max()
    return max_move_x, max_move_y

#######################################################
### 複刻 step6_data_pipline.py 寫的 get_train_test_move_map_db 
def get_maxmin_train_move_from_path(move_map_train_path):
    train_move_maps = get_dir_move(move_map_train_path)
    max_train_move = train_move_maps.max() ###  236.52951204508076
    min_train_move = train_move_maps.min() ### -227.09562801056995
    return max_train_move, min_train_move

def get_maxmin_train_move(db_dir="datasets", db_name="1_unet_page_h=384,w=256"):
    move_map_train_path = db_dir + "/" + db_name + "/" + "train/move_maps" 
    train_move_maps = get_dir_move(move_map_train_path)
    max_train_move = train_move_maps.max() ###  236.52951204508076
    min_train_move = train_move_maps.min() ### -227.09562801056995
    return max_train_move, min_train_move

#######################################################
### 用來給視覺化參考的顏色map
def get_reference_map( max_move=0, max_from_move_dir=False, move_dir="", x_decrease=False, y_decrease=False, color_shift=5): ### 根據你的db內 最大最小值 產生 參考流的map
    max_move = max_move
    if(max_from_move_dir) : max_move = find_db_max_move(move_dir)
    visual_row = 512
    visual_col = visual_row
    x = np.linspace(-max_move,max_move,visual_col)
    if(x_decrease): x = x[::-1]
    x_map = np.tile(x, (visual_row,1))

    y = np.linspace(-max_move,max_move,visual_col)
    if(y_decrease): y = y[::-1]
    y_map = np.tile(y, (visual_row,1))
    y_map = y_map.T

    map1 = method1(x_map, y_map, max_value=max_move)
    map2 = method2(x_map, y_map, color_shift=color_shift)
    return map1, map2, x_map, y_map

def find_db_max_move(ord_dir):
    move_map_list = get_dir_move(ord_dir)
    max_move = np.absolute(move_map_list).max()
    print("max_move:",max_move)
    return max_move

####################################################### 
### 視覺化方法1：感覺可以！但缺點是沒辦法用cv2，而一定要搭配matplot的imshow來自動填色
def method1(x, y, max_value=-10000): ### 這個 max_value的值 意義上來說要是整個db內位移最大值喔！這樣子出來的圖的顏色強度才會準確
    h, w = x.shape[:2]
    z = np.ones(shape=(h, w))
    visual_map = np.dstack( (x,y) )                  ### step1.
    if(max_value==-10000):                           ### step2.確定max_value值，沒有指定 max_value的話，就用資料自己本身的
        max_value = visual_map.max()
    visual_map = ((visual_map/max_value)+1)/2        ### step3.先把值弄到 0~1
    visual_map = np.dstack( (visual_map, z))         ### step4.再concat channel3，來給imshow自動決定顏色
#    plt.imshow(visual_map)
    return visual_map

### 視覺化方法2：用hsv，感覺可以！
def method2(x, y, color_shift=1):       ### 最大位移量不可以超過 255，要不然顏色強度會不準，不過實際用了map來顯示發現通常值都不大，所以還加個color_shift喔~
    h, w = x.shape[:2]                  ### 影像寬高
    fx, fy = x, y                       ### u是x方向怎麼移動，v是y方向怎麼移動
    ang = np.arctan2(fy, fx) + np.pi    ### 得到運動的角度
    val = np.sqrt(fx*fx+fy*fy)          ### 得到運動的位移長度
    hsv = np.zeros((h, w, 3), np.uint8) ### 初始化一個canvas
    hsv[...,0] = ang*(180/np.pi/2)      ### B channel為 角度訊息的顏色
    hsv[...,1] = 255                    ### G channel為 255飽和度
    hsv[...,2] = np.minimum(val*color_shift, 255)   ### R channel為 位移 和 255中較小值来表示亮度，因為值最大為255，val的除4拿掉就ok了！
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) ### 把得到的HSV模型轉換為BGR顯示
    if(True):
        white_back = np.ones((h, w, 3),np.uint8)*255
        white_back[...,0] -= hsv[...,2]
        white_back[...,1] -= hsv[...,2]
        white_back[...,2] -= hsv[...,2]
    #        cv2.imshow("white_back",white_back)
        bgr += white_back
    return bgr

#######################################################
def predict_unet_move_maps_back(predict_move_maps):
    from step0_access_path import access_path
    train_move_maps = get_dir_move(access_path+"datasets/pad2000-512to256/train/move_maps")
    max_train_move = train_move_maps.max()
    min_train_move = train_move_maps.min()
    predict_back_list = []
    for predict_move_map in predict_move_maps:
        predict_back = (predict_move_map[0]+1)/2 * (max_train_move-min_train_move) + min_train_move ### 把 -1~1 轉回原始的值域
        predict_back_list.append(predict_back)
    return np.array(predict_back_list, dtype=np.float32)



#######################################################

import matplotlib.pyplot as plt
import matplotlib as mpl 
mpl.rcParams["figure.max_open_warning"] = 0

def use_plt_show_move(move, color_shift=1):
    move_bgr = method2(move[:,:,0], move[:,:,1], color_shift=color_shift)
    move_rgb = move_bgr[:,:,::-1]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(move_rgb) ### 這裡不會秀出來喔！只是把圖畫進ax裡面而已
    return fig, ax



def time_util(cost_time):
    hour = cost_time//3600 
    minute = cost_time%3600//60 
    second = cost_time%3600%60
    return "%02i:%02i:%02i"%(hour, minute, second)

#######################################################
def _save_or_show(save, save_name, show):
    if(save==True and show==True):
        print("不能同時 save 又 show圖，預設用show圖囉！")
        plt.show()
    elif(save==True  and show==False): plt.savefig(save_name)
    elif(save==False and show==True ): plt.show()
    plt.close()


def Show_3d_scatter(one_channel_img, save=False, save_name="", show=False):
    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np 
    import cv2 

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(10,10)
    ax = Axes3D(fig)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z") ### 設定 x,y,z軸顯示的字
    # ax.set_zlim(-20, 30) ### 設定 z範圍

    ### 3D Scatter
    row, col = one_channel_img.shape[:2]
    x, y = get_xy_map(row,col)
    ax.scatter(x,y,one_channel_img, 
    # ax.scatter(x[one_channel_img!=0],y[one_channel_img!=0],one_channel_img[ one_channel_img!=0 ],  ### 這可以 挑 z !=0 的點再plot
               s=1,                     ### 點點的 大小
            #    linewidths = 1,        ### 點點的 邊寬
            #    edgecolors = "black"   ### 點點的 邊邊顏色
              c = np.arange(row*col),   ### 彩色
              )
    _save_or_show(save, save_name, show)

    ### 2D 直接show
    fig_img, ax_img = plt.subplots(1,1)
    ax_img.imshow(one_channel_img)
    _save_or_show(save, save_name+"-one_channel_img", show)



def Show_3d_scatter_along_xy(one_channel_img, along, save_name):
    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np 
    import cv2 
    import time

    ### 第一張圖：one_channel_img的長相
    fig_img, ax_img = plt.subplots(1,1) ### 建立新畫布
    ax_img.imshow(one_channel_img)      ### 畫上原影像

    ### 第二張圖：沿著x走一個個col顯示結果 或 沿著y走一個個row顯示結果
    row, col = one_channel_img.shape[:2]  ### 取得row, col
    x, y = get_xy_map(row,col)            ### 取得 x=[[0,1,2,...,col],[0,1,2,...,col],...,[0,1,2,...,col]] 和 y=[[0,0,...,0],[1,1,...,1],...,[row,row,...,row]]

    fig, ax = plt.subplots(1,1) ### 建立新畫布
    fig.set_size_inches(10,10)  ### 設定畫布大小
    ax = Axes3D(fig)            ### 轉成3D畫布
    ax.set_xlabel("x") ; ax.set_ylabel("y") ; ax.set_zlabel("z")    ### 設定 x,y,z軸顯示的字
    ax.set_xlim(0, col); ax.set_ylim(0, row); ax.set_zlim(-30,  30) ### 設定 x,y,z顯示的範圍

    plt.ion()
    plt.show()
    ### 沿著x走一個個col顯示結果
    if  (along=="x"): 
        for go_x in range(col): 
            print("go_x=",go_x)
            ax.scatter(np.ones(row)*go_x, y[:, go_x] ,one_channel_img[:, go_x], s=1,c = np.arange(row),)
            plt.pause(1)
    ### 沿著y走一個個row顯示結果
    elif(along=="y"): 
        for go_y in range(row): 
            print("go_y=",go_y)
            ax.scatter(x[go_y], np.ones(col)*go_y ,one_channel_img[go_y], s=1,c = np.arange(col),)
            plt.pause(1)
    
    plt.show()

def Show_2d_scatter_along_x(one_channel_img, save_name):
    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np 
    import cv2 

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(10,10)
    ax.set_xlabel("y"); ax.set_ylabel("z") ### 設定 x,y軸顯示的字
    # ax.set_zlim(-20, 30) ### 設定 z範圍


    row, col = one_channel_img.shape[:2]
    x, y = get_xy_map(row,col)
    ax.scatter(y[:,0],one_channel_img[:,0], 
               s=1,                     ### 點點的 大小
            #    linewidths = 1,        ### 點點的 邊寬
            #    edgecolors = "black"   ### 點點的 邊邊顏色
              c = np.arange(row),   ### 彩色
              )

    for go_x in range(col):
        ax.set_offsets(one_channel_img[:,go_x]) 

    fig_img, ax_img = plt.subplots(1,1)
    ax_img.imshow(one_channel_img)
    plt.show()


def Show_3d_bar(one_channel_img, save_name):
    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np 
    import cv2


    fig = plt.figure(0)
    fig.set_size_inches(10, 10)
    ax = Axes3D(fig)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_zlim(-20, 30)

    h = 360
    w = 270
    sub = 5
    one_channel_img = cv2.resize(one_channel_img, (int(w/sub), int(h/sub))) 
    print("one_channel_img.shape",one_channel_img.shape)
    height, width = one_channel_img.shape[:2]
    draw_x = np.zeros(one_channel_img.shape[:2]) + np.arange(width ).reshape(1,-1) 
    ### draw_x 長得像：
    # [[ 0.  1.  2. ... 37. 38. 39.]
    # [ 0.  1.  2. ... 37. 38. 39.]
    # [ 0.  1.  2. ... 37. 38. 39.]
    # ...
    # [ 0.  1.  2. ... 37. 38. 39.]
    # [ 0.  1.  2. ... 37. 38. 39.]
    # [ 0.  1.  2. ... 37. 38. 39.]]
    draw_y = np.zeros(one_channel_img.shape[:2]) + np.arange(height).reshape(-1,1)
    ### draw_y 長得像：
    # [[ 0.  0.  0. ...  0.  0.  0.]
    # [ 1.  1.  1. ...  1.  1.  1.]
    # [ 2.  2.  2. ...  2.  2.  2.]
    # ...
    # [37. 37. 37. ... 37. 37. 37.]
    # [38. 38. 38. ... 38. 38. 38.]
    # [39. 39. 39. ... 39. 39. 39.]]

    ### ravel是拉平的意思，相當於flatten的概念
    ax.bar3d( draw_x.ravel(), draw_y.ravel(), np.zeros(height*width), 1, 1, one_channel_img.ravel())#s = 1,edgecolors = "black")

    cv2.imshow("one_channel_img",one_channel_img)
    # plt.savefig( save_name+".png" )
    plt.show()


def Show_move_map_apply(move_map):
    import matplotlib.pyplot as plt
    row, col = move_map.shape[:2] ### 取得 row, col
    x, y = get_xy_map(row, col)   ### 取得 x, y 起始座標
    xy = np.dstack((x, y))        ### concat起來
    xy_move = xy + move_map       ### apply move


    fig, ax = plt.subplots(1,1)  ### 建立新圖
    ax.set_title("move_map_apply") ### 設定圖的title
    # ax_img = ax.scatter(xy_move[...,0],xy_move[...,1]) ### 單色
    ax_img = ax.scatter(xy_move[...,0],xy_move[...,1], c = np.arange(row*col).reshape(row,col), cmap="brg") ### 彩色
    ax = ax.invert_yaxis() ### 整張圖上下顛倒，為了符合影像是左上角(0,0)
    fig.colorbar(ax_img,ax=ax)
    plt.show()

import matplotlib.cm as cmx
import matplotlib.colors as colors
def get_cmap(color_amount, cmap_name='hsv'):
    '''Returns a function that maps each index in 0, 1,.. . N-1 to a distinct
    RGB color.
    '''
    color_norm = colors.Normalize(vmin=0, vmax=color_amount-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap_name)
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

############################################################################################################
############################################################################################################
class Matplot_ax_util():
    @staticmethod
    def Draw_ax_loss_during_train( ax, logs_dir, cur_epoch, epochs , ylim=LOSS_YLIM ): ### logs_dir 不能改丟 result_obj喔！因為See裡面沒有Result喔！
        x_epoch = np.arange(cur_epoch+1) ### x座標畫多少，畫到目前訓練的 cur_epoch，+1是為了index轉數量喔
        
        logs_file_names = get_dir_certain_file_name(logs_dir, "npy") ### 去logs_dir 抓 當時訓練時存的 loss.npy
        for loss_i, logs_file_name in enumerate(logs_file_names):
            y_loss_array = np.load( logs_dir + "/" + logs_file_name) ### 去logs_dir 抓 當時訓練時存的 loss.npy
            loss_name = logs_file_name.split(".")[0]
            Matplot_ax_util._Draw_ax_loss(ax, cur_epoch, loss_name, loss_i, x_array=x_epoch, y_array=y_loss_array, xlim=epochs, ylim=ylim)
    
    
    @staticmethod
    ### 注意這會給 see, result, c_results 用喔！ 所以多 result的情況也要考慮，所以才要傳 min_epochs，
    ### 且因為有給see用，logs_dir 不能改丟 result_obj喔！因為See裡面沒有Result喔！
    def Draw_ax_loss_after_train( ax, logs_dir, cur_epoch, min_epochs , ylim=LOSS_YLIM ): 
        x_epoch = np.arange(min_epochs) ### x座標畫多少
        
        logs_file_names = get_dir_certain_file_name(logs_dir, "npy") ### 去logs_dir 抓 當時訓練時存的 loss.npy
        for loss_i, logs_file_name in enumerate(logs_file_names):
            y_loss_array = np.load( logs_dir + "/" + logs_file_name)  ### 把loss讀出來
            loss_amount = len(y_loss_array)     ### 訓練的當下存了多少個loss
            if( (min_epochs-1) == loss_amount): ### 如果現在result剛好是訓練最少次的result，要注意有可能訓練時中斷在存loss前，造成 epochs數 比 loss數 多一個喔！這樣畫圖會出錯！
                y_loss_array = np.append(y_loss_array, y_loss_array[-1]) ### 把loss array 最後補一個自己的尾巴
            y_loss_array_used = y_loss_array[:min_epochs] ### 補完後，別忘了考慮多result的情況，result裡挑最少量的的loss數量 來show
            loss_name = logs_file_name.split(".")[0]

            # print("len(x_epoch)", len(x_epoch))
            # print("len(y_loss_array_used)", len(y_loss_array_used))
            Matplot_ax_util._Draw_ax_loss(ax, cur_epoch, loss_name, loss_i, x_array=x_epoch, y_array=y_loss_array_used, xlim=min_epochs, ylim=ylim)
    
    @staticmethod
    def _Draw_ax_loss(ax, cur_epoch, loss_name, loss_i, x_array, y_array, xlim, ylim=LOSS_YLIM, x_label="epoch loss avg", y_label="epoch_num"):
        cmap = get_cmap(8)  ### 隨便一個比6多的數字，嘗試後8的顏色分布不錯！
        plt.sca(ax)  ### plt指向目前的 小畫布 這是為了設定 xylim 和 xylabel
        plt.ylim(0, ylim) ;plt.ylabel( x_label )
        plt.xlim(0, xlim) ;plt.xlabel( y_label ) 

        ### 畫線
        ax.plot(x_array, y_array, c=cmap(loss_i), label=loss_name) 
        ### 畫點
        ax.scatter(cur_epoch, y_array[cur_epoch], color=cmap(loss_i))
        ### 點旁邊註記值
        ax.annotate( s="%.3f" % y_array[cur_epoch],      ### 顯示的文字
                     xy=(cur_epoch, y_array[cur_epoch]), ### 要標註的目標點
                     xytext=( 0 , 10*loss_i),         ### 顯示的文字放哪裡
                     textcoords='offset points',         ### 目前東西放哪裡的坐標系用什麼
                     arrowprops=dict(arrowstyle="->",    ### 畫箭頭的資訊
                                    connectionstyle= "arc3",
                                    color = cmap(loss_i),
                                    ))
        ### 
        ax.legend(loc='best')
            

class Matplot_fig_util(Matplot_ax_util):
    @staticmethod
    def Save_fig(dst_dir, epoch):
        plt.savefig(dst_dir+"/"+"epoch=%04i"%epoch )
        plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！

class Matplot_util(Matplot_fig_util): pass


class Matplot_single_row_imgs(Matplot_fig_util):
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

    def _step1_build_check(self):
        #### 防呆 ####################################################
        if( self.col_titles_amount < self.col_imgs_amount):
            for _ in range(self.col_imgs_amount - self.col_titles_amount):
                self.img_titles.append("")

        elif(self.col_titles_amount > self.col_imgs_amount):
            print("title 太多了，沒有圖可以對應")
            return 

        if(self.col_imgs_amount == 0): 
            print("沒圖可show喔！")
            return 
        ###########################################################

    def _get_one_row_canvas_height(self):
        height_list = []    ### imgs是個list，裡面放的圖片可能不一樣大喔
        for img in self.imgs: height_list.append(img.shape[0])
        return  (max(height_list) // 100+1.0)*1.0 +1.5 ### 慢慢試囉～ +1.5是要給title 和 matplot邊界margin喔

    def _get_one_row_canvas_width(self):
        width = 0
        for img in self.imgs: width += img.shape[1]

        if  (self.col_imgs_amount==3): return  (width // 100 +0)*1.0 +5.7 ### 慢慢試囉～ col=3時
        elif(self.col_imgs_amount==4): return  (width // 100 +0)*1.0 +6.8 ### 慢慢試囉～ col=4時
        elif(self.col_imgs_amount==5): return  (width // 100 +0)*1.0 +8.5 ### 慢慢試囉～ col=5時
        elif(self.col_imgs_amount==6): return  (width // 100 +0)*1.0 +10.5 ### 慢慢試囉～ col=6時
        elif(self.col_imgs_amount==7): return  (width // 100 +0)*1.0 +11.5 ### 慢慢試囉～ col=7時
        elif(self.col_imgs_amount >7): return  (width // 100 +0)*1.0 +11.5 ### 慢慢試囉～ col=7時，沒有試過用猜的，因為覺得用不到ˊ口ˋ用到再來試

    def _step2_set_canvas_hw_and_build(self):
        ### 設定canvas的大小
        self.canvas_height = self._get_one_row_canvas_height()
        self.canvas_width  = self._get_one_row_canvas_width()
        if(self.add_loss):   ### 多一些空間來畫loss
            self.canvas_height += 3   ### 慢慢試囉～ 
            self.canvas_width  -= 1.5*self.col_imgs_amount  ### 慢慢試囉～ 
            self.row_imgs_amount += 1 ### 多一row來畫loss
        # print("canvas_height",canvas_height)
        # print("canvas_width",canvas_width)
        # print("row_imgs_amount", row_imgs_amount)
        # print("col_imgs_amount", col_imgs_amount)

        ### 建立canvas出來
        self.fig, self.ax = plt.subplots(nrows=self.row_imgs_amount, ncols=self.col_imgs_amount)
        self.fig.set_size_inches(self.canvas_width, self.canvas_height) ### 設定 畫布大小
        
    def _step3_draw(self, used_ax):
        ### 這就是手動微調 text的位置囉ˊ口ˋ
        self.fig.text(x=0.5, y=0.945, s=self.fig_title, fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
        
        for go_img, img in enumerate(self.imgs):
            if(self.bgr2rgb):img[...,::-1] ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
            if(self.col_imgs_amount > 1):
                used_ax[go_img].imshow(img) ### 小畫布 畫上影像，別忘記要bgr -> rgb喔！
                used_ax[go_img].set_title( self.img_titles[go_img], fontsize=16 ) ### 小畫布上的 title
                
                plt.sca(used_ax[go_img])  ### plt指向目前的 小畫布 這是為了設定 yticks和xticks
                plt.yticks( (0, img.shape[0]), (0, img.shape[0]) )  ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                plt.xticks( (0, img.shape[1]), ("", img.shape[1]) ) ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
            else:
                used_ax.imshow(img) ### 小畫布 畫上影像
                used_ax.set_title( self.img_titles[go_img], fontsize=16 ) ### 小畫布上的 title
                
                plt.yticks( (0, img.shape[0]), (0, img.shape[0]) )  ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                plt.xticks( (0, img.shape[1]), ("", img.shape[1]) ) ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字

    def Draw_img(self):
        ###############################################################
        ### 注意 _draw_single_row_imgs 的 ax 只能丟 一row，所以才寫這if/else
        if(not self.add_loss): used_ax = self.ax
        elif(self.add_loss):   used_ax = self.ax[0]  ### 只能丟第一row喔！因為_draw_single_row_imgs 裡面的操作方式 是 一row的方式，丟兩row ax維度會出問題！
        self._step3_draw(used_ax)
        ###############################################################
        ### 想畫得更漂亮一點，兩種還是有些一咪咪差距喔~
        if(not self.add_loss): self.fig.tight_layout(rect=[0,0,1,0.93])
        else:                  self.fig.tight_layout(rect=[0,0.006,1,0.95])
        ###############################################################
        ### Draw_img完，不一定要馬上Draw_loss喔！像是train的時候 就是分開的 1.see(Draw_img), 2.train, 3.loss(Draw_loss)






### imgs是個list，裡面放的圖片可能不一樣大喔
def _get_one_row_canvas_height(imgs):
    height_list = []
    for img in imgs: height_list.append(img.shape[0])
    # return  (max(height_list) // 100+2.0)*0.8  ### 沒有弄得很精準，+1好了
    return  (max(height_list) // 100+1.0)*1.0 +1.5 ### 慢慢試囉～ +1.5是要給title 和 matplot邊界margin喔

def _get_one_row_canvas_width(imgs):
    width = 0
    for img in imgs: width += img.shape[1]

    if  (len(imgs)==3): return  (width // 100 +0)*1.0 +5.7 ### 慢慢試囉～ col=3時
    elif(len(imgs)==4): return  (width // 100 +0)*1.0 +6.8 ### 慢慢試囉～ col=4時
    elif(len(imgs)==5): return  (width // 100 +0)*1.0 +8.5 ### 慢慢試囉～ col=5時
    elif(len(imgs)==6): return  (width // 100 +0)*1.0 +10.5 ### 慢慢試囉～ col=6時
    elif(len(imgs)==7): return  (width // 100 +0)*1.0 +11.5 ### 慢慢試囉～ col=7時
    elif(len(imgs) >7): return  (width // 100 +0)*1.0 +11.5 ### 慢慢試囉～ col=7時，沒有試過用猜的，因為覺得用不到ˊ口ˋ用到再來試


def _get_row_col_canvas_height(r_c_imgs):
    height = 0
    for row_imgs in r_c_imgs: height += row_imgs[0].shape[0]
    return (height // 100 +0)*1.2  ### 慢慢試囉～ +1.5是要給title 和 matplot邊界margin喔

def _get_row_col_canvas_width(r_c_imgs):
    width = 0
    for col_imgs in r_c_imgs[0]: width += col_imgs.shape[1]
    return (width //100 + 1)*1.2 ### 慢慢試囉～

### single_row 的處理方式 還是跟 multi_row 有些許不同，所以不能因為時做出 multi後取代single喔！ 比如 ax[] 的維度、取長寬比之類的～
def _draw_single_row_imgs(fig, ax, col_imgs_amount, canvas_height, canvas_width, img_titles, imgs, fig_title="epoch = 1005", bgr2rgb=True):
    ### 這就是手動微調 text的位置囉ˊ口ˋ
    fig.text(x=0.5, y=0.945, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
    
    for go_img, img in enumerate(imgs):
        if(bgr2rgb):img[...,::-1] ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
        if(col_imgs_amount > 1):
            ax[go_img].imshow(img) ### 小畫布 畫上影像，別忘記要bgr -> rgb喔！
            ax[go_img].set_title( img_titles[go_img], fontsize=16 ) ### 小畫布上的 title
            
            plt.sca(ax[go_img])  ### plt指向目前的 小畫布 這是為了設定 yticks和xticks
            plt.yticks( (0, img.shape[0]), (0, img.shape[0]) )  ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
            plt.xticks( (0, img.shape[1]), ("", img.shape[1]) ) ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
        else:
            ax.imshow(img) ### 小畫布 畫上影像
            ax.set_title( img_titles[go_img], fontsize=16 ) ### 小畫布上的 title
            
            plt.yticks( (0, img.shape[0]), (0, img.shape[0]) )  ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
            plt.xticks( (0, img.shape[1]), ("", img.shape[1]) ) ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字


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
        canvas_height += 3   ### 慢慢試囉～ 
        canvas_width  -= 1.5*col_imgs_amount  ### 慢慢試囉～ 
        row_imgs_amount += 1 ### 多一row來畫loss
    # print("canvas_height",canvas_height)
    # print("canvas_width",canvas_width)
    # print("row_imgs_amount", row_imgs_amount)
    # print("col_imgs_amount", col_imgs_amount)

    ### 建立canvas出來
    fig, ax = plt.subplots(nrows=row_imgs_amount, ncols=col_imgs_amount)
    fig.set_size_inches(canvas_width, canvas_height) ### 設定 畫布大小
    ###############################################################
    ### 注意 _draw_single_row_imgs 的 ax 只能丟 一row，所以才寫這if/else
    if(not add_loss): used_ax = ax
    elif(add_loss):   used_ax = ax[0]  ### 只能丟第一row喔！因為_draw_single_row_imgs 裡面的操作方式 是 一row的方式，丟兩row ax維度會出問題！
    _draw_single_row_imgs(fig, used_ax, col_imgs_amount, canvas_height, canvas_width, img_titles, imgs, fig_title, bgr2rgb)
    ###############################################################
    ### 想畫得更漂亮一點，兩種還是有些一咪咪差距喔~
    if(not add_loss): fig.tight_layout(rect=[0,0,1,0.93])
    else:             fig.tight_layout(rect=[0,0.006,1,0.95])
    ###############################################################
    ### 統一不存，因為可能還要給別人後續處理，這裡只負責畫圖喔！
    # plt.savefig(dst_dir+"/"+file_name)
    # plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！
    return fig, ax


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

        if(len(self.r_c_imgs)==1):
            print("本function 不能處理 single_row_imgs喔，因為matplot在row只有1時的維度跟1以上時不同！麻煩呼叫相對應處理single_row的function！")

    def _get_row_col_canvas_height():
        height = 0
        for row_imgs in self.r_c_imgs: height += row_imgs[0].shape[0]
        return (height // 100 +0)*1.2  ### 慢慢試囉～ +1.5是要給title 和 matplot邊界margin喔

    def _get_row_col_canvas_width(r_c_imgs):
        width = 0
        for col_imgs in self.r_c_imgs[0]: width += col_imgs.shape[1]
        return (width //100 + 1)*1.2 ### 慢慢試囉～

    def _step2_set_canvas_hw_and_build(self):
        ###########################################################
        ### 設定canvas的大小
        self.canvas_height = _get_row_col_canvas_height(self.r_c_imgs)
        self.canvas_width  = _get_row_col_canvas_width (self.r_c_imgs)
        if(self.add_loss):   ### 多一些空間來畫loss
            self.canvas_height += 3.0  ### 慢慢試囉～
            self.canvas_width  -= 0.55*self.col_imgs_amount  ### 慢慢試囉～
            self.canvas_height *= 1.1 #1.2最好，但有點佔記憶體  ### 慢慢試囉～ 
            self.canvas_width  *= 1.1 #1.2最好，但有點佔記憶體  ### 慢慢試囉～
            self.row_imgs_amount += 1 ### 多一row來畫loss
        # print("canvas_height",canvas_height)
        # print("canvas_width",canvas_width)
        # print("row_imgs_amount", row_imgs_amount)

        ### 建立canvas出來
        self.fig, self.ax = plt.subplots(nrows=self.row_imgs_amount, ncols=self.col_imgs_amount)
        self.fig.set_size_inches(self.canvas_width, self.canvas_height) ### 設定 畫布大小

    def _step3_draw(self):
        ### 這就是手動微調 text的位置囉ˊ口ˋ
        self.fig.text(x=0.5, y=0.95, s=self.fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)

        for go_row, row_imgs in enumerate(self.r_c_imgs): 
            for go_col, col_img in enumerate(row_imgs):
                if(self.bgr2rgb):col_img[...,::-1] ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
                if(self.col_imgs_amount > 1):
                    self.ax[go_row, go_col].imshow(col_img) ### 小畫布 畫上影像，別忘記要bgr -> rgb喔！
                    if  (len(self.r_c_titles) >1): self.ax[go_row, go_col].set_title( self.r_c_titles[go_row][go_col], fontsize=16 ) ### 小畫布　標上小標題
                    elif(len(self.r_c_titles)==1 and go_row==0):self.ax[go_row, go_col].set_title( self.r_c_titles[go_row][go_col], fontsize=16 ) ### 小畫布　標上小標題
                    
                    plt.sca(self.ax[go_row, go_col])  ### plt指向目前的 小畫布 這是為了設定 yticks和xticks
                    plt.yticks( (0, col_img.shape[0]), (0, col_img.shape[0]) )  ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                    plt.xticks( (0, col_img.shape[1]), ("", col_img.shape[1]) ) ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                else: ### 要多這if/else是因為，col_imgs_amount==1時，ax[]只會有一維！用二維的寫法會出錯！所以才獨立出來寫喔～
                    self.ax[go_row].imshow(col_img) ### 小畫布 畫上影像
                    if  (len(self.r_c_titles) >1): self.ax[go_row].set_title( self.r_c_titles[go_row][go_col], fontsize=16 ) ### 小畫布　標上小標題
                    elif(len(self.r_c_titles)==1 and go_row==0): self.ax[go_row].set_title( self.r_c_titles[go_row][go_col], fontsize=16 ) ### 小畫布　標上小標題
                    plt.yticks( (0, col_img.shape[0]), (0, col_img.shape[0]) )  ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                    plt.xticks( (0, col_img.shape[1]), ("", col_img.shape[1]) ) ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字

    def Draw_img(self):
        self._step3_draw()
        if(not self.add_loss): self.fig.tight_layout(rect=[0,0,1,0.95]) ### 待嘗試喔！
        else:                  self.fig.tight_layout(rect=[0,0.0035,1,0.95]) ### 待嘗試喔！
        ###############################################################
        ### Draw_img完，不一定要馬上Draw_loss喔！但 multi的好像可以馬上Draw_loss~ 不過想想還是general一點分開做好了~~




def _draw_multi_row_imgs(fig, ax, row_imgs_amount, col_imgs_amount, canvas_height, canvas_width, rows_cols_titles, rows_cols_imgs, fig_title="epoch = 1005", bgr2rgb=True):
### 這就是手動微調 text的位置囉ˊ口ˋ
    fig.text(x=0.5, y=0.95, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
    # if  (col_imgs_amount <  3):fig.text(x=0.5, y=0.92, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
    # elif(col_imgs_amount == 3):fig.text(x=0.5, y=0.91, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
    # elif(col_imgs_amount >  3):
    #     if  (row_imgs_amount <  3):fig.text(x=0.5, y=0.915, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
    #     elif(row_imgs_amount == 3):fig.text(x=0.5, y=0.90 , s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)
    #     elif(row_imgs_amount >  3):fig.text(x=0.5, y=0.897, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',) ### 再往下覺得用不到就沒有試囉ˊ口ˋ有用到再來微調八~~
    
    for go_row, row_imgs in enumerate(rows_cols_imgs): 
        for go_col, col_img in enumerate(row_imgs):
            if(bgr2rgb):col_img[...,::-1] ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
            if(col_imgs_amount > 1):
                ax[go_row, go_col].imshow(col_img) ### 小畫布 畫上影像，別忘記要bgr -> rgb喔！
                if  (len(rows_cols_titles) >1): ax[go_row, go_col].set_title( rows_cols_titles[go_row][go_col], fontsize=16 ) ### 小畫布　標上小標題
                elif(len(rows_cols_titles)==1 and go_row==0):ax[go_row, go_col].set_title( rows_cols_titles[go_row][go_col], fontsize=16 ) ### 小畫布　標上小標題
                
                plt.sca(ax[go_row, go_col])  ### plt指向目前的 小畫布 這是為了設定 yticks和xticks
                plt.yticks( (0, col_img.shape[0]), (0, col_img.shape[0]) )  ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                plt.xticks( (0, col_img.shape[1]), ("", col_img.shape[1]) ) ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
            else: ### 要多這if/else是因為，col_imgs_amount==1時，ax[]只會有一維！用二維的寫法會出錯！所以才獨立出來寫喔～
                ax[go_row].imshow(col_img) ### 小畫布 畫上影像
                if  (len(rows_cols_titles) >1): ax[go_row].set_title( rows_cols_titles[go_row][go_col], fontsize=16 ) ### 小畫布　標上小標題
                elif(len(rows_cols_titles)==1 and go_row==0): ax[go_row].set_title( rows_cols_titles[go_row][go_col], fontsize=16 ) ### 小畫布　標上小標題
                plt.yticks( (0, col_img.shape[0]), (0, col_img.shape[0]) )  ### 設定 y軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字
                plt.xticks( (0, col_img.shape[1]), ("", col_img.shape[1]) ) ### 設定 x軸 顯示的字，前面的tuple是位置，後面的tuple是要顯示的字

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

    if(len(rows_cols_imgs)==1):
        print("本function 不能處理 single_row_imgs喔，因為matplot在row只有1時的維度跟1以上時不同！麻煩呼叫相對應處理single_row的function！")
    ###########################################################
    ### 設定canvas的大小
    canvas_height = _get_row_col_canvas_height(rows_cols_imgs)
    canvas_width  = _get_row_col_canvas_width (rows_cols_imgs)
    if(add_loss):   ### 多一些空間來畫loss
        canvas_height += 3.0  ### 慢慢試囉～
        canvas_width  -= 0.55*col_imgs_amount  ### 慢慢試囉～
        canvas_height *= 1.2  ### 慢慢試囉～ 
        canvas_width  *= 1.2  ### 慢慢試囉～
        row_imgs_amount += 1 ### 多一row來畫loss
    # print("canvas_height",canvas_height)
    # print("canvas_width",canvas_width)
    # print("row_imgs_amount", row_imgs_amount)

    ### 建立canvas出來
    fig, ax = plt.subplots(nrows=row_imgs_amount, ncols=col_imgs_amount)
    fig.set_size_inches(canvas_width, canvas_height) ### 設定 畫布大小
    ###############################################################
    _draw_multi_row_imgs(fig, ax, row_imgs_amount, col_imgs_amount, canvas_height, canvas_width, rows_cols_titles, rows_cols_imgs, fig_title, bgr2rgb)
    ###############################################################
    ### 想畫得更漂亮一點，兩種還是有些一咪咪差距喔~
    if(not add_loss): fig.tight_layout(rect=[0,0,1,0.95]) ### 待嘗試喔！
    else:             fig.tight_layout(rect=[0,0.0035,1,0.95]) ### 待嘗試喔！
    ###############################################################
    ### 統一不存，因為可能還要給別人後續處理，這裡只負責畫圖喔！
    # plt.show()
    # plt.savefig(dst_dir+"/"+file_name)
    # plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！
    return fig, ax



def draw_loss_util(fig, ax, logs_dir, epoch, epochs ):
    x_epoch = np.arange(epochs)

    logs_file_names = get_dir_certain_file_name(logs_dir, "npy")
    y_loss_array = np.load( logs_dir + "/" + logs_file_names[0])
    
    plt.sca(ax)  ### plt指向目前的 小畫布 這是為了設定 xylim 和 xylabel
    plt.ylim(0,LOSS_YLIM)     ;plt.ylabel(logs_file_names[0])
    plt.xlim(0, epochs) ;plt.xlabel("epoch_num") 
    ax.plot(x_epoch, y_loss_array)
    ax.scatter(epoch, y_loss_array[epoch], c="red")
    return fig, ax

############################################################################################################
############################################################################################################


def multi_processing_interface(core_amount, task_amount, task, task_args=None, print_msg=False):
    from multiprocessing import Process
    processes = [] ### 放 Process 的 list
    split_amount = int(task_amount //core_amount) ### split_amount 的意思是： 一個core 可以"分到"幾個任務，目前的想法是 一個core對一個process，所以下面的process_amount 一開始設定==split_amount喔！
    fract_amount = int(task_amount % core_amount) ### fract_amount 的意思是： 任務不一定可以均分給所有core，分完後還剩下多少個任務沒分出來

    for go_core_i in range(core_amount):
        ### 決定 core_start_index 和 core_task_amount：
        ###     core_start_index：core 要處理的任務的 start_index
        ###     core_task_amount：core 要處理的任務數量
        if(core_amount >= task_amount): 
            core_start_index = go_core_i ### 如果 core的數量 比 任務數量多，一個任務一個core
            core_task_amount = 1         ### 如果 core的數量 比 任務數量多，一個任務一個core
            if(go_core_i >= task_amount):break ### 任務分完了，就break囉！要不然沒任務分給core拉
            
        elif( core_amount < task_amount):
            core_start_index = split_amount*go_core_i  
            core_task_amount = split_amount            ### 一個process 要處理幾個任務，目前的想法是 一個core對一個process，所以 一開始設定==split_amount喔！
            if(fract_amount != 0): ### 如果 任務分完後還剩下任務沒分完
                if  (go_core_i == 0): core_task_amount += fract_amount  ### 把 沒分完的任務給第一個core！因為在分配Process給core的過程也會花時間，這時間就可以給第一個core處理分剩的任務囉！
                elif(go_core_i  > 0): core_start_index += fract_amount  ### 第一個後的core 任務 index 就要做點位移囉！
            ### 下面這寫法是把 沒分完的任務給第最後一個core，這樣最後的core最慢被分到又要做最多事情，會比較慢喔～
            # if( go_core_i==(core_amount-1) and (fract_amount!=0) ): core_task_amount += fract_amount ### process分配到最後 如果 task_amount 還有剩，就加到最後一個process
        
        if(task_args is None):processes.append(Process( target=task, args=(core_start_index, core_task_amount) ) ) ### 根據上面的 core_start_index 和 core_task_amount 來 創建 Process
        else:                 processes.append(Process( target=task, args=(core_start_index, core_task_amount, *task_args) ) ) ### 根據上面的 core_start_index 和 core_task_amount 來 創建 Process
        if(print_msg): print("registering process_%02i dealing %04i~%04i task"% (go_core_i, core_start_index, core_start_index+core_task_amount-1) ) ### 大概顯示這樣的資訊：registering process_00 dealing 0000~0003 task

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if(__name__=="__main__"):
    from step0_access_path import access_path
    # in_imgs = get_dir_img(access_path+"datasets/wei_book/in_imgs")
    # gt_imgs = get_dir_img(access_path+"datasets/wei_book/gt_imgs")
    
    # db = zip(in_imgs, gt_imgs)
    # for imgs in db:
    #     print(type(imgs))

    get_max_db_move_xy(db_dir=access_path+"datasets", db_name="1_unet_page_h=384,w=256")