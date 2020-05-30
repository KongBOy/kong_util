import numpy as np  
import cv2 
import os

from tqdm import tqdm

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


### imgs是個list，裡面放的圖片可能不一樣大喔
def _get_one_row_canvas_height(imgs):
    height_list = []
    for img in imgs: height_list.append(img.shape[0])
    # return  (max(height_list) // 100+2.0)*0.8  ### 沒有弄得很精準，+1好了
    return  (max(height_list) // 100+1.0)*0.9 ### 慢慢試囉～
def _get_one_row_canvas_width(imgs):
    width = 0
    for img in imgs: width += img.shape[1]
    # return  (width // 100 +3)*0.8### 沒有弄得很精準，+1好了
    return  (width // 100 +0)*0.9 ### 慢慢試囉～

def _get_row_col_canvas_height(r_c_imgs):
    height = 0
    for row_imgs in r_c_imgs: height += row_imgs[0].shape[0]
    return (height // 100 +0)*0.9 ### 慢慢試囉～
    

def _get_row_col_canvas_width(r_c_imgs):
    width = 0
    for col_imgs in r_c_imgs[0]: width += col_imgs.shape[1]
    return (width //100 + 1)*0.9 ### 慢慢試囉～

### single_row 的處理方式 還是跟 multi_row 有些許不同，所以不能因為時做出 multi後取代single喔！ 比如 ax[] 的維度、取長寬比之類的～
def matplot_visual_single_row_imgs(img_titles, imgs, fig_title="epoch = 1005", dst_dir=".", file_name="one_row_img.png", bgr2rgb=True):
    title_amount = len(img_titles)
    img_amount   = len(imgs)

    #### 防呆 ####################################################
    if( title_amount < img_amount):
        for _ in range(img_amount - title_amount):
            img_titles.append("")
    elif(title_amount > img_amount):
        print("title 太多了，沒有圖可以對應")
        return 
    
    if(img_amount == 0): 
        print("沒圖可show喔！")
        return 
    ###########################################################

    canvas_height = _get_one_row_canvas_height(imgs)*2.0
    canvas_width  = _get_one_row_canvas_width(imgs)*2.0
    # print("canvas_height",canvas_height)
    # print("canvas_width",canvas_width)
    
    fig, ax = plt.subplots(nrows=1, ncols=img_amount)
    ### 這就是手動微調 text的位置囉ˊ口ˋ
    if  (img_amount <  3):fig.text(x=0.5, y=0.95, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)#, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    elif(img_amount == 3):fig.text(x=0.5, y=0.90, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)#, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    elif(img_amount >  3):fig.text(x=0.5, y=0.90, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)#, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    fig.set_size_inches(canvas_width, canvas_height) ### 設定 畫布大小
    
    
    for go_img, img in enumerate(imgs):
        if(bgr2rgb):img[...,::-1] ### 如果有標示 輸入進來的 影像是 bgr，要轉rgb喔！
        if(img_amount > 1):
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

    plt.savefig(dst_dir+"/"+file_name)
    plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！
    

def matplot_visual_multi_row_imgs(rows_cols_titles, rows_cols_imgs, fig_title="epoch=1005", dst_dir=".", file_name="one_row_img.png", bgr2rgb=True):
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
    canvas_height = _get_row_col_canvas_height(rows_cols_imgs)
    canvas_width  = _get_row_col_canvas_width (rows_cols_imgs)
    # print("canvas_height",canvas_height)
    # print("canvas_width",canvas_width)
    
    fig, ax = plt.subplots(nrows=row_imgs_amount, ncols=col_imgs_amount)
    ### 這就是手動微調 text的位置囉ˊ口ˋ
    if  (col_imgs_amount <  3):fig.text(x=0.5, y=0.95, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)#, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    elif(col_imgs_amount == 3):fig.text(x=0.5, y=0.94, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)#, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    elif(col_imgs_amount >  3):#fig.text(x=0.5, y=0.93, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)#, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if  (row_imgs_amount <  3):fig.text(x=0.5, y=0.93, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)#, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        elif(row_imgs_amount == 3):fig.text(x=0.5, y=0.92, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)#, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        elif(row_imgs_amount >  3):fig.text(x=0.5, y=0.907, s=fig_title,fontsize=20, c=(0.,0.,0.,1.),  horizontalalignment='center',)#, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    fig.set_size_inches(canvas_width, canvas_height) ### 設定 畫布大小
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
    # plt.show()
    plt.savefig(dst_dir+"/"+file_name)
    plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！


def multi_processing_interface(core_amount, task_amount, task, task_args=None):
    from multiprocessing import Process
    processes = [] ### 放 Process 的 list

    for go_core_i in range(core_amount):
        ### 決定 core_start_index 和 core_task_amount：
        ###     core_start_index：core 要處理的任務的 start_index
        ###     core_task_amount：core 要處理的任務數量
        if(core_amount >= task_amount): 
            core_start_index = go_core_i ### 如果 core的數量 比 任務數量多，一個任務一個core
            core_task_amount = 1         ### 如果 core的數量 比 任務數量多，一個任務一個core
            if(go_core_i >= task_amount):break ### 任務分完了，就break囉！要不然沒任務分給core拉
            
        elif( core_amount < task_amount):
            split_amount = int(task_amount //core_amount) ### split_amount 的意思是： 一個core 可以"分到"幾個任務，目前的想法是 一個core對一個process，所以下面的process_amount 一開始設定==split_amount喔！
            fract_amount = int(task_amount % core_amount) ### fract_amount 的意思是： 任務不一定可以均分給所有core，分完後還剩下多少個任務沒分出來

            core_start_index = split_amount*go_core_i  
            core_task_amount = split_amount            ### 一個process 要處理幾個任務，目前的想法是 一個core對一個process，所以 一開始設定==split_amount喔！
            if( go_core_i==(core_amount-1) and (fract_amount!=0) ): core_task_amount += fract_amount ### process分配到最後 如果 task_amount 還有剩，就加到最後一個process

        if(task_args is None):processes.append(Process( target=task, args=(core_start_index, core_task_amount) ) ) ### 根據上面的 core_start_index 和 core_task_amount 來 創建 Process
        else:                 processes.append(Process( target=task, args=(core_start_index, core_task_amount, *task_args) ) ) ### 根據上面的 core_start_index 和 core_task_amount 來 創建 Process
        print("registering process_%02i dealing %04i~%04i task"% (go_core_i, core_start_index, core_start_index+core_task_amount-1) ) ### 大概顯示這樣的資訊：registering process_00 dealing 0000~0003 task

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