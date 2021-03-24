import pdb
def _get_triangle_list(num):  ### 不想被外面的.py 呼叫，所以加底線
    tri_amount = num  ## 換個名字比較好理解
    tri_height_list = []
    while(tri_amount != 0):  ### 狀況一：如果一開始的數字就0 直接break，狀況二：如果是第二輪以上，數字非0就繼續找，為0就代表找完了
        ### 找出最接近 tri_amount 的 三角形高度
        tri_acc_temp = 0
        for go_tri in range(1, tri_amount + 1):
            tri_acc_temp += go_tri
            if(tri_acc_temp > tri_amount):
                tri_height_list.append(go_tri - 1)  ### 超過總數了，所以紀錄 go_tri - 1 為 我們要的三角形高度
                break
            elif(tri_acc_temp == tri_amount):
                tri_height_list.append(go_tri)      ### 剛好等於總數讚讚！，直接紀錄這個go_tri 為 我們要的三角形高度
                break

        ### 把triangle 做出來，為下一輪要找的tri_amount做準備
        triangle = 0
        for i in range(1, tri_height_list[-1] + 1):
            triangle += i

        ### 下一輪的 tri_amount = 把這輪的三角形總數扣掉，看還剩多少要找
        tri_amount -= triangle

        ### DEBUG用
        # print("triangle:", triangle, ", tri_amount:", tri_amount)
        # breakpoint()
    return tri_height_list

def _merge_tri_list(num, tri_height_list):  ### 不想被外面的.py 呼叫，所以加底線
    merge_list = [0] * num
    for tri_height in tri_height_list:
        for go_tri in range(tri_height):
            merge_list[go_tri] += tri_height - go_tri
    return merge_list

def get_tri_task_list(core_amount, fract_amount):
    tri_height_list = _get_triangle_list(fract_amount)
    merge_list = _merge_tri_list(core_amount, tri_height_list)
    return merge_list



if(__name__ == "__main__"):  ### 這邊是測試用，所以有呼叫 _ 的function喔！
    for i in range(100):
        tri_height_list = _get_triangle_list(i)
        print(i, tri_height_list)
        merge_list = _merge_tri_list(i, tri_height_list)
        print(merge_list)
        print()
