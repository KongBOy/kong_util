import numpy as np

def Pring_classmate_grouping(classmates, group_amount, print_msg=False):
    classmates = np.array(classmates)
    '''
    classmates 結構大概長這樣：
    classmates = [[1, 206410465, "溫  儒"],
                  [2, 207440388, "林永泰"],
                  [3, 404081134, "羅培瑋"],
                  [4, 405416131, "羅海鉞"],
                  [5, 405416511, "鄭又綸"],
                  [6, 405416545, "蕭聖儒"],......]
    '''

    people_amount = len(classmates)  ### 人數
    np.random.shuffle(classmates)    ### 洗牌
    groups = np.split(classmates, range(0, people_amount, np.ceil(people_amount / group_amount).astype(np.uint8))[1:])  ### 分組， split用法：https://ithelp.ithome.com.tw/articles/10203624

    if(print_msg): print("range(0, people_amount, people_amount // group_amount)", np.arange(0, people_amount, people_amount // group_amount)[1:])
    # for classmate in classmates:
    #     print(classmate)

    ### 每組內部自己排序
    sorted_group = []
    for go_p, group in enumerate(groups):
        sorted_group.append(sorted( group, key= lambda classmate: int(classmate[0])))

    ### 每組內的人show出來
    for go_p, group in enumerate(sorted_group):
        print(f"{chr(ord('A') + go_p)}組, {len(group)} 人, Total group:{len(groups)}")
        for classmate in group:
            print(classmate)
        print("")


if(__name__ == "__main__"):
    classmates = [
        [1, 206410465, "溫  儒"],
        [2, 207440388, "林永泰"],
        [3, 404081134, "羅培瑋"],
        [4, 405416131, "羅海鉞"],
        [5, 405416511, "鄭又綸"],
        [6, 405416545, "蕭聖儒"],
        [7, 406380484, "鄭皓予"],
        [8, 406390442, "朱俊翰"],
        [9, 406410372, "陳彥甫"],
        [10, 406410638, "李昀錡"],
        [11, 406410786, "林清峰"],
        [12, 406411248, "𡍼永呈"],
        [13, 406416023, "趙崇樂"],
        [14, 406416510, "雲冠惟"],
        [15, 406510171, "陳瑀欣"],
        [16, 406854017, "王朵含"],
        [17, 407400596, "謝立仁"],
        [18, 407410454, "陳宜秀"],
        [19, 407410603, "楊家宏"],
        [20, 407410652, "李天文"],
        [21, 407410744, "賴彥銘"],
        [22, 407410892, "平星皓"],
        [23, 407411072, "朱冠霖"],
        [24, 407411270, "李可名"],
        [25, 407417194, "林仲廷"],
        [26, 407510055, "韓侑丞"],
        [27, 407510469, "賴思潔"],
        [28, 408408051, "林裕恩"],
        [29, 408411287, "李承謙"],
        [30, 408411469, "林伯翰"],
        [31, 408416013, "金度漢"],
        [32, 409416590, "蔡承城"],
        [33, "A09991223", "林均樺"],
        [34, "A09993146", "葛柏毅"],
        [35, "A09993229", "林瑋倫"],
        [36, "A09997287", "楊濟華"],
        [37, "A09997436", "邱英展"],
        [38, "A09997824", "鄧福翔"],
        [39, "A09997998", "蔡翔安"],
        [40, "A09998145", "盧承濬"],
        [41, "A09998194", "盧鋐霖"],
        [42, "A09998277", "馬梓曦"],
        [43, "A09998855", "王鉦傑"],
        [44, "A09999325", "盖建坤"],
        [45, 206410440, "陳峻賢"],
        # [46, 408416518, "黃柏惀"],
        [46, 208410257, "何冠勳"],
        [47, 408411444, "王偉安"],
        [48, 408416518, "黃柏惀 不見了"]]


    Pring_classmate_grouping(classmates, group_amount=5)
