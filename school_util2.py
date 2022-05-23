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
        ["01", 110621017, "李育綺",  7],
        ["02", 110624219, "簡加祐", 18],
        ["03", 208418011, "邱崇碩",  7],
        ["04", 404411059, "王竑迪",  4],
        ["05", 406410943, "林讓言",  8],
        ["06", 406500339, "吳承恩",  7],
        ["07", 407030419, "劉兆崴", 15],
        ["08", 407410181, "謝宗昱", 17],
        ["09", 407430718, "陳堂安", 10],
        ["10", 408106242, "郭婉梅", 10],
        ["11", 408410560, "吳俊逸",  7],
        ["12", 408416013, "金度漢",  8],
        ["13", 408416518, "黃柏惀", 12],
        ["14", 408734019, "陶奕臻", 10],
        ["15", 409410031, "張云瀚", 11],
        ["16", 409410064, "陳宜謙", 10],
        ["17", 409410098, "林品睿",  8],
        ["18", 409410122, "張博堯", 11],
        ["19", 409410155, "黃品寧", 12],
        ["20", 409410189, "周義翔",  8],
        ["21", 409410213, "陳建良", 10],
        ["22", 409410247, "莊宜璇", 11],
        ["23", 409410270, "湯鈞凱", 12],
        ["24", 409410304, "姜彥銘",  9],
        ["25", 409410320, "陳軍翰", 10],
        ["26", 409410338, "楊舒涵", 13],
        ["27", 409410361, "林嘉榆",  8],
        ["28", 409410395, "吳念詠",  7],
        ["29", 409410429, "邱信翰",  7],
        ["30", 409410486, "李元禎",  7],
        ["31", 409410510, "游家碩", 12],
        ["32", 409410544, "李鴻君",  7],
        ["33", 409410577, "梁婕綿", 11],
        ["34", 409410601, "凌煜淳", 10],
        ["35", 409410635, "周勁含",  8],
        ["36", 409410668, "張弘謀", 11],
        ["37", 409410692, "許瀚文", 11],
        ["38", 409410726, "洪維澤",  9],
        ["39", 409410759, "李至楷",  7],
        ["40", 409410783, "林威達",  8],
        ["41", 409410817, "許博瑄", 11],
        ["42", 409410841, "林威霖",  8],
        ["43", 409410874, "陳光翊", 10],
        ["44", 409410908, "許家瑜", 11],
        ["45", 409410932, "許銘津", 11],
        ["46", 409410965, "林廷恩",  8],
        ["47", 409410999, "王翔麟",  4],
        ["48", 409411021, "馬儒彬", 10],
        ["49", 409411054, "周庭蔚",  8],
        ["50", 409411088, "温晨軒", 12],
        ["51", 409411112, "陳佩榆", 10],
        ["52", 409411146, "蔡弘杰", 15],
        ["53", 409411179, "黃晨芯", 12],
        ["54", 409411203, "林子祐",  8],
        ["55", 409411237, "吳宜謙",  7],
        ["56", 409411260, "朱柏宇",  6],
        ["57", 409411351, "吳信篁",  7],
        ["58", 409411385, "蔣易軒", 15],
        ["59", 409411476, "廖浚棋", 14],
        ["60", 409411500, "林冠宇",  8],
        ["61", 409411534, "呂仲衡",  7],
        ["62", 409411567, "黃冠瑄", 12],
        ["63", 409411625, "范皓翔",  9],
        ["64", 409411658, "許廷安", 11],
        ["65", 409411682, "李佳容",  7],
        ["66", 409411716, "巫冠君",  7],
        ["67", 409411740, "賴柏全", 16],
        ["68", 409411773, "葉耿全", 13],
        ["69", 409415014, "韓定恩", 17],
        ["70", 409415030, "齋藤巧", 17],
        ["71", 409415105, "劉富田", 15],
        ["72", 409416061, "葉展強", 13],
        ["73", 409416095, "鄧智恒", 14],
        ["74", 409416186, "梁俊彥", 11],
        ["75", 409416533, "陳皓圓", 10],
        ["76", 409417051, "陳翊嘉", 10],
        ["77", 409418034, "林沂臻",  8],
        ["78", 409418083, "王姵文",  4],
        ["79", 409418182, "劉昱成", 15],
        ["80", 410417017, "曾翊瑋", 12],
        ["81", 410417033, "簡岑芳", 18],
        ["82", 410417124, "徐  敬", 10],
        ["83", 410417165, "周聖庭",  8],
        ["84", 410418031, "傅譯賢", 12],
        ["85", 410418064, "林博涵",  8],
        ["86", 410418106, "林弘翔",  8],
        ["87", 410418171, "呂坤璘",  7],
    ]
    classmates_sorted = sorted( classmates, key= lambda classmate: classmate[3] )


    for classmate in classmates_sorted:
        print(classmate)



電腦01
['04', 404411059, '王竑迪', 4,   9]
['78', 409418083, '王姵文', 4,   9]
['47', 409410999, '王翔麟', 4,  12]
['56', 409411260, '朱柏宇', 6,   9]
['55', 409411237, '吳宜謙', 7,   8]
['06', 406500339, '吳承恩', 7,   8]
['28', 409410395, '吳念詠', 7,   8]
['11', 408410560, '吳俊逸', 7,   9]
['57', 409411351, '吳信篁', 7,   9]

電腦02
['61', 409411534, '呂仲衡', 7,   6]
['87', 410418171, '呂坤璘', 7,   8]
['66', 409411716, '巫冠君', 7,   9]
['30', 409410486, '李元禎', 7,   4]
['39', 409410759, '李至楷', 7,   6]
['01', 110621017, '李育綺', 7,   7]
['65', 409411682, '李佳容', 7,   8]
['32', 409410544, '李鴻君', 7,  17]
['29', 409410429, '邱信翰', 7,   9]

電腦03
['03', 208418011, '邱崇碩', 7,  11]
['35', 409410635, '周勁含', 8,   9]
['49', 409411054, '周庭蔚', 8,  10]
['20', 409410189, '周義翔', 8,  13]
['83', 410417165, '周聖庭', 8,  13]
['54', 409411203, '林子祐', 8,   3]
['86', 410418106, '林弘翔', 8,   5]
['46', 409410965, '林廷恩', 8,   7]
['77', 409418034, '林沂臻', 8,   7]

電腦04
['60', 409411500, '林冠宇', 8,   9]
['17', 409410098, '林品睿', 8,   9]
['40', 409410783, '林威達', 8,   9]
['42', 409410841, '林威霖', 8,   9]
['85', 410418064, '林博涵', 8,  12]
['27', 409410361, '林嘉榆', 8,  14]
['05', 406410943, '林讓言', 8,  24]
['12', 408416013, '金度漢', 8,   9]
['24', 409410304, '姜彥銘', 9,   9]

電腦05
['63', 409411625, '范皓翔', 9,  12]
['38', 409410726, '洪維澤', 9,  14]
['82', 410417124, '徐  敬', 10,  0]
['10', 408106242, '郭婉梅', 10, 11]
['34', 409410601, '凌煜淳', 10, 13]
['48', 409411021, '馬儒彬', 10, 16]
['43', 409410874, '陳光翊', 10,  6]
['51', 409411112, '陳佩榆', 10,  8]
['16', 409410064, '陳宜謙', 10,  8]

電腦06
['25', 409410320, '陳軍翰', 10,  9]
['21', 409410213, '陳建良', 10,  9]
['09', 407430718, '陳堂安', 10, 11]
['76', 409417051, '陳翊嘉', 10, 11]
['75', 409416533, '陳皓圓', 10, 12]
['14', 408734019, '陶奕臻', 10,  9]
['15', 409410031, '張云瀚', 11,  4]
['36', 409410668, '張弘謀', 11,  5]
['18', 409410122, '張博堯', 11, 12]

電腦07
['74', 409416186, '梁俊彥', 11,  9]
['33', 409410577, '梁婕綿', 11, 11]
['22', 409410247, '莊宜璇', 11,  8]
['64', 409411658, '許廷安', 11,  7]
['44', 409410908, '許家瑜', 11, 10]
['41', 409410817, '許博瑄', 11, 12]
['45', 409410932, '許銘津', 11, 14]
['37', 409410692, '許瀚文', 11, 19]
['84', 410418031, '傅譯賢', 12, 20]

電腦08
['80', 410417017, '曾翊瑋', 12, 11]
['31', 409410510, '游家碩', 12, 10]
['23', 409410270, '湯鈞凱', 12, 12]
['62', 409411567, '黃冠瑄', 12,  9]
['19', 409410155, '黃品寧', 12,  9]
['13', 408416518, '黃柏惀', 12,  9]
['53', 409411179, '黃晨芯', 12, 11]
['50', 409411088, '温晨軒', 12, 11]
['26', 409410338, '楊舒涵', 13, 12]

電腦09
['68', 409411773, '葉耿全', 13, 10]
['72', 409416061, '葉展強', 13, 10]
['59', 409411476, '廖浚棋', 14, 10]
['73', 409416095, '鄧智恒', 14, 12]
['07', 407030419, '劉兆崴', 15,  6]
['79', 409418182, '劉昱成', 15,  9]
['71', 409415105, '劉富田', 15, 12]
['52', 409411146, '蔡弘杰', 15,  5]
['58', 409411385, '蔣易軒', 15,  8]

電腦10
['67', 409411740, '賴柏全', 16,  9]
['08', 407410181, '謝宗昱', 17,  8]
['69', 409415014, '韓定恩', 17,  8]
['70', 409415030, '齋藤巧', 17, 19]
['02', 110624219, '簡加祐', 18,  5]
['81', 410417033, '簡岑芳', 18,  7]