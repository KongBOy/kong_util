# def get_triangle_list(num):
#     acc_num = 0
#     tri_list = []
#     while(acc_num != num):
#         acc = 0
#         for i in range(num):
#             acc += i
#             if(acc > num):
#                 tri_list.append(i - 1)
#                 break

#         for i in range(1, tri_list[-1] + 1):
#             acc_num += i
#         num -= acc_num

import multiprocessing
from multiprocessing import Process

def multi_processing_interface(core_amount, task_amount, task, task_start_index=0, task_args=None, print_msg=False):
    if(print_msg):
        print("core_amount:", core_amount)
        print("task_amount:", task_amount)
        print("task_start_index:", task_start_index)
        print("task:", task.__name__)
        print("task_args:", task_args)
    '''
    理論上 core 就是 worker，但實際上我寫的想法是 core 可以無限切，但是實際的 worker 就要看cpu 是有限的囉！
    core切多的狀況 用在：當一個core要做的事情 需要的記憶體太大時，比如：wc 100000(十萬) 張，切 500 個 core之類的 一個core 只需要處理 200 張wc的記憶體 這樣子，但實際worker 就是固定的這樣子 (我設定下面 core_count*2+2)

    task_amount 和 task_start_amount 的關係 應該要在 外面就自己算清楚囉！
    例如 500個任務，想從 第2個開始，task_start_index=1，task_amount外面就要自己算好丟499喔！
    '''
    processes = []   ### 放 Process 的 list
    split_amount = int(task_amount // core_amount)  ### split_amount 的意思是： 一個core 可以"分到"幾個任務，目前的想法是 一個core對一個process，所以下面的process_amount 一開始設定==split_amount喔！
    fract_amount = int(task_amount % core_amount)   ### fract_amount 的意思是： 任務不一定可以均分給所有core，分完後還剩下多少個任務沒分出來


    ### 給方法4用的
    from kong_util.multiprocess_triangle_task_util import get_tri_task_list
    tri_task_list = get_tri_task_list(core_amount, fract_amount)
    if(print_msg): print("tri_task_list:", tri_task_list)

    current_index = 0  ### 給方法3, 4 用的

    for go_core_i in range(core_amount):
        ### 決定 core_start_index 和 core_task_amount：
        ###     core_start_index：core 要處理的任務的 start_index
        ###     core_task_amount：core 要處理的任務數量
        if(core_amount >= task_amount):   ### 如果 core的數量 比 任務數量多 或 一樣 的情況
            core_start_index = go_core_i  ### 如果 core的數量 比 任務數量多，一個任務一個core
            core_task_amount = 1          ### 如果 core的數量 比 任務數量多，一個任務一個core
            if(go_core_i >= task_amount): break  ### 任務分完了，就break囉！要不然沒任務分給core拉

        elif( core_amount < task_amount):  ### 如果 core的數量 比 任務數量多 少 的情況
            ### 在同core的情況下，快的排序應該是：4 > 2 > 3 > 1
            ########################################################################################################################
            ### 寫法4：越前面的core分配越多任務
            core_start_index = current_index
            core_task_amount = split_amount + tri_task_list[go_core_i]

            current_index += core_task_amount      ### 準備 下一個 core 的 start_index
            ########################################################################################################################
            ### 寫法3：最值觀的想法，把fraction 平均分給每個前面的core，但因為分配任務給core需要時間，越前面的core通常會越早做完，就要等後面的core，有點浪費！
            # core_start_index = current_index
            # core_task_amount = split_amount

            # if(go_core_i < fract_amount): core_task_amount += 1  ### 把 fract_amount 平均分給 前面的 core
            # current_index += core_task_amount                    ### 準備 下一個 core 的 start_index

            ########################################################################################################################
            ### 寫法2：fraction 全部都丟給 第一個core，因為分配任務給core也需要時間，所以第一個丟多一點在分配的過程中也可以做事情，缺點是如果 core_amount 越多，fraction數就可能越大，可能第一個被分到的任務太多，最後一個core都做完了 第一個core還沒做完
            # core_start_index = split_amount * go_core_i  ### 定出 task_index 起始位置
            # core_task_amount = split_amount            ### 一個process 要處理幾個任務，目前的想法是 一個core對一個process，所以 一開始設定==split_amount喔！
            # if(fract_amount != 0): ### 如果 任務分完後還剩下任務沒分完
            #     if  (go_core_i == 0): core_task_amount += fract_amount  ### 把 沒分完的任務給第一個core！因為在分配Process給core的過程也會花時間，這時間就可以給第一個core處理分剩的任務囉！
            #     elif(go_core_i  > 0): core_start_index += fract_amount  ### 第一個後的core 任務 index 就要做點位移囉！

            ########################################################################################################################
            ### 寫法1
            ### 下面這寫法是把 沒分完的任務給第最後一個core，這樣最後的core最慢被分到又要做最多事情，會比較慢喔～
            # if( go_core_i == (core_amount-1) and (fract_amount!=0) ): core_task_amount += fract_amount ### process分配到最後 如果 task_amount 還有剩，就加到最後一個process

        if(task_args is None): processes.append(Process( target=task, args=(task_start_index + core_start_index, core_task_amount) ) )              ### 起始點從 task_start_index 開始，根據上面的 core_start_index 和 core_task_amount 來 創建 Process
        else:                  processes.append(Process( target=task, args=(task_start_index + core_start_index, core_task_amount, *task_args) ) )  ### 起始點從 task_start_index 開始，根據上面的 core_start_index 和 core_task_amount 來 創建 Process
        if(print_msg): print("registering process_%02i dealing %04i~%04i task" % (go_core_i, task_start_index + core_start_index, task_start_index + core_start_index + core_task_amount - 1) )  ### 大概顯示這樣的資訊：registering process_00 dealing 0000~0003 task

    ###############################################################################################################################################################################################
    ###############################################################################################################################################################################################
    ###############################################################################################################################################################################################
    ###############################################################################################################################################################################################
    ### 方法2，看某個worker做完，馬上分process給他做
    max_worker = multiprocessing.cpu_count() * 2 + 2  ### 應該吧~~ 通常都 1_core 2_thread，但實際嘗試後發現 通常 比 core*thread 大也沒問題 且 較容易 cpu 100 % 運轉
    worker_amount = min(core_amount, max_worker)      ### 一開始註解說的，core_amount 可以無限切，但實際的worker數是有限的！
    if(print_msg): print(f"worker_amount:{worker_amount} (core_amount:{core_amount}, max_worker:{max_worker})")



    process_amount = len(processes)  ### core_amount == process_amount ， 但以語意上理解，後面用 process_amount 較好思考，所以就多用了 len(processes) 這樣子囉～
    workers = [Process()] * worker_amount  ### 模擬一開始沒事做的 worker
    ### 補充：不能用 for go_p, process in enumerate(processes):  ### 因為如果 worker全在忙 ， 用for 就會強制跳到下一個 process 了！
    ### 所以要用while 才對， 直到 process 被分配了才 跳到下一個 process
    go_p = 0  ### current process index
    while(go_p < process_amount):  ### 一定要用while來寫 才代表直到 process 被分配了， 才跳到下一個process喔
        ### 如果process 一直沒有被分配， 就會一直跑 worker 迴圈， 直到找到 worker處理process 為止！
        for worker_id, worker in enumerate(workers):
            if(worker.is_alive() is False):  ### 一開始 worker 沒做事 .is_alive() 為 False 或 後來做完事情了， .is_alive() 會變False
                ### 指定新 Process 給 沒事做 或 做完事情的 worker
                workers[worker_id] = processes[go_p]
                workers[worker_id].start()  ### .is_alive() 會變True
                if(print_msg): print(" workers[%i] doing %i/%i process is starting" % (worker_id, go_p + 1, process_amount))
                go_p += 1   ### 換下一個Process
                break   ### 目前的 Process 已分配完畢， 可以換下一個Process了，所以就break出 worker迴圈 囉！

    ### 方法1，但還是要等前面的process做完 才分配 下一個 process
    # for go_p, process in enumerate(processes):
        # if( (go_p + 1) % 8 == 0):
        #     for go_stop_p in range(go_p+1):
        #         if(processes[go_stop_p].is_alive()):
        #             processes[go_stop_p].join()
        #     time.sleep(10)

    for process in processes:  ### 一定要join喔！要不然先做完的process 可能會偷跑！如果下一個任務需要等前面的任務跑完 就會出錯！
        process.join()
