"""
在使用get_cor.py调用api的时候会出现不知名的错误而退出，此文件就是要接着继续调用
"""

import os
import openai
import evaluate
import json
import re
import multiprocessing
from requests import HTTPError
import torch
import time


from utils import (create_logger,
                   SrcRefPreCor,
                   API_KEY,
                   TASK,
                   get_prompt_input,
                   get_response,
                   get_result_from_response,
                   save_result)


openai.api_key = API_KEY[0]

task = TASK[1]
## filter paras by bleu
LOW_BLEU = 4
HIGH_BLEU = 24

DEBUG = False

logger = create_logger(filepath='./tr-en-log/ka-en-test-continue.log')

def load_paras_from_torch(path):
    assert os.path.exists(path)
    paras = torch.load(path)
    return paras

data_path = "test/ka-en/"

def sleep(i_id_thread):
    t = 10 + i_id_thread * 3
    t = 40 if t>40 else t
    logger.info(f"子进程{i_id_thread}开始睡眠了， 睡{t}秒")
    time.sleep(t)

def thread_fun(i_id_thread, span=r'\d+\.', num_sent_per_api=12, res=None, paras=None, cor=None, prompts=None, num_error=0, step=0):
    """res 记录每次调用API的返回结果, paras 要调用的句子对， cor记录每条句子对应的返回结果, step从第step步开始"""
    res = [] if res==None else res
    cor = [] if cor==None else cor
    prompts = [] if prompts==None else prompts

    total_num_api = len(paras) // num_sent_per_api
    total_num_api = total_num_api + 1 if (len(paras) % num_sent_per_api)!=0 else total_num_api
    if DEBUG:
        total_num_api = min(step+1, total_num_api)
    for i in range(step, total_num_api):
        send_paras = paras[i*num_sent_per_api : (i+1)* num_sent_per_api]
        prompt_input, length = get_prompt_input(send_paras, task)    #取出pre打包
        while (True):           # 应对查询速率限制
            try:
                res_context = get_response(prompt_input)
                break
            except Exception as e:
                if isinstance(e, openai.error.RateLimitError) and \
                    'You exceeded your current quota, please check your plan and billing details.' in e.user_message:
                    logger.critical("你没钱了")
                    break
                else:
                    logger.error(f" {i_id_thread} error message {e}")
                    logger.info(" 睡一会，稍后继续")
                    sleep(i_id_thread)
                    continue
        if i % 3 == 0:          
            logger.info(f"子进程{i_id_thread}正在第{i}次调用api 已获取结果")
        prompt_input = re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f\u0130\u0131]+",u"",prompt_input)
        prompts.append(prompt_input)
        res.append(str(res_context))  # 结果记录

        # res_context = res_context.choices[0].text             # text-davinci-003 结果记录方式
                             
        cor_res, num_err = get_result_from_response(res_context, send_paras, span)

        for p, c in zip(paras[i*num_sent_per_api : (i+1)* num_sent_per_api], cor_res):  #实时加入
            if task == TASK[0]:
                p.add_cor(c)
            elif task == TASK[1]:
                p.add_pre(c)

        cor += cor_res
        num_error += num_err
        s_res = [f"total_num_sentence:{len(paras)},  num_error_sentence: {num_error}"]
        if i%5 == 0:    
            save_result(i_id_thread, data_path, task, s_res+res, paras, cor, prompts)
    res = [f"total_num_sentence:{len(paras)},  num_error_sentence: {num_error}"] + res
    if not DEBUG:
        save_result(i_id_thread, data_path, task, res, paras, cor, prompts)

def get_cor_continue(file_path, i_id_thread, file_id, span=r'\d+\.', num_sent_per_api=12):
    """
    file_path: 文件夹地址
    """
    def read_file(p):
        data = []
        with open(p, 'r', encoding="utf") as f:
            data = f.readlines()
        data = [d.lstrip("\n").rstrip("\n") for d in data]
        return data
    assert os.path.exists(file_path)
    res_path = os.path.join(file_path, f"res-{file_id}.txt")
    paras_path = os.path.join(file_path, f"paras-{file_id}.bin")
    prompts_path = os.path.join(file_path, f"prompts-{file_id}.txt")
    if task == TASK[0]:
        cor_path = os.path.join(file_path, f"cor-{file_id}.txt")
        cor = read_file(cor_path)
    else:
        cor=None
    res = read_file(res_path)
    prompts = read_file(prompts_path)
    
    paras = torch.load(paras_path)

    total_num_api = len(paras) // num_sent_per_api
    total_num_api = total_num_api + 1 if (len(paras) % num_sent_per_api)!=0 else total_num_api
    for step in range(total_num_api+1):# 寻找开始继续调用的地方
        send_paras = paras[step*num_sent_per_api : (step+1)* num_sent_per_api]
        if task == TASK[0]:                
            if all([p.cor == None for p in send_paras]):
                break
        elif task == TASK[1]:
            if all([p.pre == None for p in send_paras]):
                break
    if step == total_num_api + 1:                           # 此进程都已经查完了，没有需要继续调用的例子了
        logger.critical(f"进程{i_id_thread}的数据已经处理完了，无需继续")
        return
    num_error = res[0]
    res = res[1:]
    num_error = num_error[num_error.find("num_error_sentence")+ len("num_error_sentence") + 1:]
    num_error = int(num_error.lstrip(" "))
    thread_fun(file_id, span, num_sent_per_api, res, paras, cor, prompts, num_error, step)

if __name__ == '__main__':

    logger.info(f"开始进行继续调用 {data_path}")

    ## 设置num_thread
    # num_thread = 12 if not DEBUG else 1
    # num_sent_per_api = 12 if not DEBUG else 2        # 没次调用api最多使用的句子
    num_sent_per_api = 2

    prefix = "thread"
    
    dir_list = [f_p for f_p in os.listdir(data_path) if f_p.startswith(prefix)]
    thread_path = [os.path.join(data_path, f"{prefix}{i}") for i in range(len(dir_list))]
    file_id = [i for i in range(len(thread_path))]
    # file_id = [1,2,5,11]
    # thread_path = [os.path.join(data_path, f"{prefix}{i}") for i in file_id]

    logger.info(f"处理的进程的文件夹为{thread_path}")
    logger.info(f"每次调用api处理的句子数{num_sent_per_api}")

    num_thread = len(thread_path)                       # 更改为有多少是需要继续的
    if DEBUG:
        num_thread = 1

    pool = multiprocessing.Pool(processes=num_thread)
    processes = []

    for i in range(num_thread):
        p = pool.apply_async(get_cor_continue, (thread_path[i], i, file_id[i], r'\d+\.', num_sent_per_api))
        processes.append(p)
    pool.close()
    pool.join()
    for p in processes:
        p.get()