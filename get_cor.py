import os
import openai
import evaluate
import json
import re
import multiprocessing
import torch
import time


from utils import (create_logger, paras_filter_by_belu, SrcRefPreCor, API_KEY,
                   TASK, save_result, get_response, get_prompt_input,
                   get_result_from_response)

# nohup ./get_cor.py > get_cor.log 2>&1 &

openai.api_key = API_KEY[0]

task = TASK[0]
## filter paras by bleu
LOW_BLEU = 4
HIGH_BLEU = 24

DEBUG = False

lang_pairs = "ur-en"
data_path = f"mono_corr/{lang_pairs}-100w/filter/"

logger = create_logger(filepath=f'./tr-en-log/{lang_pairs}-mono-cor.log')

def load_paras_from_torch(path):
    assert os.path.exists(path), f"doesn't exist {path}"
    paras = torch.load(path)
    return paras
def filte_by_bleu(src_ref_pre, compute_batch, low_bleu, high_bleu):
    logger.info(f"low bleu is {low_bleu} and high bleu is {high_bleu}")
    bleu = compute_batch(ref=[x[1] for x in src_ref_pre], pre=[x[2] for x in src_ref_pre])
    logger.info(f"平均bule值为{sum(bleu) / len(bleu)}")
    logger.info(f"经过bleu过滤之前{len(src_ref_pre)}")
    src_ref_pre_filt = paras_filter_by_belu(src_ref_pre, bleu, patience=low_bleu, high=high_bleu)
    logger.info(f"经过过滤之后的{len(src_ref_pre_filt)}")
    return src_ref_pre_filt


def sleep(i_id_thread):
    t = 10 + i_id_thread *2
    t = 45 if t>45 else t
    logger.info(f"子进程{i_id_thread}开始睡眠了， 睡{t}秒")
    time.sleep(t)

def thread_fun(i_id_thread, span=r'\d+\.', num_sent_per_api=8, res=None, paras=None, cor=None, prompts=None):
    """res 记录每次调用API的返回结果, paras 要调用的句子对， cor记录每条句子对应的返回结果"""
    res = [] if res==None else res
    cor = [] if cor==None else cor
    prompts = [] if prompts==None else prompts
    num_error = 0

    total_num_api = len(paras) // num_sent_per_api
    total_num_api = total_num_api + 1 if (len(paras) % num_sent_per_api)!=0 else total_num_api
    for i in range(total_num_api):
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
                else:           #不知名的错误，保存跑路
                    logger.error(f"{i_id_thread} error message {e.user_message}")
                    sleep(i_id_thread)
                    res = [f"total_num_sentence:{len(paras)},  num_error_sentence: {num_error}"] + res
                    continue
        if i % 3 == 0:          
            logger.info(f"子进程{i_id_thread}正在第{i}次调用api 已获取结果")
        
        prompts.append(prompt_input)
        res.append(str(res_context))  # 结果记录

        # res_context = res_context.choices[0].text             # text-davinci-003 结果记录方式
                             
        cor_res, num_err = get_result_from_response(res_context, send_paras, span)
        for p, c in zip(paras[i*num_sent_per_api : (i+1)* num_sent_per_api], cor_res):                  # 组装成4元组
            if task == TASK[0]:
                p.add_cor(c)
            elif task == TASK[1]:
                p.add_pre(c)
            else:
                assert task in TASK
        cor += cor_res
        num_error += num_err

        s_res = [f"total_num_sentence:{len(paras)},  num_error_sentence: {num_error}"]
        if i%5 == 0:                                            #每5次保存一下
            save_result(i_id_thread, data_path, task, s_res+res, paras, cor, prompts)
    res = s_res + res
    save_result(i_id_thread, data_path, task, res, paras, cor, prompts)


if __name__ == '__main__':
    src_ref_pre_path = os.path.join(data_path, "src-ref-pre.bin")

    logger.info(src_ref_pre_path)
    
    src_ref_pre = load_paras_from_torch(src_ref_pre_path)
    logger.info("load src_ref_pre three tuple over")

    src_ref_pre_filt = src_ref_pre

    ## ! 使用类封装以下
    if not isinstance(src_ref_pre_filt[0], SrcRefPreCor):
        src_ref_pre_filt = [SrcRefPreCor(s, r, p) for s, r, p in src_ref_pre_filt]
    ## ! debug
    if DEBUG:
        src_ref_pre_filt = src_ref_pre_filt[:5]

    ## 设置num_thread
    num_thread = 16 if not DEBUG else 2
    num_sent_per_api = 12 if not DEBUG else 2        # 没次调用api最多使用的句子
    num_sent_pre_thread = len(src_ref_pre_filt) // num_thread
    # 将句子列表拆分，给每个子线程一个句子列表， 最后一个进程要把剩下的全部包括进去
    thread_src_ref_pre = [src_ref_pre_filt[i*num_sent_pre_thread : (i+1)*num_sent_pre_thread] \
                      if i!=num_thread-1 else src_ref_pre_filt[i*num_sent_pre_thread : ]   \
                      for i in range(0, num_thread) ]

    logger.info(f"every thread process num of sentence {num_sent_pre_thread}, the last thread process num {len(thread_src_ref_pre[-1])}")
    logger.info(f"每次调用api处理的句子数{num_sent_per_api}")

    pool = multiprocessing.Pool(processes=num_thread)
    processes = []
    for i in range(num_thread):
        p = pool.apply_async(thread_fun, (i, r'\d+\.', num_sent_per_api,
                                          None, thread_src_ref_pre[i], None))
        processes.append(p)
    pool.close()
    pool.join()
    for p in processes:
        p.get()