from datetime import timedelta
import os
import openai
import evaluate
import json
import re
import multiprocessing
import torch
import time
import logging

from utils import create_logger ,paras_filter_by_belu, SrcRefPreCor

"""
文件加载、保存用的torch.load，
默认10个子进程
"""

## ! 创建logger的
logger = create_logger(filepath='./run.log')

openai.api_key = "你的key"
DEBUG = False

def get_response(prompt, temperature=0.1, max_tokens=2048):
    """调用函数，max_tokens是要生成的tokens的最大长度， 
        max_tokens+ num_prompt_tokens最多是4096 受调用的model限制"""
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.3,
    )
    return response

# 构造自己的模板，后面结果的接受识别与模板的构造直接相关
SENTENCE = "\n    %d.  %s"
PROMPT = f"Revise the sentences given below after translation, correct grammar and background errors, and return the corrected sentences "

def load_paras_from_torch(path):
    """加载数据"""
    assert os.path.exists(path)
    paras = torch.load(path)
    return paras

def get_prompt_input(pre):
    """返回填充后的prompt_input 和字符串长度"""
    prompt_input = PROMPT + "".join([SENTENCE % (i+1, s) for i, s in enumerate(pre)])
    return prompt_input, len(prompt_input)

def save_result(i_id_thread, res, paras, cor, data_path="./"):
    """
    保存结果
    这里传进来的paras需要时SrcRefPreCor的对象列表
    """
    save_path = os.path.join(data_path, f"thread{i_id_thread}")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_path = os.path.join(save_path, f"res-{i_id_thread}.txt")
    cor_path = os.path.join(save_path, f"cor-{i_id_thread}.txt")
    paras_path = os.path.join(save_path, f"paras-{i_id_thread}.bin")
    res = [r+"\n" for r in res]
    cor = [r+"\n" for r in cor]
    # paras = [p.add_cor(c) for p, c in zip(paras, cor)]    #这里在此前就已经加入了
    with open(res_path, "w") as f:
        f.writelines(res)
    with open(cor_path, "w") as f:
        f.writelines(cor)
    
    torch.save(paras, paras_path)
    logger.critical("结果保存完成")

def sleep(i_id_thread):
    """
    处理拥挤的函数，那个进程遇到拥挤了，就让它睡会
    免费api的限制是 每秒最多60次请求， 每分钟最多150000 个tokens
    """
    t = 1 * (i_id_thread * 2  + 5)
    time.sleep(t)

def thread_fun(i_id_thread, span="\n\d.", num_sent_per_api=8, res=None, paras=None, cor=None):
    """
    子进程里运行的函数
    span 使用正则表达式检测所需答案所在句子位置，截取答案保存
    res 记录每次调用API的返回结果, paras 要调用的句子对， cor记录每条句子对应的返回结果
    """
    res = [] if res==None else res
    cor = [] if cor==None else cor
    num_error = 0

    total_num_api = len(paras) // num_sent_per_api
    total_num_api = total_num_api + 1 if (len(paras) % num_sent_per_api)!=0 else total_num_api
    for i in range(total_num_api):
        send_paras = paras[i*num_sent_per_api : (i+1)* num_sent_per_api]
        prompt_input, length = get_prompt_input([p.pre for p in send_paras])    #取出pre打包
        while (True):           # 应对查询速率限制
            try:
                res_context = get_response(prompt_input)
                break
            except Exception as e:
                if isinstance(e, openai.error.RateLimitError):
                    sleep(i_id_thread)      #处理拥挤的函数，就只是睡着而已
                    continue
                else:
                    break
        logger.info(f"子进程{i_id_thread}正在第{i}次调用api 已获取结果")
        
        ## ! 处理接收到的结果 可自定义
        res.append(str(res_context))                                                 # 结果记录
        res_context = res_context.choices[0].text

        result = re.finditer(span, res_context)
        locs = [(match.start(), match.end()) for match in result]
        locs_start = [e for s, e in locs]
        locs_end = [s for s, e in locs]
        locs_end = locs_end[1:] + [len(res_context)+1]

        # 取出每个纠正句子，根据开始和结束位置, i+1 是为了去除句子前面的空格
        if len(locs) > 0:                           # 如果没有定位到，那就用整条句子作为结果
            res_context = [res_context[i+1:j] for i, j in zip(locs_start, locs_end)]
        else:
            res_context = [res_context.lstrip("\n").rstrip("\n")]

        if len(res_context) < len(send_paras):      #生成内容补全
            num_error += len(send_paras)-len(res_context)
            res_context += [res_context[-1] if len(res_context)>0 else '' for _ in range(len(send_paras)-len(res_context))]
        if len(res_context) > len(send_paras):      #生成内容过多
            num_error += len(res_context)-len(send_paras)
            res_context = res_context[:len(send_paras)]
        cor += res_context                          #这样最后一组哪怕不是num_sent_per_api个也照样可以用

    paras = [p.add_cor(cor) for p, cor in zip(paras, cor)]                 # 组装成4元组
    res = [f"num_error_sentence: {num_error}"] + res
    save_result(i_id_thread, res, paras, cor)       ## ! 保存结果


if __name__ == '__main__':
    # 数据加载和保存位置
    data_path = "./"
    src_ref_pre_path = os.path.join(data_path, "src_ref_pre.bin")
    logger.info(src_ref_pre_path)
    
    src_ref_pre = load_paras_from_torch(src_ref_pre_path)
    logger.info("load src_ref_pre three tuple over")

    if not DEBUG:
        # 过滤哪些句子要调用的，不用过滤可以直接删掉 留下else后的那句话
        if src_ref_pre[0].ref != None:     #也就是说这是单语句子的生成不需要过滤:
            src_ref_pre_filt = filte_by_bleu(src_ref_pre, compute_batch, LOW_BLEU, HIGH_BLEU)
        else:
            src_ref_pre_filt = src_ref_pre
    else:
        src_ref_pre_filt = src_ref_pre

    ## ! 使用类封装一下，这是我自己的数据封装的方法，更换
    if not isinstance(src_ref_pre_filt[0], SrcRefPreCor):
        src_ref_pre_filt = [SrcRefPreCor(s, r, p) for s, r, p in src_ref_pre_filt]
    ## ! debug
    if DEBUG:
        src_ref_pre_filt = src_ref_pre_filt[:7]

    ## ! 设置num_thread
    num_thread = 10 if not DEBUG else 3
    # 每次调用api 加入prompt里最多处理的句子，我一次prompt处理9条数据
    num_sent_per_api = 9 if not DEBUG else 2
    num_sent_pre_thread = len(src_ref_pre_filt) // num_thread

    # 将数据列表拆分，给每个子线程一个数据列表， 最后一个进程要把剩下的全部包括进去
    thread_src_ref_pre = [src_ref_pre_filt[i*num_sent_pre_thread : (i+1)*num_sent_pre_thread] \
                      if i!=num_thread-1 else src_ref_pre_filt[i*num_sent_pre_thread:]   \
                      for i in range(0, num_thread) ]
    
    logger.info(f"every thread process num of sentence {num_sent_pre_thread}, the last thread process num {len(thread_src_ref_pre[-1])}")
    logger.info(f"每次调用api处理的句子数{num_sent_per_api}")

    pool = multiprocessing.Pool(processes=num_thread)
    processes = []
    for i in range(num_thread):
        # thread_fun 子进程内运行的函数， 后面的()里的是传给此函数的参数，子进程的结果保存在thread{i}里
        p = pool.apply_async(thread_fun, (i, "\n\d.", num_sent_per_api,
                                          None, thread_src_ref_pre[i], None))
        processes.append(p)
    pool.close()
    pool.join()
    for p in processes:
        p.get()