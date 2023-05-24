import time
from tqdm import tqdm
import os
import openai
from logging import getLogger

import multiprocessing
from translation_correct_rewrite import get_data, save_result
import translation_correct_rewrite.main as get_response

DEBUG = True
logger = getLogger("")
API_KEYS = ("",)

def get_prompt(x):
    """给定数据x组装prompt并返回"""
    return f"prompt: {x}"

# def get_response(prompt):
#     """给prompt调用api，并返回结果，模型及tempture等超参自定"""
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         temperature=0.1,
#         top_p=1,
#         max_tokens=4096,    # 最大token个数根据model定
#         messages=[
#           {"role": "user", "content": f"{prompt}"}
#         ]
#       )
#     return completion

def get_result_from_response(res_context):
    """分析并处理调用api的返回结果，不同model不同参数，返回结构可能不一样，最好自己在notebook里看看"""
    res_context = res_context.choices[0].message.content
    return res_context

def sleep(id_thread):
    t = min(45,5 + id_thread *3)
    logger.info(f"子进程{id_thread}开始睡眠了， 睡{t}秒")
    time.sleep(t)

# def save_result(results):
#     """保存结果，需要注意，每5次调用会保存一次，不要把之前的覆盖了"""
#     pass

def thread_fun(id_thread, start, data, save_path):
    """
    start: 这一批data在整个数据集中的起始位置
    每个进程所执行的程序，每个进程的处理数据过程写在此处
    x, y 进程函数参数
    """
    results = []
    save_path = os.path.join(save_path, f"thread{id_thread}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, x in tqdm(enumerate(data)):
        while(True):                                           # 主要应对速度限制，这里除非报没钱了的错误，否则一直循环
            try:
                # response = get_response(prompt)
                response = get_response(id_thread, start+i, x, API_KEYS[0])
                break
            except Exception as e:
                if isinstance(e, openai.error.RateLimitError) and \
                    'You exceeded your current quota, please check your plan and billing details.' in e.user_message:
                    logger.critical("你没钱了")                 # logger
                    break
                else:                                           #不知名的错误
                    logger.error(f"{id_thread} error message {e.user_message}")
                    sleep(id_thread)
                    continue
        if i % 3 == 0:          
            logger.info(f"子进程{id_thread}正在第{i}次调用api 已获取结果")
        save_result(response, save_path)
        results.append(response)
    #     if i % 5 == 0:
    #         save_result(results)
    # save_result(results)
    return None

if __name__=="__mian__":
    num_thread = 8 if not DEBUG else 2
    source_lang, target_lang = "French", "German"

    with open("language2code.txt") as f:
        language2code = eval(f.read())
    save_path = f'./data/{language2code[source_lang]}2{language2code[target_lang]}_result/'

    thread_src_ref_pre = []                         # 每个元素对应着thread要处理的那一批数据（按下标对应）
    save_path = ""                                  # save_path

    pool = multiprocessing.Pool(processes=num_thread)
    processes = []
    for i in range(num_thread):
        p = pool.apply_async(thread_fun, (i, thread_src_ref_pre[i], save_path))
        processes.append(p)
    pool.close()
    pool.join()
    for p in processes:
        p.get()