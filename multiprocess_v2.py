from datetime import timedelta
import os
import json
import time
from typing import Dict
import random
from tqdm import tqdm
import multiprocessing
import openai
import logging


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath=None, rank=0, name=None):
    """
    Create a logger.
    Use a different log file for each process.
    filepath 为None的时候即不输出到文本里面去，
    rank为0的时候即单线程
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    if name != None:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

logger = create_logger("./run.log")


openai.api_key = ""

# proxy = {
# 'http': 'http://localhost:4780',
# 'https': 'http://localhost:4780'
# }

# openai.proxy = proxy

def get_messages(data: Dict):

    message = f"this is your messages : {str(data)}"
    logger.info(message)                                           # 打印调用prompt

    messages = {'messages': [{"role": "user", "content": message}, {"role": "system", "content": ""}]}
    return messages

def get_response(messages, temperature=0):
  response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        messages=messages,
        temperature=temperature
    )
  return response['choices'][0]['message']['content']

def while_try_response(get_response_func, input):
    """在调用函数外面包一个循环，报错了就打印出来，然后继续调用，直到成功"""
    # input写成一个字典，这样在里面直接解析就可以
    while True:
        try:
            result = get_response_func(**input)
            break
        except Exception as e:
            logger.critical(e)
            time.sleep(10 + random.random() * 15)                   # 出错了就等一会再继续调用
    return result

def thread_fun(file_name, file_dir,  output_file, result_name):
    logger.info(f"🖐️ 调取数据 {file_name} ......")
    with open(os.path.join(file_dir, file_name)) as f:
        data = json.loads(f.read())                                 # 读出数据
        
    messages = get_messages(data)                                   # 获取消息

    logger.info(messages)                                           # 打印调用prompt
    result = while_try_response(get_response, messages)
    
    data[result_name] = result

    with open(f'{output_file}/{file_name}', 'w') as f:              # 存入结果
        json.dump(data, f, indent=4)
    logger.info(f"🔥 {file_name} was done")


if __name__ == '__main__':
    
    openai.api_key = ""
    
    num_thread = 10
    file_dir = f'./data'                                            # 调用的数据位置    
    output_file = file_dir + '_result'                              # 结果写入位置
    result_name = "res"                                             # 结果字段名字
    
    os.makedirs(output_file, exist_ok=True)

    logger.info('start')
    pool = multiprocessing.Pool(processes=num_thread)
    process = []

    num_slice = -1                                                  # 选择调几条， 用于初步测试
    call_file_list = list(os.listdir(file_dir))
    # random.shuffle(call_file_list)                                # 是否随机调
    call_file_list = call_file_list[:num_slice] if num_slice > -1  else call_file_list

    logger.info(f"❗️❗️❗️ 调用数据个数 {len(call_file_list)}")
    for file_name in call_file_list:
        if file_name not in os.listdir(output_file):                # 判断是否已经调用过的逻辑
            logger.info(file_name)
            p = pool.apply_async(thread_fun, (file_name, file_dir, output_file, result_name))
            process.append(p)
            logger.info(f"❗️ 还剩下 {len(call_file_list) - len(os.listdir(output_file))} 条数据")
            # thread_fun(file_name, length, file_dir, output_file)
        # break
    pool.close()
    pool.join()
    
    for p in process:
        p.get()
        
    logger.info('done!!')    
        
    
        
    
        