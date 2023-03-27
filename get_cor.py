import os
import openai
import evaluate
import json
import re
import multiprocessing
import torch
import time


from utils import create_logger, paras_filter_by_belu, SrcRefPreCor

# nohup ./get_cor.py > get_cor.log 2>&1 &

# class SrcRefPreCor(object):
#     """用来保存成对的src pre ref cor 的内容"""
#     def __init__(self, src=None, ref=None, pre=None, cor=None) -> None:
#         self.src = src if src else None
#         self.ref = ref if ref else None
#         self.pre = pre if pre else None
#         self.cor = cor if cor else None
#         pass
#     def add_ref(self, ref):
#         assert self.ref, "ref 不为空"
#         self.ref = ref
#     def add_pre(self, pre):
#         assert self.pre, "pre 不为空"
#         self.pre = pre
#     def add_cor(self, cor):
#         assert self.cor, "cor 不为空"
#         self.cor = cor
    
#     def __getitem__(self, i):
#         if i==0:
#             return self.src
#         elif i==1:
#             assert self.ref, f"i={i}, 取ref，但是ref为空"
#             return self.ref
#         elif i==2:
#             assert self.pre, f"i={i}, 取pre，但是pre为空"
#             return self.pre
#         elif i==3:
#             assert self.cor, f"i={i}, 取cor，但是cor为空"
#             return self.cor
#         else:
#             assert -1<i<4, f"i的取值{i}, 无效"
#     def __str__(self) -> str:
#         return json.dumps(self.__dict__, ensure_ascii=False)
#     def __repr__(self) -> str:
#         return self.__str__()

# openai.api_key = os.getenv("OPENAI_API_KEY")
def get_response(prompt, temperature=0.1, max_tokens=2048):
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


openai.api_key = ""

# 模板
SENTENCE = "\n    %d.  %s"
PROMPT = f"Revise the sentences given below after translation, correct grammar and background errors, and return the corrected sentences "

def get_prompt_input(pre):
    """返回填充后的prompt_input 和字符串长度"""
    prompt_input = PROMPT + "".join([SENTENCE % (i+1, s) for i, s in enumerate(pre)])
    return prompt_input, len(prompt_input)

## filter paras by bleu
LOW_BLEU = 3
HIGH_BLEU = 24

DEBUG = False

logger = create_logger(filepath='./log/get_cor.log')

metric = evaluate.load('sacrebleu')
def compute_metric(ref, pre):
    bleu = metric.compute(predictions=[pre], references=[[ref]])
    return bleu
def compute_batch(ref, pre):
    """给两个列表，返回二者的bleu列表"""
    bleu = []
    for r, p in zip(ref, pre):
        bleu.append(compute_metric(r, p)['score'])
    return bleu
def load_paras_from_torch(path):
    assert os.path.exists(path)
    paras = torch.load(path)
    return paras
def filte_by_bleu(src_ref_pre, compute_batch, low_bleu, high_bleu):
    bleu = compute_batch(ref=[x[1] for x in src_ref_pre], pre=[x[2] for x in src_ref_pre])
    logger.info(f"平均bule值为{sum(bleu) / len(bleu)}")
    logger.info(f"经过bleu过滤之前{len(src_ref_pre)}")
    src_ref_pre_filt = paras_filter_by_belu(src_ref_pre, bleu, patience=low_bleu, high=high_bleu)
    logger.info(f"经过过滤之后的{len(src_ref_pre_filt)}")
    return src_ref_pre_filt

data_path = "checkpoint/mbart50/ur-en-23K/"

def save_result(i_id_thread, res, paras, cor):
    """这里传进来的paras需要时SrcRefPreCor的对象列表"""
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


def thread_fun(i_id_thread, span="\n\d.", num_sent_per_api=8, res=None, paras=None, cor=None):
    """res 记录每次调用API的返回结果, paras 要调用的句子对， cor记录每条句子对应的返回结果"""
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
                    t = 1 * (i_id_thread * 1.5  + 10)
                    time.sleep(t)
                    continue
                else:
                    break
        if i_id_thread==1:
            pass
        logger.info(f"子进程{i_id_thread}正在第{i}次调用api 已获取结果")

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
    save_result(i_id_thread, res, paras, cor)


if __name__ == '__main__':
    src_ref_pre_path = os.path.join(data_path, "src_ref_pre.bin")
    # src_ref_pre_cor_path = os.path.join(data_path, "src_ref_pre_cor.bin")
    logger.info(src_ref_pre_path)
    
    src_ref_pre = load_paras_from_torch(src_ref_pre_path)
    logger.info("load src_ref_pre three tuple over")

    if not DEBUG:
        if src_ref_pre[0].ref != None:     #也就是说这是单语句子的生成不需要过滤:
            src_ref_pre_filt = filte_by_bleu(src_ref_pre, compute_batch, LOW_BLEU, HIGH_BLEU)
        else:
            src_ref_pre_filt = src_ref_pre
    else:
        src_ref_pre_filt = src_ref_pre

    ## ! 使用类封装以下
    if not isinstance(src_ref_pre_filt[0], SrcRefPreCor):
        src_ref_pre_filt = [SrcRefPreCor(s, r, p) for s, r, p in src_ref_pre_filt]
    ## ! debug
    if DEBUG:
        src_ref_pre_filt = src_ref_pre_filt[:7]
    else:
        src_ref_pre_filt = src_ref_pre_filt[8000:]

    ## 设置num_thread
    num_thread = 10 if not DEBUG else 3
    num_sent_per_api = 9 if not DEBUG else 2        # 没次调用api最多使用的句子
    num_sent_pre_thread = len(src_ref_pre_filt) // num_thread
    # 将句子列表拆分，给每个子线程一个句子列表， 最后一个进程要把剩下的全部包括进去
    thread_src_ref_pre = [src_ref_pre_filt[i*num_sent_pre_thread : (i+1)*num_sent_pre_thread] \
                      if i!=num_thread-1 else src_ref_pre_filt[i*num_sent_pre_thread:]   \
                      for i in range(0, num_thread) ]
    logger.info(f"every thread process num of sentence {num_sent_pre_thread}, the last thread process num {len(thread_src_ref_pre[-1])}")
    logger.info(f"每次调用api处理的句子数{num_sent_per_api}")

    pool = multiprocessing.Pool(processes=num_thread)
    processes = []
    for i in range(num_thread):
        # p = multiprocessing.Process(target=thread_fun(
        #     i_id_thread=i,
        #     span="\n\d+.",
        #     num_sent_per_api=num_sent_per_api,
        #     paras=thread_src_ref_pre[i],
        # ))
        p = pool.apply_async(thread_fun, (i, "\n\d.", num_sent_per_api,
                                          None, thread_src_ref_pre[i], None))
        processes.append(p)
    pool.close()
    pool.join()
    for p in processes:
        p.get()