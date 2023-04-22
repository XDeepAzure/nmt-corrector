from datetime import timedelta
import math
import random
import re
import time
from functools import partial
from datasets import Dataset, DatasetDict
import numpy as np
import torch
import logging
import os
import evaluate
import json
import openai
from tqdm import tqdm

from transformers import (Seq2SeqTrainingArguments,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          AutoModel
                          )

logger = logging.getLogger(__name__)

API_KEY=("",     # "hyxu"
         ) 

PRETRAINED_MODEL = (
                    # "microsoft/xlm-align-base",
                    # "facebook/mbart-large-cc25",
                    "facebook/mbart-large-50", 
                    "facebook/mbart-large-50-many-to-many-mmt")
TASK = ("correct", "translate")

SENTENCE = {TASK[0]:"\n    %d. %s",}
PROMPT = {TASK[0]:f"Revise the sentences given below after translation, correct grammar and background errors, and return the corrected sentences:",
          TASK[1]:f"Translate these sentences from South Azerbaijani to English:",
        }

def get_prompt_input(send_paras, task=TASK[0]):
    """返回填充后的prompt_input 和字符串长度"""
    assert task in PROMPT
    if task == TASK[0]:
        lst = [p.pre for p in send_paras]
    elif task == TASK[1]:
        lst = [p.src for p in send_paras]
    prompt_input = PROMPT[task] + "".join([SENTENCE[TASK[0]] % (i+1, s) for i, s in enumerate(lst)])
    return prompt_input, len(prompt_input)

def avg(x):
    return sum(x)/len(x)

class SrcRefPreCor(object):
    """用来保存成对的src pre ref cor 的内容"""
    def __init__(self, src=None, ref=None, pre=None, cor=None) -> None:
        self.src = src if src else None
        self.ref = ref if ref else None
        self.pre = pre if pre else None
        self.cor = cor if cor else None
        pass
    def add_ref(self, ref):
        assert self.ref==None, "ref 不为空"
        self.ref = ref
    def add_pre(self, pre):
        assert self.pre==None, "pre 不为空"
        self.pre = pre
    def add_cor(self, cor):
        assert self.cor==None, "cor 不为空"
        self.cor = cor
    
    def __getitem__(self, i):
        if i==0:
            return self.src
        elif i==1:
            assert self.ref, f"i={i}, 取ref，但是ref为空"
            return self.ref
        elif i==2:
            assert self.pre, f"i={i}, 取pre，但是pre为空"
            return self.pre
        elif i==3:
            assert self.cor, f"i={i}, 取cor，但是cor为空"
            return self.cor
        else:
            assert -1<i<4, f"i的取值{i}, 无效"
    def __str__(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)
    def __repr__(self) -> str:
        return self.__str__()


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

def avg(d):
    return sum(d) / len(d)

def filter_by_bleu(paras, r_p_bleu, r_c_bleu, return_bleu=False):
    """bleu过滤，条件: c_b>5 and c_b>p_b-5 """
    data = [(paras[i], p_b, c_b) for i, (p_b, c_b) in enumerate(zip(r_p_bleu, r_c_bleu)) if c_b>5 and c_b>p_b-5]
    if return_bleu:
        return data
    else:
        return [p for p, p_b, c_b in data]
def trans_filter_by_len(paras, r_p_bleu=None, l_len=7, h_len=50, per_len=0.3, return_bleu=False):
    """len 过滤条件，
        对于src_ref_pre 来说 ref和pre 词的个数大于7 且小于50 且句子的词数差别小于30%
        对于src_pre 来说 src和pre 词的个数大于8且句子的词数差别小于40% 这就要换一个函数了
    """
    assert r_p_bleu == None and return_bleu == False
    def fn(src, tgt):
        l1 = len(src.split(" "))
        l2 = len(tgt.split(" "))
        if h_len > l1 > l_len and h_len > l2 > l_len:
            if (abs(l1-l2)/l1) <= per_len:
                return True
        return False
    if r_p_bleu != None and return_bleu:
        data = [(paras[i], b) for i, b in enumerate(r_p_bleu) if fn(paras[i].ref, paras[i].pre)]
    else:
        data = [p for p in paras if fn(p.src, p.pre)]
    if return_bleu:
        return data
    else:
        return data

def clear_fn(paras, src_ref_pre_cor, is_filter=True, batch_size=20):

    ref = [p.ref for p in paras if p.cor != None]
    pre = [p.pre for p in paras if p.cor != None]
    cor = [p.cor.lstrip(" ").lstrip('\n').rstrip("\n").rstrip(" ").replace("\n", "") for p in paras if p.cor!=None]
    num_batch = len(cor)//batch_size if len(cor)%batch_size==0 else len(cor)//batch_size +1
    r_p_bleu, r_c_bleu = [], []
    if is_filter:
        for i in tqdm(range(num_batch+1)):
            r_p_bleu += compute_batch(ref=ref[i*batch_size:(i+1)*batch_size],
                                      pre=pre[i*batch_size:(i+1)*batch_size])
        for i in tqdm(range(num_batch+1)):
            r_c_bleu += compute_batch(ref=ref[i*batch_size:(i+1)*batch_size],
                                      pre=cor[i*batch_size:(i+1)*batch_size])
    paras = [p for p in paras if p.cor!=None]
    for p, c in zip(paras, cor):                    #将干净的cor赋值给p
        p.cor = c
    src_ref_pre_cor += filter_fn(paras, r_p_bleu, r_c_bleu) if is_filter else paras
    return r_p_bleu, r_c_bleu

def process_data_from_dir(data_dir, is_filter=True):
    """处理用chatgpt调用得到的数据data_dir 进程结果保存位置，clear_fn 清理函数"""
    file_prefix = 'thread'
    paras_prfix = 'paras-'
    total_r_p_belu, total_r_c_bleu = [], [] 
    total_paras = []
    src_ref_pre_cor = []
    dir_list = os.listdir(data_dir)
    for i, p in enumerate(dir_list):
        if not p.startswith(file_prefix): continue
    
        id_thread = p[len(file_prefix):]
        print(f"正在处理第{id_thread}个进程结果")
        paras_data = torch.load(os.path.join(data_dir, p, f"{paras_prfix}{id_thread}.bin"))
        total_paras += paras_data
    
        r_p_bleu, r_c_bleu = clear_fn(paras_data, src_ref_pre_cor, is_filter)
        if is_filter:
            print(f"pre平均bleu:{avg(r_p_bleu)},  cor平均bleu:{avg(r_c_bleu)}")
            total_r_p_belu += r_p_bleu
            total_r_c_bleu += r_c_bleu
    return src_ref_pre_cor, total_paras, (total_r_p_belu, total_r_c_bleu)

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

def get_response(prompt, temperature=0.1, max_tokens=2048):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=temperature,
    top_p=1,
    max_tokens=max_tokens,
    messages=[
      {"role": "user", "content": f"{prompt}"}
    ]
  )
  return completion

def save_result(i_id_thread, data_path, task, res, paras, cor, prompts):
    assert os.path.exists(data_path)
    assert paras != None and res != None and prompts != None
    if task == TASK[1]:
        save_result_translate(i_id_thread,
                              data_path,
                              res,
                              paras,
                              prompts)
    elif task == TASK[0]:
        assert cor != None
        save_result_correct(i_id_thread,
                            data_path,
                            res,
                            paras,
                            cor,
                            prompts)
    else:
        assert task in TASK
def save_result_translate(i_id_thread, data_path, res, paras, prompts):
    """这里传进来的paras需要时SrcRefPreCor的对象列表"""

    save_path = os.path.join(data_path, f"thread{i_id_thread}")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_path = os.path.join(save_path, f"res-{i_id_thread}.txt")
    prompt_path = os.path.join(save_path, f"prompts-{i_id_thread}.txt")
    paras_path = os.path.join(save_path, f"paras-{i_id_thread}.bin")
    res = [r+"\n" for r in res]
    prompts = [r+"\n" for r in prompts]
    
    with open(res_path, "w", encoding="utf-8") as f:
        f.writelines(res)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.writelines(prompts)
    
    torch.save(paras, paras_path)
    logger.critical(f" 线程{i_id_thread} 结果保存完成")

def save_result_correct(i_id_thread, data_path, res, paras, cor, prompts):
    """这里传进来的paras需要时SrcRefPreCor的对象列表"""

    save_path = os.path.join(data_path, f"thread{i_id_thread}")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_path = os.path.join(save_path, f"res-{i_id_thread}.txt")
    cor_path = os.path.join(save_path, f"cor-{i_id_thread}.txt")
    prompt_path = os.path.join(save_path, f"prompts-{i_id_thread}.txt")
    paras_path = os.path.join(save_path, f"paras-{i_id_thread}.bin")
    res = [r+"\n" for r in res]
    cor = [r+"\n" for r in cor]
    prompts = [r+"\n" for r in prompts]
    
    with open(res_path, "w", encoding="utf-8") as f:
        f.writelines(res)
    with open(cor_path, "w", encoding="utf-8") as f:
        f.writelines(cor)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.writelines(prompts)
    
    torch.save(paras, paras_path)
    logger.critical(f" 线程{i_id_thread} 结果保存完成")

def get_result_from_response(res_context, send_paras, span):
    num_error = 0
    res_context = res_context.choices[0].message.content

    sentences = [s.lstrip(" ").rstrip(" ") for s in res_context.split("\n") if len(s)>1]

    pattern = r'\d+\.'
    res_match = []
    res_all = []                                        # 记录全部的句子
    not_match_idx = []                                  # 记录没匹配到的句子的idx
    for i, s in enumerate(sentences):
        s = s.lstrip(" ").rstrip(" ")
        match = re.match(pattern, s)
        if match:
            s = s[match.end():]
            res_match.append(s)
        else:
            not_match_idx.append(i)
        res_all.append(s)

    if len(res_match) < len(send_paras):                #生成内容补全
        num_error += len(send_paras)-len(res_match)
        res_match += ["\n".join(sentences) for _ in range(len(send_paras)-len(res_match))]
    if len(res_match) > len(send_paras):                #生成内容过多
        num_error += len(res_match)-len(send_paras)
        res_match = res_match[:len(send_paras)]         #这样最后一组哪怕不是num_sent_per_api个也照样可以用
    return res_match, num_error 
    
# def get_result_from_response(res_context, send_paras, span):
    #   """考虑使用\n作为分割句子"""
#     num_error = 0
#     res_context = res_context.choices[0].message.content

#     result = re.finditer(span, res_context)
#     locs = [(match.start(), match.end()) for match in result]
#     locs_start = [e for s, e in locs]
#     locs_end = [s for s, e in locs]
#     locs_end = locs_end[1:] + [len(res_context)+1]

#     # 取出每个纠正句子，根据开始和结束位置, i+1 是为了去除句子前面的空格
#     if len(locs) > 0:                                   # 如果没有定位到，那就用整条句子作为结果
#         res_context = [res_context[i+1:j].replace("\n", "").lstrip(" ").rstrip(" ")
#                             for i, j in zip(locs_start, locs_end)]
#     else:
#         res_context = [res_context.lstrip("\n").rstrip("\n").lstrip(" ").rstrip(" ")]

#     if len(res_context) < len(send_paras):              #生成内容补全
#         num_error += len(send_paras)-len(res_context)
#         res_context += [res_context[-1] if len(res_context)>0 else '' for _ in range(len(send_paras)-len(res_context))]
#     if len(res_context) > len(send_paras):              #生成内容过多
#         num_error += len(res_context)-len(send_paras)
#         res_context = res_context[:len(send_paras)]     #这样最后一组哪怕不是num_sent_per_api个也照样可以用
#     return res_context, num_error     


def preprocess_function(examples, src_lang, tgt_lang, tokenizer, max_input_length, max_target_length):
    inputs = [ex for ex in examples[src_lang]]
    targets = [ex for ex in examples[tgt_lang]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets 源语言与目标语言使用联合词典的
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    # model_inputs["labels_attention_mask"] = labels["attention_mask"]
    return model_inputs

def get_tokenized_datasets(tokenizer, trans_para, src_lang, tgt_lang, max_input_length, max_target_length):
    """只进行tokenized不做split"""
    batch_tokenize_fn = partial(preprocess_function,
                                tokenizer=tokenizer,
                                src_lang=src_lang,
                                tgt_lang=tgt_lang,
                                max_input_length=max_input_length,
                                max_target_length=max_target_length,
                                )

    trans_para = {
        src_lang: [src for src, _ in trans_para],
        tgt_lang: [tgt for _, tgt in trans_para]
    }
    raw_datasets = Dataset.from_dict(trans_para)
    raw_datasets = DatasetDict({'train': raw_datasets})

    tokenized_datasets = raw_datasets.map(batch_tokenize_fn, batched=True,
                                          remove_columns=raw_datasets['train'].column_names)
    return tokenized_datasets
    
def get_translate_paras_from_file(src_f, tgt_f):    
    src_data, tgt_data = [], []
    with open(src_f, 'r') as src_f, open(tgt_f, 'r') as tgt_f:
        src_data = src_f.readlines()
        tgt_data = tgt_f.readlines()
    src_data = [src.rstrip('\n') for src in src_data]
    tgt_data = [tgt.rstrip('\n') for tgt in tgt_data]
    trans_para = [[src, tgt] for src, tgt in zip(src_data, tgt_data) if len(src) > 0 and len(tgt) > 0]
    return trans_para

def split_datasets(dataset, test=3000, valid=0, seed=10):
    """如果valid是0 那么就之分train 和 test 不分 valid"""
    if isinstance(dataset, Dataset):
        split_dataset_dict = dataset.train_test_split(test_size=test, seed=seed)
    elif isinstance(dataset, DatasetDict):
        split_dataset_dict = dataset['train'].train_test_split(test_size=test, seed=seed)
    if valid != 0:
        valid_dataset = split_dataset_dict.pop("test")
        split_dataset_dict = split_dataset_dict['train'].train_test_split(test_size=valid, seed=seed)
        split_dataset_dict['valid'] = valid_dataset
    return split_dataset_dict

def paras_filter_by_belu(correct_paras, bleu, patience=-1, high=101):
    """大于patience小于high 不包含等于"""
    return [(pa,b) for pa, b in zip(correct_paras, bleu) if b>patience and b<high]


def load_tokenizer(args):
    """当args中需要有args.src_lang_code与args.tgt_lang_code"""
    assert hasattr(args, "src_lang_code") and hasattr(args, "tgt_lang_code")
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint != '':
        path = os.path.join(args.resume_from_checkpoint)
    else:
        logger.info(args.model_name)
        logger.info(PRETRAINED_MODEL)
        assert args.model_name in PRETRAINED_MODEL, "model don't load"
        path = os.path.join(args.pretrained_model, args.model_name.split('/')[-1])
    logger.critical(path)

    tokenizer = AutoTokenizer.from_pretrained(path, src_lang=args.src_lang_code, tgt_lang=args.tgt_lang_code)

    tokenizer.src_lang = args.src_lang_code
    tokenizer.tgt_lang = args.tgt_lang_code
    logger.info(f"load tokenizer form {path}")
    logger.info(tokenizer)
    return tokenizer

def initialize_exp(args):
    if not hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint == "":
        os.mkdir(args.saved_dir)
    with open(os.path.join(args.saved_dir, 'train.log'), 'w') as f:
        f.write("")
    name = "train_resume.log" if args.resume_from_checkpoint != "" else 'train.log'
    logger = create_logger(os.path.join(args.saved_dir, name), rank=getattr(args, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(args)).items())))
    logger.info("The experiment will be stored in %s\n" % args.saved_dir)
    logger.info("")
    return logger
    pass

def get_data_collator(args, tokenizer):
    """可以在这里自定datacollator"""
    if hasattr(args, "data_collator") and args.data_collator != "":
        return torch.load(args.data_collator)
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, 
                                        max_length=args.max_sentence_length)
    return data_collator

def get_model(args, config=None):
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint != "":
        path = os.path.join(args.resume_from_checkpoint)
    else:
        path = os.path.join(args.pretrained_model, args.model_name.split('/')[-1])
    logger.info(path)
    model_type = AutoModel
    if args.model_name == PRETRAINED_MODEL[0]:
        from transformers import MBartForConditionalGeneration
        model_type = MBartForConditionalGeneration
    # model = model_type(config)
    # model.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin')))
    model = model_type.from_pretrained(path)
    logger.critical("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))
    logger.info(model)
    return model

def get_training_args(args):
    
    logger.info(f"实验中数据存在此中：{args.saved_dir}")
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint != "":
        training_args = torch.load(os.path.join(args.resume_from_checkpoint, "training_args.bin"))
    else:

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.saved_dir,
            evaluation_strategy="steps",
            learning_rate=args.lr if hasattr(args, "lr") else 2e-5,
            per_device_eval_batch_size=args.batch_size,
            per_device_train_batch_size=args.batch_size,
            weight_decay=0.01,
            save_total_limit=4,
            num_train_epochs=50,
            generation_max_length=args.max_generate_length,
            max_length=args.max_length if hasattr(args, "max_length") else 256,
            num_beams=args.num_beams if hasattr(args, "num_beams") else 1,
            seed=args.seed,
            predict_with_generate=True,
            fp16=args.fp16 if hasattr(args, "fp16") else True,
            load_best_model_at_end=True,
            eval_steps=args.eval_steps if hasattr(args, "eval_steps") else 5000,
            save_steps=args.save_steps if hasattr(args, "save_steps") else 5000,
            warmup_steps=args.warmup_steps if hasattr(args, "warmup_steps") else 100,
            metric_for_best_model="bleu",
            report_to=['tensorboard']
        )
    logger.info(training_args)
    return training_args

metric = evaluate.load('sacrebleu')
def compute_metric(ref, pre):
    bleu = metric.compute(predictions=[pre], references=[[ref]], tokenize="flores200")
    # bleu = metric.compute(predictions=[pre], references=[[ref]])
    return bleu
def compute_batch(ref, pre):
    """给两个列表，返回二者的bleu列表"""
    bleu = []
    for r, p in tqdm(list(zip(ref, pre))):
        bleu.append(compute_metric(r, p)['score'])
    return bleu

def get_compute_metrics(args, tokenizer):
    if 'bleu' in args.evaluate_metrics:
        metric = evaluate.load('sacrebleu')
    def compute_metrics(eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_lables = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_lablels = [[label.strip()] for label in decoded_lables]

        result = metric.compute(predictions=decoded_preds, references=decoded_lablels)
        return {'bleu': result['score']}
    return compute_metrics