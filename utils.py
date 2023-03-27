from datetime import timedelta
import math
import random
import time
from functools import partial
from datasets import Dataset, DatasetDict
import numpy as np
import torch
import logging
import os
import evaluate
import json

from transformers import (Seq2SeqTrainingArguments,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          AutoModel
                          )

logger = logging.getLogger(__name__)


PRETRAINED_MODEL = (
                    # "microsoft/xlm-align-base",
                    # "facebook/mbart-large-cc25",
                    "facebook/mbart-large-50", 
                    "facebook/mbart-large-50-many-to-many-mmt")
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

def try_gpu(i=0):
    if i <= torch.cuda.device_count():
        return torch.device(f"cuda:{i}")
    return torch.device('cpu')

def to_cuda(**params):

    return [p.cuda() for p in params]

def setup_seed(seed):
    torch.manual_seed(seed)                                 #不是返回一个生成器吗？
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True               #使用确定性的卷积算法

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
    return [pa for pa, b in zip(correct_paras, bleu) if b>patience and b<high]


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

"""下面的三个加噪音函数copy from => fairseq.data.denoising_dataset.py DenoisingDataset"""
def add_permuted_noise(tokens, p):
    num_words = len(tokens)
    num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
    substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
    tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
    return tokens

def add_rolling_noise(tokens):
    offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
    tokens = torch.cat(
        (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
        dim=0,
    )
    return tokens

# def add_insertion_noise(self, tokens, p):
#     if p == 0.0:
#         return tokens

#     num_tokens = len(tokens)
#     n = int(math.ceil(num_tokens * p))

#     noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
#     noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
#     noise_mask[noise_indices] = 1
#     result = torch.LongTensor(n + len(tokens)).fill_(-1)

#     num_random = int(math.ceil(n * self.random_ratio))
#     result[noise_indices[num_random:]] = self.mask_idx
#     result[noise_indices[:num_random]] = torch.randint(low=1, high=len(self.vocab), size=(num_random,))

#     result[~noise_mask] = tokens

#     assert (result >= 0).all()
#     return result


