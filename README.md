# nmt-corrector
`get_cor.py`是我自己调用api的原始代码
`chatgpt_api.py` 是整理后的，如有错误以`get_cor.py`文件的为准

结果保存结构的目录结构， `ur-en-23k` 是数据目录

![image](https://user-images.githubusercontent.com/46342773/227868196-96fa696b-7d81-4939-b29e-823447601d9f.png)


# update

- 在`get_cor.py`里增加了每5次请求完成就保存一次的功能，如果出现异常，那么在退出前会保存
- 更改了结果的`re`匹配方式先用`\n`分割句子，然后用re匹配
- 在`utils`里增加新的prompt完成别的任务

增加了continue，如果在运行`get_cor.py`的时候数据没有请求完就因为不知名原因断了，可以运行continue接着继续

# update 23-05-23

主要流程 `get_cor.py`
- 将全部数据读入列表`src_ref_pre_filt`，自动根据进程数`num_thread`计算每个进程需处理得数据，并存入`thread_src_ref_pre`。
- 进程开启会调用`thread_fun`函数，并在元组传入参数
```python
p = pool.apply_async(thread_fun, (i, r'\d+\.', num_sent_per_api,
                                          None, thread_src_ref_pre[i], None))
```
- 进程函数内部使用`get_prompt_input`获取每次调用api时的prompt
- 取得prompt用`get_response`调用
- 用`get_result_from_response`解析返回结果

# update 23-05-24

- 将多进程调用框架提取出来到 `multiprocess_call.py` 文件里。
- 顺着if __name__=="__main__": 读，实现相应的功能即可
- 注意logger，默认的输出级别是warning。
- 保存每个进程结果的时候，每个进程是并行执行的，所以顺序会乱，最好每个每个进程先分别保存，最后整合。
