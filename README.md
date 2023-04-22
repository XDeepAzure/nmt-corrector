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
