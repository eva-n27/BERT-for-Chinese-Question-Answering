## BERT-for-Chinese-Question-Answering
本仓库的代码来源于[PyTorch Pretrained Bert](https://github.com/huggingface/pytorch-pretrained-BERT)，仅做适配中文的QA任务的修改

主要修改的地方为read_squad_examples函数，由于SQuAD是英文的，因此源代码处理的方式是按照英文的方式，即[此处](https://github.com/huggingface/pytorch-pretrained-BERT/blob/04826b0f2cdaec92db859e6dc07f31e3b3381d0d/examples/run_squad.py#L122)。

另外，增加了训练中每隔save_checkpoints_steps次进行evaluate，并保存dev上效果最好的模型参数。

因此修改为：

1.先使用tokenizer先使用tokenizer.basic_tokenizer.tokenize对doc进行处理得到doc_tokens（代码161行）

2.对orig_answer_text使用tokenizer.basic_tokenizer.tokenize，然后再计算answer的start_position和end_position（代码172-191）

### 使用方法

*   首先需要将你的语料转换成SQuAD形式的，将数据以及模型文件放到data目录下（需要自己创建）

*   执行
```
python3 run_squad.py \
  --do_train 
  --do_predict 
  --save_checkpoints_steps 3000 
  --train_batch_size 12 
  --num_train_epochs 5
```

*   测试
eval.py中增加了使用BERT的tokenization，然后再计算EM和F1
```
python3 eval.py data/squad_dev.json output/predictions.json
```

欢迎各位大佬批评和指正，感谢
