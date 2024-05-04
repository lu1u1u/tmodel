## hotel 下载地址

https://www.cs.cmu.edu/~jiweil/html/hotel-review.html

解压后取`review.csv`

## 微调脚本 hotel_newmodel.sh

需要更改的有：
```
ztokens=${1} # 16 32 64
alpha=${2} # default=1
beta=${3} # default=1

(32行) trainf=xxx # review.csv
(54行) --num_train_epochs 1 
```

## 测试生成脚本 hotel_gen.sh

需要更改的有：

```

largepath=xxx #计算ppl使用的 gpt2-large 路径

gen_bs=128 # 可以调高些
rp=1.2
topk=50
topp=1.0
beams=4
temperature=1.0  # 生成参数

ztokens=${1} # 需要填下，处理测试集需要用到 
desc=xxx 
modelpath=xxx # 这里填微调的ckp路径

trainf=xxx # review.csv
```

