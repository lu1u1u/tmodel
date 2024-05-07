cd /yinyongjing/junran/cgae/Next-Token-Failures
export PATH=/yinyongjing/anaconda3/bin/:$PATH


modelname="l"

deg=${1}
path=${2}
a=${3}
b=${4}
snl=6
ztokens=${5}
zdim=32

if  [ "$modelname" = "s" ];
then
    modelpath=/yinyongjing/hfmodels/gpt2_small
    bs=128
fi

if  [ "$modelname" = "m" ];
then
    modelpath=/yinyongjing/hfmodels/gpt2/medium
    bs=32
fi

if  [ "$modelname" = "l" ];
then
    modelpath=/yinyongjing/hfmodels/gpt2/large
    bs=32
fi

echo $modelpath
echo $bs



logname=fullenc-$modelname-d$deg-p$path-ztokens$ztokens-a$a-b$b
python3 ./finetune.py \
    --fullenc \
    --model $modelpath \
    --desc $logname \
    --n_train 200000 \
    --n_test 20000 \
    --batch_size $bs \
    --epochs 100 \
    --eval_every 3000 \
    --dataset graph \
    --deg $deg \
    --path $path \
    --a $a \
    --b $b \
    --snl $snl \
    --num_nodes 50 \
    --save_every 50000 \
    --lr 0.00001 > ./logs/log.$logname 2>&1