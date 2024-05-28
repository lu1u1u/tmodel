cd /yinyongjing/junran/cgae/tmodel-master/Next-Token-Failures/
export PATH=/yinyongjing/anaconda3/bin/:$PATH
source activate jr
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu
export TRANSFORMERS_CACHE=/yinyongjing/hfcache/
export HF_HOME=/yinyongjing/hfcache/
export HF_ENDPOINT=https://hf-mirror.com
modelname="e" # TODO: in ['mini', 's', 'm', 'l', 'e']

dp=0.1
wd=0.01
lr=1e-5
ep=100 
nn=50 # num_nodes
deg=${1}
path=${2}
a=${3}
b=${4}
snl=6
ztokens=${5}
zdim=${6}


if  [ $((deg * path)) -gt 50 ];
then
    nn=$((deg * path))
fi

if  [ "$modelname" = "s" -o "$modelname" = "mini" ];
then
    modelpath=/yinyongjing/hfmodels/gpt2_small
    bs=128
fi

if  [ "$modelname" = "mini" ];
then
    ep=500
    lr=5e-5
fi


if  [ "$modelname" = "m" ];
then
    modelpath='gpt2-medium'
    bs=32
fi

if  [ "$modelname" = "l" ];
then
    modelpath=/yinyongjing/hfmodels/gpt2/large
    bs=16
fi

if  [ "$modelname" = "e" ];
then
    modelpath='EleutherAI/gpt-neo-125M'
    bs=512
fi

echo $modelpath
echo $bs
echo $nn
echo $lr



logname=newt-suffix-$modelname-d$deg-p$path-ztokens$ztokens-a$a-b$b-zdim-$zdim-lr$lr
python ./finetune.py \
    --model $modelpath \
    --desc $logname \
    --n_train 200000 \
    --n_test 20000 \
    --batch_size $bs \
    --epochs $ep \
    --eval_every 6000 \
    --dataset graph \
    --deg $deg \
    --path $path \
    --a $a \
    --b $b \
    --zdim $zdim \
    --ztokens $ztokens \
    --weight_decay $wd \
    --dp $dp \
    --snl $snl \
    --num_nodes $nn \
    --save_every 500000 \
    --lr $lr > ./newtlogs/log.$logname 2>&1

    #--from_scratch --use_minigpt \
