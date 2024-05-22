export PATH=/home/yinyongjing/anaconda3/bin/:$PATH
source activate minilm
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu

modelname="l" # TODO: in ['mini', 's', 'm', 'l']

lr=1e-4
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
    modelpath=/data3/home/fulian/djr/small
    bs=128
fi

if  [ "$modelname" = "mini" ];
then
    ep=500
    lr=5e-4
fi


if  [ "$modelname" = "m" ];
then
    modelpath='gpt2-medium'
    bs=32
fi

if  [ "$modelname" = "l" ];
then
    modelpath=/data3/home/fulian/djr/large
    bs=32
fi

echo $modelpath
echo $bs
echo $nn
echo $lr



logname=newt-suffix-$modelname-d$deg-p$path-ztokens$ztokens-a$a-b$b-zdim-$zdim-lr$lr
python ./finetune.py \
    --model $modelpath \ #--from_scratch --use_minigpt \
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
    --zdim $zdim \ # -1 stands for model_size
    --ztokens $ztokens \
    --snl $snl \
    --num_nodes $nn \
    --save_every 500000 \
    --lr $lr > ./newtlogs/log.$logname 2>&1
