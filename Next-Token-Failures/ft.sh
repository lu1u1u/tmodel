export PATH=/home/yinyongjing/anaconda3/bin/:$PATH
source activate minilm
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu
export TOKENIZERS_PARALLELISM=false
modelname="s"

deg=${1}
path=${2}
a=${3}
b=${4}
snl=6
ztokens=${5}
zdim=${6}

if  [ "$modelname" = "s" ];
then
    modelpath=/data3/home/fulian/djr/small
    bs=512
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



#ae_model_name_or_path=/data3/home/fulian/djr/small
ae_model_name_or_path=/data3/home/fulian/djr/large
logname=MINI-$modelname-d$deg-p$path-ztokens$ztokens-a$a-b$b-zdim-$zdim-lr$lr
python ./finetune.py \
    --use_minigpt --from_scratch \
    --use_flash_attention \
    --model $modelpath \
    --ae_model_name_or_path $ae_model_name_or_path \
    --desc $logname \
    --n_train 200000 \
    --n_test 20000 \
    --batch_size $bs \
    --epochs 500 \
    --eval_every 3000 \
    --dataset graph \
    --deg $deg \
    --path $path \
    --a $a \
    --b $b \
    --ztokens $ztokens \
    --snl $snl \
    --num_nodes 50 \
    --save_every 50000 \
    --lr 0.00001 > ./newlogs/log.$logname 2>&1

    #    --use_minigpt \     --from_scratch \
