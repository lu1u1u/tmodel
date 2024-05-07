export PATH=/yinyongjing/anaconda3/bin/:$PATH
#export CPATH=/usr/local/cuda/include:$CPATH

modelpath=/yinyongjing/hfmodels/gpt2/medium


scrip_dir=/yinyongjing/junran/cgae
export OMP_NUM_THREADS=1
export TRANSFORMERS_CACHE=/yinyongjing/hfcache/
export HF_HOME=/yinyongjing/hfcache/
export HF_ENDPOINT=https://hf-mirror.com
gpu=0,1,2,3
export CUDA_VISIBLE_DEVICES=$gpu

block_size=1024

ztokens=${1} # 16 32 64
alpha=${2} # default=1
beta=${3} # default=1

zdim=32
lr=5e-5
bs=16
wd=0.01
acc=1

modeltype=medium
modelname=cnn-$modeltype-z$ztokens-a$alpha-b$beta-$lr

cmd="$scrip_dir/run_cnn_newmodel.py"

cnnf=/yinyongjing/junran/cgae/data/cnn_dailymail/

#python $cmd \
deepspeed --include=localhost:${gpu} --master_port $RANDOM $cmd \
    --deepspeed $scrip_dir/ds_config.json \
    --model_name_or_path $modelpath \
    --preprocessing_num_workers 8 \
    --ztokens $ztokens \
    --alpha $alpha \
    --beta $beta \
    --zdim $zdim \
    --cnnf $cnnf \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $acc \
    --logging_steps 50 \
    --weight_decay $wd --learning_rate $lr --lr_scheduler_type constant --warmup_ratio 0.1 \
    --seed 42 \
    --bf16 \
    --block_size $block_size \
    --save_total_limit 1 --save_strategy epoch \
    --do_train --num_train_epochs 20 \
    --do_eval --evaluation_strategy epoch \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --output_dir $scrip_dir/checkpoints/$modelname > $scrip_dir/log.$modelname 2>&1
    #--load_best_model_at_end \ --do_eval --evaluation_strategy epoch \

    


