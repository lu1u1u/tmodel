export PATH=/yinyongjing/anaconda3/bin/:$PATH
#export CPATH=/usr/local/cuda/include:$CPATH

modelpath=/yinyongjing/hfmodels/gpt2/medium
largepath=/yinyongjing/hfmodels/gpt2/large

scrip_dir=/yinyongjing/junran/cgae
export OMP_NUM_THREADS=1
export TRANSFORMERS_CACHE=/yinyongjing/hfcache/
export HF_HOME=/yinyongjing/hfcache/
export HF_ENDPOINT=https://hf-mirror.com
gpu=0,1
export CUDA_VISIBLE_DEVICES=$gpu

block_size=512

ztokens=${1} # 16 32 64
alpha=${2} # default=1
beta=${3} # default=1

zdim=32
lr=1e-4
bs=32
wd=0.01
acc=1

modeltype=medium
modelname=newhotel-$modeltype-z$ztokens-a$alpha-b$beta-$lr

cmd="$scrip_dir/run_hotel_newmodel.py"

trainf=/yinyongjing/junran/cgae/data/reviews.csv

#python $cmd \
deepspeed --include=localhost:${gpu} --master_port $RANDOM $cmd \
    --deepspeed $scrip_dir/ds_config.json \
    --model_name_or_path $modelpath \
    --preprocessing_num_workers 8 \
    --ztokens $ztokens \
    --alpha $alpha \
    --beta $beta \
    --zdim $zdim \
    --train_file $trainf \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $acc \
    --logging_steps 50 \
    --weight_decay $wd --learning_rate $lr --lr_scheduler_type constant --warmup_ratio 0.1 \
    --seed 42 \
    --bf16 \
    --block_size $block_size \
    --save_total_limit 1 --save_strategy epoch \
    --do_train \
    --num_train_epochs 1 \
    --do_eval --evaluation_strategy epoch \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --output_dir $scrip_dir/checkpoints/$modelname > $scrip_dir/log.$modelname 2>&1

    


