export PATH=/yinyongjing/anaconda3/bin/:$PATH
export OMP_NUM_THREADS=1
export TRANSFORMERS_CACHE=/yinyongjing/hfcache/
export HF_HOME=/yinyongjing/hfcache/

gpu=0
export CUDA_VISIBLE_DEVICES=$gpu

largepath=/yinyongjing/hfmodels/gpt2/large
scrip_dir=/yinyongjing/junran/cgae


gen_bs=128
rp=1.2
topk=50
topp=1.0
beams=4
temperature=1.0

ztokens=${1} 

desc=cnn-medium-z16-a1-b1-5e-5
modelpath=/yinyongjing/junran/cgae/checkpoints/cnn-medium-z16-a1-b1-5e-5/checkpoint-40383
logname=gen-$desc

cmd="$scrip_dir/eval_cnn_gen.py"

cnnf=/yinyongjing/junran/cgae/data/cnn_dailymail/

python $cmd \
    --model_name_or_path $modelpath \
    --preprocessing_num_workers 8 \
    --cnnf $cnnf \
    --ztokens $ztokens \
    --large_path $largepath \
    --per_device_eval_batch_size $gen_bs \
    --output_dir dummy \                         # 高版本不加可能报错
    --seed 42 > $scrip_dir/log.$logname 2>&1
