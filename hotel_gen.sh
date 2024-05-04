export PATH=/yinyongjing/anaconda3/bin/:$PATH
export OMP_NUM_THREADS=1
export TRANSFORMERS_CACHE=/yinyongjing/hfcache/
export HF_HOME=/yinyongjing/hfcache/
export HF_ENDPOINT=https://hf-mirror.com
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

desc=newhotel-medium-z16-a1-b1-1e-4
modelpath=/yinyongjing/junran/cgae/checkpoints/newhotel-medium-z16-a1-b1-1e-4/checkpoint-12526
logname=gen-$desc

cmd="$scrip_dir/eval_hotel_gen.py"

trainf=/yinyongjing/junran/cgae/data/reviews.csv

python $cmd \
    --model_name_or_path $modelpath \
    --preprocessing_num_workers 8 \
    --ztokens $ztokens \
    --train_file $trainf \
    --large_path $largepath \
    --gen_batch_size $gen_bs \
    --seed 42 > $scrip_dir/log.$logname 2>&1

    


