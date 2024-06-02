import argparse
from contextlib import nullcontext
import torch
from tqdm import tqdm
from accelerate import Accelerator
from data import get_dataset
from utils.training_utils import get_lr, get_run_name, AverageMeter
from torch.utils.data import DataLoader
from evaluate import evaluate, evaluate_forced
from models import get_model
from tokenizing import get_tokenizer
import wandb
import threading
from accelerate import DistributedDataParallelKwargs


# Parse arguments
parser = argparse.ArgumentParser(description="Next-token failures")
# Data
parser.add_argument(
    "--model", default='gpt2', type=str, help="Type of model"
    )
parser.add_argument(
    "--dataset", default='graph', type=str, help="Choice of dataset"
    )
parser.add_argument(
    "--n_train", default=200000, type=int, help="Number of training samples"
    )
parser.add_argument(
    "--n_test", default=5000, type=int, help="Number of test samples"
    )
parser.add_argument(
    "--num_nodes", default=50, type=int, help="Number of node values in graph"
    )
parser.add_argument(
    "--deg", default=2, type=int, help="Degree of starting node"
    )
parser.add_argument(
    "--path_len", default=5, type=int, help="Path length in star graph"
    )
parser.add_argument(
        "--mate_in", default=2, type=int, help="For chess, number of moves to checkmate"
    )
parser.add_argument(
        "--unrolled", action=argparse.BooleanOptionalAction, default=True, help="For chess, unrolled board state",
    )
parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size",
    )
parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate",
    )
parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Strength of weight decay",
    )

parser.add_argument(
        "--dp", type=float, default=0.1, help="dropout",
    )

parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="dropout",
    )

parser.add_argument(
        "--m", type=float, default=0.999, help="mom",
    )

parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs",
    )
parser.add_argument(
        "--save_every", type=int, default=5000, help="Interval (in steps) at which to save model",
    )
parser.add_argument(
        "--teacherless", action=argparse.BooleanOptionalAction, default=False, help="Standard or teacherless training",
    )
parser.add_argument(
        "--reverse", action=argparse.BooleanOptionalAction, default=False, help="Standard format or reverse targets",
    )
parser.add_argument(
        "--eval_train", action=argparse.BooleanOptionalAction, default=False, help="Eval for training set",
    )
parser.add_argument(
        "--eval_every", type=int, default=5000, help="Interval (in steps) to evaluate the model on test",
    )
parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=False, help="Whether to use wandb",
    )
parser.add_argument(
        "--wandb_entity", type=str, default=5000, help="Wandb username",
    )

#added
parser.add_argument(
        "--ztokens", type=int, default=4, help="ztokens",
    )
parser.add_argument(
        "--a", type=int, default=1, help="alpha",
    )
parser.add_argument(
        "--b", type=int, default=1, help="beta",
    )
parser.add_argument(
        "--zdim", type=int, default=32, help="zdim",
    )

parser.add_argument(
        "--snl", type=int, default=6, help="shallow_decoder_n_layer",
    )

parser.add_argument(
        "--desc", type=str, help="desc",
    )

parser.add_argument(
        "--fullenc",  action = 'store_true', default = False, help = 'choose to use full encstr',
    )

parser.add_argument(
        "--znorm",  action = 'store_true', default = False, help = 'znorm',
    )

parser.add_argument(
        "--use_flash_attention",  action = 'store_true', default = False, help = 'fls',
    )

parser.add_argument(
        "--from_scratch",  action = 'store_true', default = False, help = 'scratch',
    )

parser.add_argument(
        "--use_minigpt",  action = 'store_true', default = False, help = 'use_minigpt',
    )

parser.add_argument(
        "--use_new",  action = 'store_true', default = False, help = 'use_newnew',
    )
parser.add_argument(
        "--no_ae",  action = 'store_true', default = False, help = 'use_noae',
    )

parser.add_argument(
        "--weaken_dec",  action = 'store_true', default = False, help = 'use_noae',
    )

parser.add_argument(
        "--ae_model_name_or_path",  type=str, default = 'gpt2', help = 'ae_path',
    )

parser.add_argument(
        "--enable_ae_decoder_emb_grad",  action = 'store_true', default = False, help = 'enable_ae_decoder_emb_grad',
    )

parser.add_argument(
        "--use_ema",  action = 'store_true', default = False, help = 'use_ema',
    )

parser.add_argument(
        "--use_separate",  action = 'store_true', default = False, help = 'use_separate',
    )

parser.add_argument(
        "--use_kt",  action = 'store_true', default = False, help = 'use_predict_k_tokens',
    )

parser.add_argument(
        "--k", type=int, default=3, help="kt num heads",
    )

parser.add_argument(
        "--disable_search_unused_parameters",  action = 'store_true', default = False, help = 'd',
    )
args = parser.parse_args()

# Parallel
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=not args.disable_search_unused_parameters)
accelerator = Accelerator(mixed_precision = 'fp16', kwargs_handlers=[ddp_kwargs])


if args.use_new:
    print('Using newnew model instead of newt...')
if args.no_ae:
    print('Using noae instead of newt...')
if args.use_separate:
    assert not args.no_ae

class ModelArguments:
    model_name_or_path = args.model
    ae_model_name_or_path = args.ae_model_name_or_path
    ztokens = args.ztokens
    zdim = args.zdim
    shallow_decoder_n_layer = args.snl
    alpha = args.a
    beta = args.b
    spname = f"{args.desc}".replace('-','_')
    fullenc = args.fullenc
    znorm = args.znorm
    use_flash_attention = args.use_flash_attention 
    from_scratch = args.from_scratch
    weaken_dec = args.weaken_dec
    use_ema = args.use_ema
    use_separate = args.use_separate
    m = args.m
    k = args.k
    msenorm = 1 # 1 : l1; 2 : l2
    
model_args = ModelArguments()   
def print_settings():
    print("====================Training Details=====================")
    print("spname = ", model_args.spname)
    print("model_name_or_path = ",model_args.model_name_or_path)
    print(f"lr = {args.lr}, min_lr = {args.min_lr}")
    print("use_flash_attention = ", model_args.use_flash_attention)
    print(f"no_ae = {args.no_ae}")
    print(f"use_ema = {model_args.use_ema}")
    if not args.use_new and args.use_ema:
        print(" => m = ", model_args.m)
    print("use_kt = ", args.use_kt)
    if args.use_kt:
        print(" => k = ", model_args.k)
    print(f"use_separate = {model_args.use_separate}")
    print("\n")

    print("====================Data Details=====================")
    print("ztokens = ",model_args.ztokens)
    print("zdim = ", model_args.zdim)
    print("use full encstr = ", model_args.fullenc)
    print("\n")

    print("====================Model Details=====================")
    print("from_scratch = ", model_args.from_scratch)
    print("shallow_decoder_n_layer = ", model_args.shallow_decoder_n_layer)
    print("alpha = ", model_args.alpha)
    print("beta = ", model_args.beta)
    print("weaken_dec = ", model_args.weaken_dec)
    print("\n")

    print("====================Others=====================")
    print("ae_model_name_or_path(not used) = ", model_args.ae_model_name_or_path)
    print("znorm(not used) = ", model_args.znorm)
    print("\n")
    

with threading.Lock():
    print_settings()
# System stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb_entity = args.wandb_entity
wandb_log = args.use_wandb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Model stuff
top_k = 1

# Evaluation stuff
eval_iters = 1000
eval_interval = 5
log_interval = 10

# Optimiser
dtype = 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
beta1 = 0.9
beta2 = 0.999
decay_lr = True
args.compile = False if device == 'cuda' else False
args.use_flash = True if device == 'cuda' else False
warmup_iters = 100
min_lr = args.min_lr

run_name = get_run_name(args)
path = './checkpoints/' + run_name + '.pt'

# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args, model_args=model_args)
train_data, test_data = get_dataset(args, tokenizer, device)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

target_len = train_data.num_tokens - train_data.num_prefix_tokens
max_iters = len(train_data) * args.epochs
lr_decay_iters = max_iters

block_size = train_data.num_tokens
args.block_size = train_data.num_tokens
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None

model = get_model(args, model_args=model_args, tokenizer=tokenizer)

if args.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)



model.train()

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

model, optimizer, train_loader, test_loader = accelerator.prepare(
    model, optimizer, train_loader, test_loader)

# Setup wandb logging
if wandb_log:
    wandb.init(project='next-token-failures', entity=wandb_entity, config=args.__dict__,)
    wandb.run.name = run_name

results = {}
num_iters = 0

for ep in range(args.epochs):
    train_bar = tqdm(train_loader)
    total_loss, total_acc, total_ptk_acc = AverageMeter(), AverageMeter(), AverageMeter()

    for tp in train_bar:

        # determine and set the learning rate for this iteration
        lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with accelerator.accumulate(model), ctx:
            if isinstance(tp, list) or isinstance(tp, tuple):
                logits, loss, accs = model(*tp)
                bs = tp[0].shape[0]
                
            elif isinstance(tp, dict):
                ret = model(**tp)
                loss = ret.loss
                logits = ret.logits
                accs = ret.acc 
                bs = tp['input_ids'].shape[0]
            else:
                assert 0
            
            total_ptk_acc.update(accs['token_acc'], bs)
            total_loss.update(loss.item(), bs * train_data.num_target_tokens)
            total_acc.update(accs['acc'], bs * train_data.num_target_tokens)
            accelerator.backward(scaler.scale(loss))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # ema
            if args.use_ema:
                model._momentum_update_encoder()
        num_iters += 1
        train_bar.set_description(
            'Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f} Ptk: {}'.format(ep, args.epochs, total_loss.get(),
             total_acc.get(percentage=True),
             total_ptk_acc.get_tensor_for_display())
        )

        # evaluate the loss on train/val sets and write checkpoints
        if num_iters % args.eval_every == 0 and num_iters > 1:
            # Generate sequences and check accuracies
            if args.eval_train:
                results = evaluate(model, train_loader, temperature=0.8, top_k=top_k, results=results, mode='Train')
                results = evaluate_forced(model, train_loader, results=results, mode='train')

            results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='Test', accelerator=accelerator)
            results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='Test', accelerator=accelerator)

            print(results)
            if wandb_log:
                wandb.log(results)
