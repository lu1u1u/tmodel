from models.gpt import GPT
from models.pythia import Pythia
from models.config import GPTConfig

from models.newt import NewTModel
from transformers import (
    AutoConfig,

)



def get_model(args, **kwargs):
    if args.model == 'gpt':
        config = GPTConfig(n_layers=args.n_layer, n_heads=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                           bias=True, vocab_size=args.vocab_size, dropout=0, use_flash=args.use_flash,
                           teacherless_token=args.teacherless_token)
        model = GPT(config)

    elif args.model.startswith('gpt2'):
        model = GPT.from_pretrained(args.model, teacherless_token=args.teacherless_token)
        if args.block_size < 1024:
            model.crop_block_size(args.block_size)

    elif args.model.startswith('pythia'):
        model = Pythia.from_pretrained(args.model, teacherless_token=args.teacherless_token)
    elif args.model.startswith('/yinyongjing/hfmodels/gpt2'):
        
        model_args = kwargs.get("model_args")
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        tmodel = NewTModel(config = config, model_args = model_args )
        tokenizer = kwargs.get("tokenizer")
        tmodel.build_ed(tokenizer.vocab_size)
    return tmodel
