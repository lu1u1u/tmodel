from models.gpt import GPT
from models.pythia import Pythia
from models.config import GPTConfig


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
    #elif args.model.startswith('/yinyongjing/') or args.model.startswith('EleutherAI'):
    else:   
        model_args = kwargs.get("model_args")
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        # added
        tokenizer = kwargs.get("tokenizer")
        if args.use_minigpt:
            assert 'small' in args.model
            config.update(
                {
                    "n_layer" : 6,
                    "n_embd" : 384,
                    "n_head" : 6,
                    "resid_pdrop" : args.dp, 
                    "embd_pdrop" : args.dp,
                    "summary_first_dropout" : args.dp,
                    "hidden_dropout_prob" : args.dp
                }
            )
        config.update(
            {
                "ztokens" : model_args.ztokens,
                "zdim" : model_args.zdim,
                "z_start_id" : tokenizer.z_start_id,
                "len_tokenizer": tokenizer.vocab_size,
                "enable_ae_decoder_emb_grad" : args.enable_ae_decoder_emb_grad
            }
        )
        if model_args.use_flash_attention:
            config._attn_implementation = "flash_attention_2"
            
        if args.use_new:
            from models.newnew import NewTModel
        elif args.no_ae:
            from models.noae import NewTModel
        elif 'neo' in args.model:
            from models.neo import NewTModel
        elif args.use_kt:
            from models.kt import KT as NewTModel
        else:
            from models.newt import NewTModel
            
        tmodel = NewTModel(config = config, model_args = model_args)
        if hasattr(tmodel, "build_ed"):
            tmodel.build_ed(tokenizer.vocab_size)
        print(tmodel)
    return tmodel
