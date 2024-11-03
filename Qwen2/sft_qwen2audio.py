import os
import sys
import wandb
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from trainer.qwen2audio_sft import SFTTrainer

os.environ['WANDB__SERVICE_WAIT'] = '999999'
os.environ['TOKENIZERS_PARALLELISM'] = 'True'

if __name__ == '__main__':
   
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    os.environ['WANDB_PROJECT'] = hparams['project']
    hparams['lora_qwen'].print_trainable_parameters()
    
    if hparams['bf16']:
        hparams['lora_qwen'] = hparams['lora_qwen'].bfloat16()
        print('Cast model to bf16.')

    trainer = SFTTrainer(
        model=hparams['lora_qwen'], 
        tokenizer=hparams['tokenizer'],
        train_dataset=hparams['train_data'],
        eval_dataset=hparams['valid_data'],
        data_collator=hparams['data_collator'],
        args=hparams['train_config'],
    )
    
    try:
        trainer.train()
        trainer.save_state()
        wandb.finish()
    except Exception as E:
        print(E)
        wandb.finish()
