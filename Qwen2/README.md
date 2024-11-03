## Supervised Finetuning
Qwen2Audio:
```
python sft_qwen2audio.py hparams/sft/qwen2audio_all_vggsound_clsmc.yaml \
--project <YOUR_WANDB_PROJECT_NAME> \
--experiment qwen2audio_ALL_VGGSound_MC4_distribution_r16_ep1 \
--output_dir <WHERE_TO_SAVE_LORA_CKPT> \
--audio_root <WHERE_AUDIOS_ARR_STORED>
```
