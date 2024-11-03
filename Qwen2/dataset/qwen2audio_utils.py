import torch
import librosa
from typing import Sequence, Dict

import transformers
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
AUDIO_PAD_VALUE = 0

input_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio: <|audio_bos|><|AUDIO|><|audio_eos|>\n{question}<|im_end|>\n<|im_start|>assistant\n"
full_template  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio: <|audio_bos|><|AUDIO|><|audio_eos|>\n{question}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>"
output_template = "{solution}<|im_end|>"


# Function to find the starting index of the solution in input_ids
def find_solution_indices(input_ids_list, solution_ids_list):
    for i in range(len(input_ids_list) - len(solution_ids_list) + 1):
        if input_ids_list[i:i+len(solution_ids_list)] == solution_ids_list:
            return i, i + len(solution_ids_list)
    return None, None


def process_input(
    tokenizer, 
    processor,
    wav_path, 
    question, 
    solution,
    sr=16000
):
    # Audio input
    wav = [librosa.load(wav_path, sr=sr)[0]]
    
    # Text input
    prompt = full_template.format(question=question, solution=solution)
    x = processor(text=prompt, audios=wav, return_tensors='pt', padding=False, sampling_rate=sr)

    # Create target labels (I hope this is correct...)
    input_ids = x['input_ids'][0]
    attention_mask = x['attention_mask'][0]
    solution_ids = tokenizer(output_template.format(solution=solution))['input_ids']

    start_idx, end_idx = find_solution_indices(input_ids.tolist(), solution_ids)
    if start_idx is None:
        raise ValueError("Solution tokens not found in input_ids!")
    labels = torch.full_like(input_ids, -100)
    labels[start_idx:end_idx] = input_ids[start_idx:end_idx]

    # Why input audio features are padded to 3000?
    # Qwen2Audio fixed expected_seq_length to be 3000
    audio_features = x['input_features']
    audio_masks = x['feature_attention_mask']
    # M = audio_masks.sum(dim=1).long().max()
    # audio_features = audio_features[:, :, :M]
    # audio_masks = audio_masks[:, :M]

    return dict(
        input_ids=input_ids, # variable (L)
        labels=input_ids, # variable (L)
        attention_mask=attention_mask, # variable (L)
        input_features=audio_features, # （128, La)
        feature_attention_mask=audio_masks, # （La)
    )


class DataCollator(object):
    
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]):
        input_ids, labels, attention_mask, input_features, feature_attention_mask \
            = tuple([instance[key] for instance in instances] \
            for key in ("input_ids", "labels", "attention_mask", "input_features", "feature_attention_mask"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_TOKEN_ID
        )
                
        input_features = torch.nn.utils.rnn.pad_sequence(
            [input_feature.squeeze(0).T for input_feature in input_features], # (La, 128)
            batch_first=True,
            padding_value=AUDIO_PAD_VALUE
        ).permute(0, 2, 1) # (B, 128, La)

        feature_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [mask.squeeze(0) for mask in feature_attention_mask], # (La)
            batch_first=True,
            padding_value=False
        ) # (B, La)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )

        return batch
    