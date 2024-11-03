import os
import json
import random
import librosa

import torch
from torch.utils.data import Dataset

SR = 16_000

mc2_template = \
'''Classify the sounding object into one of the category below:
A. {}
B. {}
Respond only in A or B.'''

mc4_template = \
'''Classify the sounding object into one of the category below:
A. {}
B. {}
C. {}
D. {}
Respond only in A, B, C, or D.'''

mc6_template = \
'''Classify the sounding object into one of the category below:
A. {}
B. {}
C. {}
D. {}
E. {}
F. {}
Respond only in A, B, C, D, E, or F.'''

mc8_template = \
'''Classify the sounding object into one of the category below:
A. {}
B. {}
C. {}
D. {}
E. {}
F. {}
G. {}
H. {}
Respond only in A, B, C, D, E, F, G, or H.'''


question_templates = {
    2:mc2_template,
    4:mc4_template,
    6:mc6_template,
    8:mc8_template
}


def get_audio_path(audio_root, x):
    if isinstance(x, dict):
        _id = x['id']
        start_second = int(x['start_second'])
        name = _id + '_' + str(int(start_second)).zfill(6) + '.wav'
    elif isinstance(x, str):
        name = x + '.wav'
    else:
        raise NotImplementedError(type(x))
    return os.path.join(audio_root, name)


def get_video_path(video_root, x):
    if isinstance(x, dict):
        _id = x['id']
        start_second = int(x['start_second'])
        name = _id + '_' + str(int(start_second)).zfill(6) + '.mp4'
    elif isinstance(x, str):
        name = x + '.mp4'
    else:
        raise NotImplementedError(type(x))
    return os.path.join(video_root, name)


def draw_negative_labels_uniform(labels, pos_label, n_neg):
    neg_labels = [label for label in labels if label != pos_label]
    
    if len(neg_labels) < n_neg:
        raise ValueError(f"Not enough elements to draw {str(n_neg)} negative labels.")
    
    return random.sample(neg_labels, n_neg)


def draw_negative_labels_weighted(label_distributions, pos_label, n_neg):
    neg_labels = {label: label_distributions[label] for label in label_distributions if label != pos_label}
    
    if len(neg_labels) < n_neg:
        raise ValueError(f"Not enough elements to draw {str(n_neg)} negative labels.")

    results = random.choices(list(neg_labels.keys()), weights=list(neg_labels.values()), k=n_neg)
    while len(set(results)) < n_neg:
        results = random.choices(list(neg_labels.keys()), weights=list(neg_labels.values()), k=n_neg)
    
    return results


class RandomVGGSoundCLSMC(Dataset):
    def __init__(self, 
        manifest_path,
        label_distributions_path=None, 
        neg_choices_by='uniform',
        pos_choice_pos='random',
        n_choice=4,
    ):
        '''
        manifest_path:
            the json path of VGGSound samples with a SINGLE label
        label_distributions_path: 
            the distributions (sum to  1) of all labels
        neg_choices_by:
            how to sample negative samples    
        pos_choice_pos;
            where (should be a random position) to put the correct choice in MC
        n_choice:
            number of choices in MC including the correct one
        '''

        self.data = json.load(open(manifest_path, 'r'))

        assert neg_choices_by in ['uniform', 'distribution']
        self.neg_choices_by = neg_choices_by
        self.label_distributions = json.load(open(label_distributions_path, 'r'))
        self.labels = list(self.label_distributions.keys())

        assert pos_choice_pos == 'random'
        self.pos_choice_pos = pos_choice_pos
            
        assert n_choice in [2, 4, 6, 8]
        self.n_choice = n_choice

        self.question_template = question_templates[n_choice]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        assert len(x['labels']) == 1
        label = x['labels'][0]

        name = x['id'] + '_' + str(int(x['start_second'])).zfill(6)

        # Randomly sample n_choice - 1 negative labels
        if self.neg_choices_by == 'uniform':
            neg_labels = draw_negative_labels_uniform(self.labels, label, self.n_choice-1)
        elif self.neg_choices_by == 'distribution':
            neg_labels = draw_negative_labels_weighted(self.label_distributions, label, self.n_choice-1)
        else:
            raise NotImplementedError(self.neg_choices_by)

        choices = [label] + neg_labels

        # Make sure that the correct choice is not at a fixed position
        if self.pos_choice_pos == 'random':
            random.shuffle(choices)
        else:
            raise NotImplementedError(self.pos_choice_pos)

        question = self.question_template.format(*choices)

        label_index = choices.index(label)
        solution =  chr(label_index + 65) # 0 -> 'A', 1 -> 'B', and etc

        return {
            'name': name,
            'label': label,
            'question': question,
            'solution': solution,
        }


class ReproduceVGGSoundCLSMC(Dataset):
    def __init__(self,
        qa_path,
        data_root,
        get_path_fn=get_audio_path
    ):
        '''
        qa_path:
            the json path of all Q/A, generated by RandomVGGSoundCLSMC
        data_root:
            where wav or mp4 files are stored
        get_path_fn:
            return the audio or video path from the data_root and this sample
        '''

        self.qa = json.load(open(qa_path, 'r'))
        self.data_root = data_root
        self.get_path_fn = get_path_fn

    def __len__(self):
        return len(self.qa)

    def __getitem__(self, idx):
        x = self.qa[idx]
        data_path = self.get_path_fn(self.data_root, x['name'])
        x['path'] = data_path

        return x


######### Dataset for finetuning ##############################################

import transformers
from .qwen2audio_utils import process_input

class Qwen2AudioSFTRandomVGGSoundCLSMC(RandomVGGSoundCLSMC):

    def __init__(self, 
        manifest_path,
        data_root,
        tokenizer: transformers.PreTrainedTokenizer, 
        processor: transformers.AutoProcessor,
        get_path_fn=get_audio_path,
        label_distributions_path=None, 
        neg_choices_by='uniform',
        pos_choice_pos='random',
        n_choice=4,
    ):
        '''
        manifest_path:
            the json path of VGGSound samples with a SINGLE label
        label_distributions_path: 
            the distributions (sum to  1) of all labels
        neg_choices_by:
            how to sample negative samples    
        pos_choice_pos;
            where (should be a random position) to put the correct choice in MC
        n_choice:
            number of choices in MC including the correct one
        '''
        self.get_path_fn = get_path_fn
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.processor = processor

        super().__init__(
            manifest_path=manifest_path,
            label_distributions_path=label_distributions_path,
            neg_choices_by=neg_choices_by,
            pos_choice_pos=pos_choice_pos,
            n_choice=n_choice
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, raw=False):
        x = super().__getitem__(idx)

        if not raw:
            x = process_input(
                tokenizer=self.tokenizer,
                processor=self.processor,
                wav_path=self.get_path_fn(self.data_root, x['name']),
                question=x['question'],
                solution=x['solution'],
                sr=SR
            )

        return x

###############################################################################