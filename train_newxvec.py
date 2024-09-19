import pickle as pkl
import re
import transformers
import soundfile as sf
import torch
import json
import numpy as np
from alignment_helper_fns import *
from transformers import Wav2Vec2Model, Wav2Vec2Config
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from g2p_en import G2p
from datasets import load_dataset, load_metric, load_from_disk
from src.models import Wav2Vec2ForFrameClassificationSAT
from transformers import Trainer, TrainingArguments
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Config
import argparse
from create_child_speech_dataset import *
from src.Charsiu import charsiu_forced_aligner, charsiu_sat_forced_aligner, charsiu_attention_aligner

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='charsiu')
parser.add_argument('--audio_dir', default='/home/prad/datasets/ChildSpeechDataset/child_speech_16_khz')
parser.add_argument('--results_csvpath', default='./results_xvector_proj.csv')
parser.add_argument('--manual_textgrids_dir', default='/home/prad/datasets/ChildSpeechDataset/manually-aligned-text-grids/')
parser.add_argument('--sat_vectors_csvpath', default='./extracted_xvectors_proj_libri.csv')
parser.add_argument('--model_output_dir', default='./child_speech_pretrained')
parser.add_argument('--tokenizer_name', default='charsiu/tokenizer_en_cmu')
parser.add_argument('--model_name', default='charsiu/en_w2v2_fc_10ms')
parser.add_argument('--results_dir', default='./results_sat_xvector')

parser.add_argument('--xvec_dict', default='./xvec_dict.pkl')
parser.add_argument('--output_dir', default='./new_xvec')
parser.add_argument('--device', default='cuda')
parser.add_argument('--xvec_mode', default='utterance', help='Should be `utterance` or `utterance_xvec`')

os.environ["WANDB_DISABLED"] = "true"
args = parser.parse_args()
SAT_VECTORS_CSVPATH = args.sat_vectors_csvpath

audio_dir = args.audio_dir
manual_textgrids_dir = args.manual_textgrids_dir
output_dir = args.model_output_dir
device = args.device
tokenizer_name = args.tokenizer_name
model_name = args.model_name
mode = args.mode
dataset_dir = args.dataset_dir
results_dir = args.results_dir



if args.results_dir is None:
    ii=0
    results_dir = os.path.join('./results', model_name.split('/')[-1], 'run_%d'%ii)
    while ii<100:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=False)
        else:
            results_dir = os.path.join('./results', model_name.split('/')[-1], 'run_%d' % ii)
            ii+=1

''' Load Data Paths '''
unmatched_manual_textgrid_files = get_all_textgrids_in_directory(manual_textgrids_dir)

manual_textgrid_files = []

files_to_remove = '/home/prad/datasets/ChildSpeechDataset/child_speech_16_khz/0411_M_LM/0411_M_LMwT32.wav'

audio_files = ['/'.join([audio_dir,_path.split('/')[-2], _path.split('/')[-1][:-8]+'wav'])
               for _path in unmatched_manual_textgrid_files]
audio_files.remove(files_to_remove)
''' Load Models '''
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_name)


mapping_phone2id = tokenizer.encoder
mapping_id2phone = tokenizer.decoder

''' Functions for data processing'''
def speakerwise_train_test_split(dataset, speaker_id, speaker_col='speaker_id'):
    test_split = dataset.filter(lambda example: speaker_id in example[speaker_col])
    train_split = dataset.filter(lambda example: speaker_id not in example[speaker_col])
    return train_split, test_split

def prepare_framewise_dataset(batch, mapping=mapping_phone2id):
    batch['input_values'] = batch['audio']
    batch['labels']  = []
    # for phone in batch['phone_alignments']['utterance']:
    #     batch['labels'].append(mapping[phone])
    phoneset = list(mapping_phone2id.keys())
    # if statement deals with any 'sp' tokens
    batch['labels'] = [mapping[phone] if phone in phoneset else mapping['[UNK]']
                       for phone in batch["frame_phones"]]
    return batch


@dataclass
class DataCollatorClassificationWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"], "ixvector": feature["ixvector"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    #    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    ntimesteps = pred_ids.shape[1]
    lbs = pred.label_ids[:, :ntimesteps]
    # comparison = pred_ids.equal(pred.label_ids)
    comparison = np.equal(pred_ids, lbs)
    comparison = comparison[lbs != -100].flatten()
    acc = np.sum(comparison) / len(comparison)

    return {"phone_accuracy": acc}

def get_loso_dataset(leave_out_speaker_id, dataset, dataset_prep_fn, age_restrict=False, age_range=False):
    # csd = ChildSpeechDataset(audio_paths=audio_files)
    # child_speech_dataset = csd.return_as_datsets()
    dataset = dataset.map(dataset_prep_fn)
    # dataset.save_to_disk(dataset_dir)
    # unique_speakers = list(set(list(child_speech_dataset['speaker_id'])))
    train_dataset, loso_dataset = \
            speakerwise_train_test_split(dataset, speaker_id=leave_out_speaker_id, age_restrict=age_restrict,
                                         age_range=age_range)
    # train_dataset = load_from_disk(dataset_dir+'_train')
    # loso_dataset = load_from_disk(dataset_dir+'_loso')
    return train_dataset, loso_dataset

def write_textgrid_alignments_for_speaker(charsiu_aligner, loso_dataset, output_dir):
    ''' performs the alignment for all files in the dataset (to the output dir)'''
    audiofiles = loso_dataset['file']
    transcripts = loso_dataset['sentence']
    speaker_id = loso_dataset['speaker_id'][0]
    output_tg_dir = os.path.join(output_dir, speaker_id)

    if 'ixvector' in loso_dataset.features.keys():
        ixvector = loso_dataset['ixvector']
    else:
        ixvector = None


    print('Generating Alignments...')
    print('Writing alignments to directory:\t', output_tg_dir)
    for ii, audiofilepath in tqdm.tqdm(enumerate(audiofiles)):
        file_id = loso_dataset['id'][ii]
        output_tg_path = os.path.join(output_tg_dir, file_id+'.TextGrid')
        if not os.path.exists(output_tg_dir):
            os.makedirs(output_tg_dir, exist_ok=True)
        try:
            if ixvector is not None and type(charsiu_aligner)==charsiu_sat_forced_aligner:
                charsiu_aligner.serve(audio=audiofilepath, text=transcripts[ii], save_to=output_tg_path, ixvector=ixvector[ii])
            else:
                charsiu_aligner.serve(audio=audiofilepath, text=transcripts[ii], save_to=output_tg_path)

        except:
            print('Error could not generate alignment for file:', audiofilepath)








