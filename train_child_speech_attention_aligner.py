#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import transformers
import soundfile as sf
import torch
import json
import numpy as np
from alignment_helper_fns import *

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from g2p_en import G2p
from datasets import load_dataset, load_metric, load_from_disk
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from src.models import Wav2Vec2ForAttentionAlignment
from src.Charsiu import charsiu_attention_aligner
from transformers import Trainer, TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices,Wav2Vec2ForPreTrainingOutput
from transformers.modeling_outputs import CausalLMOutput, MaskedLMOutput
import argparse
from create_child_speech_dataset import *
np.random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='charsiu')
parser.add_argument('--audio_dir', default='/home/prad/datasets/ChildSpeechDataset/child_speech_16_khz')
parser.add_argument('--manual_textgrids_dir', default='/home/prad/datasets/ChildSpeechDataset/manually-aligned-text-grids/')
parser.add_argument('--model_output_dir', default='./child_speech_pretrained')
parser.add_argument('--tokenizer_name', default='charsiu/tokenizer_en_cmu')
parser.add_argument('--model_name', default='charsiu/en_w2v2_fs_10ms')
parser.add_argument('--results_dir', default='./results_attn_tmp')
# parser.add_argument('--tokenizer_name', default='facebook/wav2vec2-base-960h')
# parser.add_argument('--model_name', default='facebook/wav2vec2-base-960h')


parser.add_argument('--dataset_dir', default='./data/child_speech_framewise')
parser.add_argument('--output_dir', default='./outputs')
parser.add_argument('--device', default='cuda')
# parser.add_argument('--')
args = parser.parse_args()
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
    results_dir = os.path.join(results_dir, model_name.split('/')[-1], 'run_%d'%ii)
    while ii<100:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=False)
        else:
            results_dir = os.path.join(results_dir, model_name.split('/')[-1], 'run_%d' % ii)
            ii+=1

''' Load Data Paths '''
unmatched_manual_textgrid_files = get_all_textgrids_in_directory(manual_textgrids_dir)

manual_textgrid_files = []
audio_files = ['/'.join([audio_dir,_path.split('/')[-2], _path.split('/')[-1][:-8]+'wav'])
               for _path in unmatched_manual_textgrid_files]
''' Load Models '''
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_name)


mapping_phone2id = tokenizer.encoder
mapping_id2phone = tokenizer.decoder

''' Functions for data processing'''
def speakerwise_train_test_split(dataset, speaker_id, speaker_col='speaker_id'):
    test_split = dataset.filter(lambda example: speaker_id in example[speaker_col])
    train_split = dataset.filter(lambda example: speaker_id not in example[speaker_col])
    return train_split, test_split

def prepare_attn_dataset(batch, mapping=mapping_phone2id):
    batch['input_values'] = batch['audio']

    batch['labels']  = []
    # for phone in batch['phone_alignments']['utterance']:
    #     batch['labels'].append(mapping[phone])
    phoneset = list(mapping_phone2id.keys())
    # if statement deals with any 'sp' tokens
    batch['labels'] = [mapping[phone] if phone in phoneset else mapping['[UNK]']
                       for phone in batch["phone_alignments"]['utterance']]
    batch['frame_labels'] = []

    batch['frame_labels'] = [mapping[phone] if phone in phoneset else mapping['[UNK]']
                       for phone in batch["frame_phones"]]
    return batch

''' Create dataset instance'''
if not os.path.exists(dataset_dir):
    print('Dataset not found at:\t', dataset_dir, '\nCreating and saving dataset')
    csd = ChildSpeechDataset(audio_paths=audio_files)
    child_speech_dataset = csd.return_as_datsets()
    child_speech_dataset = child_speech_dataset.map(prepare_attn_dataset)
    child_speech_dataset.save_to_disk(dataset_dir)
    unique_speakers = list(set(list(child_speech_dataset['speaker_id'])))
    train_dataset, loso_dataset = \
            speakerwise_train_test_split(child_speech_dataset, speaker_id=unique_speakers[0])
    train_dataset.save_to_disk(dataset_dir+'_train')
    loso_dataset.save_to_disk(dataset_dir+'_loso')
else:
    print('Found dataset at:\t', dataset_dir, '\nLoading')
    child_speech_dataset = load_from_disk(dataset_dir)
    unique_speakers = list(set(list(child_speech_dataset['speaker_id'])))
    train_dataset = load_from_disk(dataset_dir+'_train')
    loso_dataset = load_from_disk(dataset_dir+'_loso')



''' Dataset preparation '''
def prepare_dataset_20ms(batch):
    batch["input_values"] = batch['file']
    batch["labels"] = [mapping_phone2id[p] for p in batch['frame_labels']]
    assert len(batch['frame_labels']) == len(batch['labels'])
    return batch


def prepare_dataset_10ms(batch):
    batch["input_values"] = batch['file']
    #    batch["labels"] = [mapping_phone2id[p] for p in batch['frame_labels_10ms']]
    batch["labels"] = [mapping_phone2id[p] for p in batch['labels']]
    assert len(batch['frame_labels_10ms']) == len(batch['labels'])
    return batch


def prepare_test_dataset_10ms(batch):
    batch["input_values"] = batch['file']
    batch["labels"] = [mapping_phone2id[p] for p in batch['frame_labels_10ms']]
    #    batch["labels"] = [mapping_phone2id[p] for p in batch['labels']]
    assert len(batch['frame_labels_10ms']) == len(batch['labels'])
    return batch


def prepare_dataset_cv(batch):
    batch["input_values"] = batch['path'].replace('.mp3', '.wav')
    batch["labels"] = [mapping_phone2id[p] for p in batch['labels']]
    assert len(batch['labels']) == len(batch['labels'])
    return batch


def audio_preprocess(path):
    #TODO: consider removing because this isnt used
    features, sr = sf.read(path)
    assert sr == 16000
    return processor(features, sampling_rate=16000).input_values.squeeze()


@dataclass
class SpeechCollatorWithPadding:
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
    max_framelen: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        text_len = [len(i['input_ids']) for i in label_features]



        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_attention_mask=self.return_attention_mask,
                return_tensors="pt",
            )

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        mel_len = [model._get_feat_extract_output_lengths(len(i['input_values'])) for i in input_features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_framelen,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        batch_size, raw_sequence_length = batch['input_values'].shape
        # if self.max_framelen is not None:
        #     sequence_length = self.max_framelen
        # else:
        sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)

        mask_prob = torch.randint(size=(1, ), low=1, high=40)/100
        masked_inds = _compute_mask_indices((batch_size, sequence_length), mask_prob=mask_prob, mask_length=2)
        masked_inds = torch.tensor(masked_inds)
        batch['mask_time_indices'] = masked_inds
        batch['frame_len'] = torch.tensor(mel_len)
        batch['text_len'] = torch.tensor(text_len)
        batch['labels'] = labels_batch['input_ids']
        batch['labels_attention_mask'] = labels_batch['attention_mask']

        return batch


''' [START] Models '''



'''[END] Models '''


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


# tokenizer = Wav2Vec2CTCTokenizer("./dict/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="")
# feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
#                                              return_attention_mask=False)
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def get_loso_dataset(leave_out_speaker_id, dataset, dataset_prep_fn):
    # csd = ChildSpeechDataset(audio_paths=audio_files)
    # child_speech_dataset = csd.return_as_datsets()
    dataset = dataset.map(dataset_prep_fn)
    # dataset.save_to_disk(dataset_dir)
    # unique_speakers = list(set(list(child_speech_dataset['speaker_id'])))
    train_dataset, loso_dataset = \
            speakerwise_train_test_split(dataset, speaker_id=leave_out_speaker_id)
    # train_dataset = load_from_disk(dataset_dir+'_train')
    # loso_dataset = load_from_disk(dataset_dir+'_loso')
    return train_dataset, loso_dataset

def write_textgrid_alignments_for_speaker(charsiu_aligner, loso_dataset, output_dir):
    ''' performs the alignment for all files in the dataset (to the output dir)'''
    audiofiles = loso_dataset['file']
    transcripts = loso_dataset['sentence']
    speaker_id = loso_dataset['speaker_id'][0]
    print('Generating Alignments...')
    for ii, audiofilepath in tqdm.tqdm(enumerate(audiofiles)):
        file_id = loso_dataset['id'][ii]
        output_tg_path = os.path.join(output_dir, speaker_id, file_id+'.TextGrid')
        output_tg_dir = os.path.join(output_dir, speaker_id)
        # print(audiofilepath)
        if not os.path.exists(output_tg_dir):
            os.makedirs(output_tg_dir, exist_ok=True)
        charsiu_aligner.serve(audio=audiofilepath, text=transcripts[ii], save_to=output_tg_path)
        # manualdf = textgridpath_to_phonedf('/home/prad/datasets/ChildSpeechDataset/manually-aligned-text-grids/'+speaker_id+'/'+file_id+'.TextGrid', phone_key='ha phones')
        # try:
        #     charsiu_aligner.serve(audio = audiofilepath, text=transcripts[ii], save_to = output_tg_path)
        # except:
        #     print('Error could not generate alignment for file:', audiofilepath)


if __name__ == "__main__":
    # index =
    # Remove the following file: /home/prad/datasets/ChildSpeechDataset/child_speech_16_khz/0411_M_LM/0411_M_LMwT32.wav
    # because it is detected as empty and causes an error for the aligner's DTW function
    child_speech_dataset = load_from_disk(dataset_dir)
    file_to_remove = '/home/prad/datasets/ChildSpeechDataset/child_speech_16_khz/0411_M_LM/0411_M_LMwT32.wav'
    child_speech_dataset = child_speech_dataset.filter(lambda example: file_to_remove not in example)
    unique_speakers = list(set(list(child_speech_dataset['speaker_id'])))
    speaker_wise_results = pd.DataFrame(index=unique_speakers)

    completed_speakers = [x[0].split('/')[-1] for x in os.walk(results_dir)]
    speakers_to_run = list(set(unique_speakers) - set(completed_speakers))
    for jj, loso_speaker_id in enumerate(unique_speakers):
        loso_speaker_id = '0307_M_EC'
        print('-------------------------------------------------------------------------------------------------------')
        print('************************************ Started Training for Speaker *************************************')
        print('\n\n+++++++++++++++++ Speaker %s (%d/%d)+++++++++++++++++' % (loso_speaker_id, jj, len(unique_speakers)))

        train_dataset, loso_dataset = get_loso_dataset(leave_out_speaker_id=loso_speaker_id,
                                                       dataset=child_speech_dataset,
                                                       dataset_prep_fn=prepare_attn_dataset)

        frameshift = 10

        if frameshift == 10:
            prepare_dataset = prepare_dataset_10ms
        #        prepare_dataset = prepare_dataset_cv
        else:
            prepare_dataset = prepare_dataset_20ms

        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)




        if mode == 'base':
            model = Wav2Vec2ForAttentionAlignment.from_pretrained(
                "facebook/wav2vec2-base",
                gradient_checkpointing=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=len(processor.tokenizer)
            )

        elif mode == 'charsiu':
            model = Wav2Vec2ForAttentionAlignment.from_pretrained(
                model_name,
                gradient_checkpointing=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=len(processor.tokenizer.decoder)
            )
        # freeze convolutional layers and set the stride of the last conv layer to 1
        # this increase the sampling frequency to 98 Hz
        model.wav2vec2.feature_extractor.conv_layers[6].conv.stride = (1,)
        model.config.conv_stride[-1] = 1
        model.freeze_feature_extractor()

        # Data collator
        # calculate largest length file for padding
        max_audio_len = max(child_speech_dataset['audio_len'])
        MAXIMUM_FEATURE_LEN = model._get_feat_extract_output_lengths(max_audio_len).item()
        data_collator = SpeechCollatorWithPadding(processor=processor, padding=True, max_framelen=MAXIMUM_FEATURE_LEN)

        # training settings
        training_args = TrainingArguments(
            output_dir=output_dir,
            group_by_length=True,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            evaluation_strategy="steps",
            num_train_epochs=2,
            fp16=True,
            save_steps=500,
            eval_steps=10,
            logging_steps=2,
            learning_rate=3e-4,
            weight_decay=0.0001,
            warmup_steps=1000,
            save_total_limit=2,
        )
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            # compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=loso_dataset,
            tokenizer=processor.feature_extractor,
        )


        '''
        calculate alignment accuracy
        '''
        loso_manual_textgrids = [tgpath for tgpath in unmatched_manual_textgrid_files
                                 if loso_speaker_id in tgpath]
        loso_estimated_textgrids = [os.path.join(results_dir, tgpath.split('/')[-2], tgpath.split('/')[-1])
                                    for tgpath in unmatched_manual_textgrid_files if loso_speaker_id in tgpath]

        ''' Evaluate pretrained results'''
        charsiu = charsiu_attention_aligner(model_name)
        write_textgrid_alignments_for_speaker(charsiu_aligner=charsiu, loso_dataset=loso_dataset, output_dir=results_dir)
        del charsiu
        pt_acc, pt_numcorrect, pt_numpredicted = calc_accuracy_between_textgrid_lists(loso_manual_textgrids, loso_estimated_textgrids)
        rslts_pt = trainer.evaluate(loso_dataset)
        print(rslts_pt)
        speaker_wise_results.loc[loso_speaker_id, 'NumPhonesPredicted_PT'] = pt_numpredicted
        speaker_wise_results.loc[loso_speaker_id, 'NumCorrectPhones_PT'] = pt_numcorrect
        speaker_wise_results.loc[loso_speaker_id, 'AlignmentAcc_PT'] = pt_acc
        # speaker_wise_results.loc[loso_speaker_id, 'FramewisePER_PT'] = rslts_pt['eval_phone_accuracy']
        # speaker_wise_results.loc[loso_speaker_id, 'EvalLoss_PT'] = rslts_pt['eval_loss']

        trainer.train()


        '''evaluate finetuned results'''
        # speaker_wise_results[unique_speakers[0]]
        charsiu = charsiu_attention_aligner(model_name)
        charsiu.aligner = model
        write_textgrid_alignments_for_speaker(charsiu_aligner=charsiu, loso_dataset=loso_dataset,
                                              output_dir=results_dir)
        ft_acc, ft_numcorrect, ft_numpredicted = calc_accuracy_between_textgrid_lists(loso_manual_textgrids, loso_estimated_textgrids)
        rslts_ft = trainer.evaluate(loso_dataset)
        speaker_wise_results.loc[loso_speaker_id, 'NumPhonesPredicted_FT'] = ft_numcorrect
        speaker_wise_results.loc[loso_speaker_id, 'NumCorrectPhones_FT'] = ft_numcorrect
        speaker_wise_results.loc[loso_speaker_id, 'AlignmentAcc_FT'] = ft_acc
        # speaker_wise_results.loc[loso_speaker_id, 'FramewisePER_FT'] = rslts_ft['eval_phone_accuracy']
        # speaker_wise_results.loc[loso_speaker_id, 'EvalLoss_FT'] = rslts_ft['eval_loss']
        # speaker_wise_results.loc[loso_speaker_id, 'NumFiles'] = len(loso_dataset)

        print('\n\n+++++++++++++++++ Speaker %s (%d/%d)+++++++++++++++++' % (loso_speaker_id, jj, len(unique_speakers)))
        print('Accuracy Pretrained:\t', pt_acc)
        print('Accuracy Finetuned:\t', ft_acc)

        print('Pretrained PER:\t', rslts_pt['eval_phone_accuracy'])
        print('Finetuned PER:\t',  rslts_ft['eval_phone_accuracy'])

        print('Pretrained Eval Loss:\t', rslts_pt['eval_loss'])
        print('Fineuned Eval Loss:\t', rslts_ft['eval_loss'])
        speaker_wise_results.to_csv(os.path.join(results_dir, 'tmp_results.csv'))

    speaker_wise_results.to_csv(os.path.join(results_dir, 'results_final.csv'))

