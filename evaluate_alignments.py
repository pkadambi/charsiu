import transformers
import soundfile as sf
import os
import pandas as pd
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from src.models import Wav2Vec2ForFrameClassificationSAT
from alignment_helper_fns import *
from src.Charsiu import charsiu_sat_forced_aligner, charsiu_forced_aligner
import torch

EXCLUDE_FILES = ['0505_M_EKs4T10', '0411_M_LMwT32']

def print_manual_xvold_xvnew(audiofilepath):
    fname = audiofilepath.split('/')[-1].split('.')[0]+'.TextGrid'
    speakerid = audiofilepath.split('/')[-2]
    manual = textgridpath_to_phonedf(os.path.join(manual_textgrids_dir, speakerid, fname), 'ha phones')
    xvold = textgridpath_to_phonedf(os.path.join('./results_sat_xvector', speakerid, fname), 'phones')
    xvnew = textgridpath_to_phonedf(os.path.join(output_path, speakerid, fname), 'phones')
    print('------------------------')
    print('MANUAL')
    print(manual)
    print('------------------------')
    print('XVOLD')
    print(xvold)
    print('------------------------')
    print('XVNEW')
    print(xvnew)

def get_all_audiofiles_in_dir(audio_dir):
    audiofiles = []
    for root, dirs, files in os.walk(audio_dir):
        for fname in files:
            if '.wav' in fname:
                _fpath = os.path.join(root, fname)
                audiofiles.append(_fpath)
    return audiofiles

def extract_satvectors(satvectorcsv, audiofile):
    relevant_satvectors = satvectorcsv[satvectorcsv.index==audiofile]
    return satvectorcsv[satvectorcsv.index==audiofile].values[0]

def get_transcripts_for_audiofiles(audiofiles):
    transcripts = {}
    for filename in audiofiles:
        fname = filename.split('/')[-1]
        # speaker_dir = filename.split('/')[-2]
        # print(fname)
        # fname = os.path.join(manual_textgrids_dir, speaker_dir, fname[:-8]+'lab')
        fname = filename[:-3] + 'lab'
        # print(fname)
        # break
        f = open(fname)
        _transcript = f.read()
        # print(_transcript)
        transcripts[filename] = _transcript[:-1]
    return transcripts


def load_aligner_from_modelpath(modelpath, tokenizer_name='charsiu/tokenizer_en_cmu'):
    charsiu = charsiu_sat_forced_aligner(model_path, ixvector_size=SATVECTOR_SIZE)
    # charsiu = charsiu_forced_aligner(model_path, ixvector_size=SATVECTOR_SIZE)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_name)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                             do_normalize=True,
                                             return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForFrameClassificationSAT.from_pretrained(modelpath, pad_token_id=processor.tokenizer.pad_token_id,
                                                              vocab_size=len(processor.tokenizer.decoder),
                                                              ivector_size=SATVECTOR_SIZE).cuda()
    charsiu.aligner = model
    return charsiu


def is_sil_or_unk(phone):
    silunk = ['sil', '[SIL]', '[UNK]', '<UNK>']
    return any([su in phone for su in silunk])


def extract_phones_from_textgrid(tg):
    phones = list(tg.phone.values)
    phones_no_numbers = []
    for _phone in phones:
        # _phone =  re.sub(r'[0-9]+', '', _phone)
        if 'sil' not in _phone:
            phones_no_numbers.append(_phone)
    return phones_no_numbers


def get_phoneseqs_from_textgridpaths(textgridpaths):
    phone_seqs = []
    for tgpth in textgridpaths:
        tg = textgridpath_to_phonedf(tgpth, phone_key='ha phones')
        _phoneseq = extract_phones_from_textgrid(tg)
        phone_seqs.append(tuple(_phoneseq))
    return phone_seqs


def run_aligner_on_files(audiopaths, transcripts, output_dir, satvectorcsv=None, gt_phoneme_sequences=None):
    aligner = load_aligner_from_modelpath(model_path)
    print('Evaluating Aligner...')
    for ii in tqdm.tqdm(range(len(audiopaths))):
        audiofilepath = audiopaths[ii]
        transcript = transcripts[audiofilepath]
        if satvectorcsv is not None:
            ixvec = torch.tensor(extract_satvectors(satvectorcsv, audiofilepath)).cuda().float()

        _fname = audiofilepath.split('/')[-1].split('.')[-2]
        output_tg = os.path.join(output_dir, _fname + '.TextGrid')
        audio = librosa.load(audiofilepath, sr=16000)[0]

        if gt_phoneme_sequences is not None:
            target_phones = gt_phoneme_sequences[ii]
            try:
                aligner.serve(audio, ixvector=ixvec, text=transcript, target_phones=[target_phones], save_to=output_tg)
            except:
                print('ERROR: Using ground truth target phonemes, using G2P.')
                aligner.serve(audio, ixvector=ixvec, text=transcript, save_to=output_tg)
            # aligner.serve(audiofilepath, text=transcripts[ii], target_phones=target_phones, save_to=output_tg)
        else:
            # aligner.serve(audiofilepath, text=transcripts[ii], save_to=output_tg)
            aligner.serve(audio, ixvector=ixvec, text=transcript, save_to=output_tg)

        print()
''' define directories '''
audio_dir = '/home/prad/datasets/ChildSpeechDataset/child_speech_16_khz'
manual_textgrids_dir = '/home/prad/datasets/ChildSpeechDataset/manually-aligned-text-grids/'
mfa_sat_dir = '/home/prad/datasets/ChildSpeechDataset/mfa_adapted'
output_path = './results_xvector_reevaluated'


EXCLUDE_FILES = ['0505_M_EKs4T10', '0411_M_LMwT32']

satvectors_csv = pd.read_csv('./extracted_xvectors_proj_libri.csv', index_col='Filename')

audiofiles = get_all_audiofiles_in_dir(audio_dir)
audiofiles = [filename for filename in audiofiles if not any([exf in filename for exf in EXCLUDE_FILES])]

manual_textgrids = get_all_textgrids_in_directory(manual_textgrids_dir)
manual_textgrids = [filename for filename in manual_textgrids if not any([exf in filename for exf in EXCLUDE_FILES])]

SATVECTOR_SIZE=128

speaker_ids = [dirs for root, dirs, files in os.walk(audio_dir)][0]
for kk, speaker_id in enumerate(speaker_ids):
    print(f"Ruinning speaker {speaker_id}, {kk + 1}/{len(speaker_ids)}")
    model_path = os.path.join('./sat_xvector_proj_models/' + speaker_id)
    speaker_audiofiles = [audfile for audfile in audiofiles if speaker_id in audfile]
    speaker_manual_textgridfiles = []
    for audfile in speaker_audiofiles:
        _filename = audfile.split('/')[-1].split('.')[0]
        _tgfile = os.path.join(manual_textgrids_dir, speaker_id, _filename+'.TextGrid')
        speaker_manual_textgridfiles.append(_tgfile)
    # speaker_manual_textgridfiles = [tgfile for tgfile in manual_textgrids if speaker_id in tgfile]
    speaker_transcripts = get_transcripts_for_audiofiles(audiofiles)

    speaker_textgrids_outputdir = os.path.join(output_path, speaker_id)
    os.makedirs(speaker_textgrids_outputdir, exist_ok=True)
    gtphone_seqs = get_phoneseqs_from_textgridpaths(speaker_manual_textgridfiles)

    run_aligner_on_files(speaker_audiofiles, speaker_transcripts, speaker_textgrids_outputdir,
                         satvectorcsv=satvectors_csv, gt_phoneme_sequences=None)
    # break


