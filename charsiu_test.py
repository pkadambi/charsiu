import os
import numpy as np
from alignment_helper_fns import *
from audio_utils import *

audio_dir = '/home/prad/datasets/ChildSpeechDataset/child_speech_16_khz'
manual_textgrids_dir = '/home/prad/datasets/ChildSpeechDataset/manually-aligned-text-grids/'
mfa_sat_dir = '/home/prad/datasets/ChildSpeechDataset/mfa_adapted/'
# mfa_sat_dir = '/home/prad/datasets/ChildSpeechDataset/mfa_with_sat/'

unmatched_manual_textgrid_files = get_all_textgrids_in_directory(manual_textgrids_dir)
aligner_textgrid_files = get_all_textgrids_in_directory(mfa_sat_dir)
candidate_aligner_textgrid_files = get_all_textgrids_in_directory(mfa_sat_dir)
manual_textgrid_files = []
audio_files = ['/'.join([audio_dir,_path.split('/')[-2], _path.split('/')[-1][:-8]+'wav'])
               for _path in unmatched_manual_textgrid_files]

from create_child_speech_dataset import *
# csd = ChildSpeechDataset(audio_paths=audio_files)

'''
need to match the corresponding files since the files loaded from the code above are out of order
'''
mismatched_phoneme = 0
mismatched_lengths = 0
print(len(aligner_textgrid_files))
for ii, aligned_tg_file in tqdm.tqdm(enumerate(aligner_textgrid_files)):
#     print(ii)
    _filename = aligned_tg_file.split('/')[-1].replace('-', '_')
#     print(_filename)
    matching_ind = int(np.argwhere([_filename in manual_file for manual_file in unmatched_manual_textgrid_files]).ravel())
#     print(matching_ind)
    matching_gt_file = unmatched_manual_textgrid_files[matching_ind]
    manual_textgrid_files.append(matching_gt_file)

transcripts = {}

''' Extract transcripts for w2v2 aligner'''
transcripts = {}
for filename in audio_files:
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
    # break

''' Charsiu setup'''
from src.Charsiu import charsiu_forced_aligner, charsiu_attention_aligner
# charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')
from alignment_helper_fns import *
# charsiu = charsiu_forced_aligner('charsiu/en_w2v2_fc_10ms')
charsiu = charsiu_attention_aligner('charsiu/en_w2v2_fs_10ms')

charisu_tg_files = []
error_files = []
do_align = False
if do_align:
    for ii, tgfilepath in tqdm.tqdm(enumerate(manual_textgrid_files)):
        audiofname = tgfilepath.split('/')[-1][:-8] + 'wav'
        speaker_dir = tgfilepath.split('/')[-2]
        tgfilename = tgfilepath.split('/')[-1]
        # audio_path = os.path.join(audio_dir, speaker_dir, audiofname)
        audio_path = audio_files[ii]

        output_tg_dir = os.path.join('results/charsiu_w2v2_attention_aligner_10ms/', speaker_dir)
        if not os.path.isdir(output_tg_dir):
            os.makedirs(output_tg_dir, exist_ok=True)

        tg_filepath = os.path.join(output_tg_dir, tgfilename)
        _transcript = transcripts[audio_path]

        charisu_tg_files.append(tg_filepath)
        # print('************************************************************************')
        # print(audio_path)
        # print(tgfilename)
        # print(_transcript)
        # print(textgridpath_to_phonedf(tgfilepath, phone_key='ha phones'))

        # print(tg_filepath)
        try:
            #pd.DataFrame(charsiu.align(audio=audio_path,text=_transcript)[0])
            charsiu.serve(audio = audio_path, text = _transcript, save_to = tg_filepath)

        except Exception:
            print('Could not perform alignment for file:\t ', audio_path)
            error_files.append(tgfilepath)
estimated_textgrids = [os.path.join('results/charsiu_w2v2_attention_aligner_10ms/', path.split('/')[-2], path.split('/')[-1]) for path in manual_textgrid_files]
from alignment_helper_fns import calc_accuracy_between_textgrid_lists
calc_accuracy_between_textgrid_lists(manual_textgrid_files, estimated_textgrids)





