import os
import numpy as np
from alignment_helper_fns import *
from audio_utils import *

audio_dir = '/home/prad/datasets/phonation_data'
output_tg_dir = 'results/phonation_baseline_frame'

'''
extract files with no pauses
'''
df = pd.read_csv('./phonation_data.csv')
pause_count_key = 'RatePhonationStepPauseCount'
end_time_key = 'RatePhonationStepSpeechEndTimeNoAmbient'
start_time_key = 'RatePhonationStepSpeechStartTime'
nopause_files = df[df[pause_count_key]==0]['sessionStepId'].str[:] + '.wav'
all_audio_files = df['sessionStepId'].str[:] + '.wav'
audio_files = all_audio_files
print('Will Evaluate aligner on files', nopause_files)
longest_seg_start = []
longest_seg_stop = []

''' Charsiu setup'''
from src.Charsiu import charsiu_forced_aligner, charsiu_attention_aligner
# charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')
from alignment_helper_fns import *
charsiu = charsiu_forced_aligner('charsiu/en_w2v2_fc_10ms')
import torch
charsiu.aligner.load_state_dict(torch.load('./models_test/phonation_model'))
# charsiu = charsiu_attention_aligner('charsiu/en_w2v2_fs_10ms')

charisu_tg_files = []
error_files = []
do_align = True
TRANSCRIPT = 'AH'
if do_align:
    print('************************************************************************')
    print('Doing alignment...')
    for ii, audio_file in tqdm.tqdm(enumerate(audio_files)):
        audio_filepath = os.path.join(audio_dir, audio_file)
        # output_tg_dir = os.path.join('results/charsiu_w2v2_attention_aligner_10ms/', speaker_dir)
        tg_filename = audio_filepath.split('/')[-1][:-3] + 'TextGrid'
        if not os.path.isdir(output_tg_dir):
            os.makedirs(output_tg_dir, exist_ok=True)

        tg_filepath = os.path.join(output_tg_dir, tg_filename)
        # _transcript = transcripts[audio_path]

        charisu_tg_files.append(tg_filepath)
        # print('************************************************************************')
        # print(audio_path)
        # print(tgfilename)
        # print(_transcript)
        # print(textgridpath_to_phonedf(tgfilepath, phone_key='ha phones'))

        # print(tg_filepath)
        try:
            # pd.DataFrame(charsiu.align(audio=audio_path,text=_transcript)[0])
            charsiu.serve(audio = audio_filepath, text = TRANSCRIPT, save_to = tg_filepath)

        except Exception:
            print('Could not perform alignment for file:\t ', audio_filepath)
            error_files.append(tg_filepath)
# estimated_textgrids = [os.path.join('results/phonation_audio/', path.split('/')[-2], path.split('/')[-1]) for path in manual_textgrid_files]
from alignment_helper_fns import calc_accuracy_between_textgrid_lists
# calc_accuracy_between_textgrid_lists(manual_textgrid_files, estimated_textgrids)





