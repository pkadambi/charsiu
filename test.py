import pandas as pd
import numpy as np
import os
from alignment_helper_fns import textgridpath_to_phonedf, get_all_textgrids_in_directory


df = pd.read_csv('./phone_annotations_prad.csv')
phonation_dataset_path = '/home/prad/datasets/phonation_data'
df = df[~pd.isna(df['start_time_groundtruth'])]
estimated_tgs_path = '/home/prad/github/charsiu/results/phonation_nopause_trained_frame'
manual_tgs_path = '/home/prad/datasets/phonation_data/extracted_manual_textgrids'
hold_out_inds = np.argwhere(df['RatePhonationStepPauseCount'].values!=0).ravel()

df = df.reset_index()
df = df.iloc[hold_out_inds, :]
df = df[df['PradPause']==0]
df = df[pd.isna(df['PradFlag'])]
df = df.reset_index()

# estimated_tgs = get_all_textgrids_in_directory('./results/phonation_baseline_frame')
tg_filenames = [sessid+'.TextGrid' for sessid in list(df['sessionStepId'].values)]
estimated_tgs = [os.path.join(estimated_tgs_path, tgfname) for tgfname in tg_filenames]
manual_tgs = [os.path.join(manual_tgs_path, tgfname) for tgfname in tg_filenames]
# manual_phone_dfs = np.array([textgridpath_to_phonedf(tgpath, phone_key='phones', replace_silence=False) for tgpath in manual_tgs])
# estimated_phone_dfs = np.array([textgridpath_to_phonedf(tgpath, phone_key='phones', replace_silence=False) for tgpath in estimated_tgs])
audio_files = [os.path.join(phonation_dataset_path, tgfile.split('/')[-1].split('.')[0]+'.wav') for tgfile in tg_filenames]
import traceback

manual_phone_dfs = []
estimated_phone_dfs = []
for ii in range(len(manual_tgs)):

    manual_tg = manual_tgs[ii]
    estimated_tg = estimated_tgs[ii]

    try:
        manual_phone_dfs.append(textgridpath_to_phonedf(manual_tg, phone_key='phones', replace_silence=False))
        estimated_phone_dfs.append(textgridpath_to_phonedf(estimated_tg, phone_key='phones', replace_silence=False))
    except Exception:
        print('-------------------------------------------------')
        print('Error with tg files:\t', manual_tg, estimated_tg)
        print('Audio:\t')
        print("They're probably empty")
        traceback.print_exc()

mystart_times = df['PradStart'].values
yanstart_times = df['start_time_groundtruth'].values
estim_start_times = np.array([edf.iloc[1,0] for edf in estimated_phone_dfs])

myendtimes = df['PradEnd'].values
yanendtimes = df['end_time_groundtruth'].values
estim_end_times = np.array([edf.iloc[2,0] for edf in estimated_phone_dfs])
audio_paths = [os.path.join(phonation_dataset_path, tgname.split('/')[-1][:-8]+'wav') for tgname in estimated_tgs]


'''
Calc inter-rater reliability (on hold out data)

'''

from phonation_dataset import *
tg_dir = os.path.join(phonation_dataset_path, 'extracted_manual_textgrids')
PhonationDataset(audio_paths=audio_paths, lables_df=df, textgrids_dir=tg_dir)



