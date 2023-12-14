import os

import scipy.stats
import tqdm
import copy
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from alignment_helper_fns import *
from gop_helper_fns import *
from evaluation_utils import *
from mfa_evaluation_utils import *
from g2p_en import G2p
from phoneme_info import *

g2p = G2p()

satdf = pd.read_csv('./results_sat/results.csv').set_index('Unnamed: 0')
xsatdf = pd.read_csv('./results_sat_xvector/results_xvector_proj.csv')
xsatdf = xsatdf.rename(columns={'Unnamed: 0': 'Speaker'})


results_dir_frame = './results_frame_10epochs'
results_dir_ivec = './results_sat'
# results_dir_xvec = './results_sat_xvector'
results_dir_xvec = './phone_matched_xvec_proj_textgrids'

speaker_tgs = {}

EXCLUDE_FILES = ['0505_M_EKs4T10', '0411_M_LMwT32']

allmanual_tgs = [pth for pth in get_all_textgrids_in_directory('/media/prad/data/datasets/ChildSpeechDataset/manually-aligned-text-grids/') if '.TextGrid' in pth]
allmanual_tgs = [tg for tg in allmanual_tgs if all([_excludefile not in tg for _excludefile in EXCLUDE_FILES])]

ivector_tgs =[pth for pth in get_all_textgrids_in_directory('./results_sat') if '.TextGrid' in pth]
ivector_tgs = [tg for tg in ivector_tgs if all([_excludefile not in tg for _excludefile in EXCLUDE_FILES])]


xvector_tgs =[pth for pth in get_all_textgrids_in_directory('./phone_matched_xvec_proj_textgrids') if '.TextGrid' in pth]
# xvector_tgs =[pth for pth in get_all_textgrids_in_directory('./results_xvector_reevaluated') if '.TextGrid' in pth]
xvector_tgs = [tg for tg in xvector_tgs if all([_excludefile not in tg for _excludefile in EXCLUDE_FILES])]

mfa_tgs = [pth for pth in get_all_textgrids_in_directory('./results_mfa_adapted') if '.TextGrid' in pth]
mfa_tgs = [tg for tg in mfa_tgs if all([_excludefile not in tg for _excludefile in EXCLUDE_FILES])]

print('Num Mismatched:\t', sum([f1.split('/')[-1].split('.')[0]!=f2.split('/')[-1].split('.')[0] for f1, f2 in zip(allmanual_tgs, xvector_tgs)]))

print(len(allmanual_tgs))
print(len(ivector_tgs))
print(len(xvector_tgs))
print(len(mfa_tgs))

# if not (os.path.exists('./outputs/gt_dfs.pkl') and os.path.exists('./outputs/phone_accuracy_dfs.pkl')):

methodnames = ['xvector', 'mfa_train', 'mfa_base', 'frame', 'ivector']
# methodnames = ['mfa_train', 'mfa_base']
phone_durations = {}
phone_onset_err = {}
phone_offset_err = {}
phone_ids = {}
age = {}
nexclude_per = {}
phone_dfs = {}

for method in methodnames:
    phone_dfs[method] = {}

finish = False
gt_dfs = {}

for phone in tqdm.tqdm(ENGLISH_PHONEME_LIST):
    print('Calculating error for phone:\t', phone)
    durations_gt = {}
    durations_est = {}
    onset_err = {}
    offset_err = {}

    onset_err_pct = {}
    offset_err_pct = {}

    nfiles_error = {}

    phone_dfs[phone] = {}

    for method in methodnames:
        phone_ids[method] = []
        durations_est[method] = []
        durations_gt[method] = []

        onset_err[method] = []
        offset_err[method] = []

        onset_err_pct[method] = []
        offset_err_pct[method] = []

        nfiles_error[method] = 0

        start_gt = []
        end_gt = []
        speaker_id_list_manual = []
        speaker_id_list_method = []

        for speakerid in list(satdf.index):

            tgs_manual = get_all_textgrids_in_directory(
                os.path.join('/media/prad/data/datasets/ChildSpeechDataset/manually-aligned-text-grids/', speakerid),
                verbose=False)
            if method == 'frame':
                tgs_method = get_all_textgrids_in_directory(os.path.join('./results_frame_10epochs', speakerid),
                                                            verbose=False)
            elif method == 'xvector':
                tgs_method = get_all_textgrids_in_directory(
                    os.path.join('./phone_matched_xvec_proj_textgrids', speakerid), verbose=False)
            elif method == 'ivector':
                tgs_method = get_all_textgrids_in_directory(os.path.join('./results_sat', speakerid), verbose=False)
            elif method == 'mfa_train':
                tgs_method = get_all_textgrids_in_directory(os.path.join('./results_mfa_trained', speakerid),
                                                            verbose=False)
            elif method == 'mfa_adapt':
                tgs_method = get_all_textgrids_in_directory(os.path.join('./results_mfa_adapted', speakerid),
                                                            verbose=False)
            elif method == 'mfa_base':
                tgs_method = get_all_textgrids_in_directory(os.path.join('./results_mfa_adapted_english_us_arpa', speakerid),
                                                            verbose=False)
            elif method == 'mfa_adapt_from_trained':
                tgs_method = get_all_textgrids_in_directory(os.path.join('./results_mfa_adapted_from_trained_sat3', speakerid),
                                                            verbose=False)
            for manual_tgpath in tgs_manual:
                manual_tg = textgridpath_to_phonedf(manual_tgpath, phone_key='ha phones', remove_numbers=True)
                gtstart, gtend = get_phone_startend(manual_tg, phone, loc=2)
                durations_gt[method].extend(list((gtend - gtstart) * 1000))
                speaker_id_list_manual.extend([speakerid] * len(gtstart))
                start_gt.extend(list(gtstart * 1000))
                end_gt.extend(list(gtend * 1000))

            est_dur, on_err, on_err_pct, off_err, off_err_pct, nerr = evaluate_tg_results(method, phone, tgs_method,
                                                                                          tgs_manual, durations_est,
                                                                                          onset_err, offset_err)
            onset_err[method].extend(on_err)
            offset_err[method].extend(off_err)
            onset_err_pct[method].extend(on_err_pct)
            offset_err_pct[method].extend(off_err_pct)
            durations_est[method].extend(est_dur)
            nfiles_error[method] += nerr
            speaker_id_list_method.extend([speakerid] * len(est_dur))
        df_gt = pd.DataFrame.from_dict(
            {'phone': [phone] * len(durations_gt[method]), 'speakerid': speaker_id_list_manual,
             'start_time': start_gt, 'end_time': end_gt})
        df_method = pd.DataFrame.from_dict(
            {'phone': [phone] * len(durations_est[method]), 'speakerid': speaker_id_list_method,
             f'durations_est_{method}': durations_est[method],
             f'onset_err_{method}': onset_err[method],
             f'offset_err_{method}': offset_err[method],
             f'onset_err_pct_{method}': onset_err_pct[method],
             f'offset_err_pct_{method}': offset_err_pct[method]})

        gt_dfs[phone] = df_gt
        phone_dfs[phone][method] = df_method

print()

pkl.dump(phone_dfs, open('./outputs/phone_accuracy_dfs_corrected_mfa_train_mfa_base_ONLY.pkl', 'wb'))


