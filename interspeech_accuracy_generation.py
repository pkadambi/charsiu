import os
import tqdm
import copy
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from alignment_helper_fns import *
from gop_helper_fns import *
from mfa_evaluation_utils import *
from g2p_en import G2p
from evaluation_utils import *

def calc_acc_between_tg_lists(manual_tg_list, estimated_tg_list, collapse_shortphones: bool,  manual_phonekey='ha phones', aligner_phonekey='phones',
                             ignore_numbers=True, ignore_extras=True, ignore_silence=False, verbose=True):


    correct_indicator = []
    correct_indicator_matched = []
    matched_indicator = []
    matched_phones = []
    allphonelist = []
    nmismatch = 0
    nlendiff = 0
    nconsecutive = 0
    all_transcripts = []
    all_timingdfs = []
    for ii, (manual_textgridpath, estimated_textgridpath) in tqdm.tqdm(enumerate(zip(manual_tg_list, estimated_tg_list))):
        # try:

        manualdf = textgridpath_to_phonedf(manual_textgridpath, phone_key=manual_phonekey, remove_numbers=ignore_numbers)
        alignerdf = textgridpath_to_phonedf(estimated_textgridpath, phone_key=aligner_phonekey, remove_numbers=ignore_numbers,
                                    replace_silence=True)
        transcript = get_transcript_from_tgfile(estimated_textgridpath)
        manual_phones = np.array(remove_sil_from_phonelist(manualdf.phone.values))

        aligner_phones = np.array(remove_sil_from_phonelist(alignerdf.phone.values))
        _ismatch = contains_same_phones(manual_phones, aligner_phones)

        # _alignerdf = process_silences(alignerdf, transcript)
        _alignerdf = alignerdf.copy()
        _alignerphones = list(_alignerdf.phone.values)
        allphonelist.extend(list(_alignerdf.phone.values))
        # matched_indicator.append(ismatch)
        palignerdf = process_silences(alignerdf, transcript)
        speakerid = manual_textgridpath.split('/')[-2]
        all_transcripts.append(transcript)
        assert manual_textgridpath.split('/')[-1]==estimated_textgridpath.split('/')[-1], \
            f'Comparison Error! Comparing different textgrids. \nManualpath {manual_textgridpath}, \nEstimatedPath {estimated_textgridpath}'
        _correct_indicator, ismatch, timingdf = \
            calc_alignment_accuracy_between_textgrids(manual_textgridpath = manual_textgridpath,
                                                                      aligner_textgridpath = estimated_textgridpath,
                                                                      manual_phonekey=manual_phonekey,
                                                                      aligner_phonekey=aligner_phonekey,
                                                                      ignore_extras=ignore_extras,
                                                                      collapse_shortphones=collapse_shortphones)

        fileid = manual_textgridpath.split('/')[-1].split('.')[0]
        timingdf['Speaker'] = speakerid
        timingdf['File'] = fileid
        __correct_indicator = np.copy(_correct_indicator)
        correct_indicator.append(_correct_indicator)
        running_acc = np.mean(np.concatenate(correct_indicator))
        all_timingdfs.append(timingdf)

        if len(np.array(remove_sil_from_phonelist(_alignerphones)))  != len(manual_phones):

            if any([manual_phones[jj]==manual_phones[jj+1] for jj in range(len(manual_phones)-1)]):
                nconsecutive+=1

            nlendiff+=1

        elif len(_alignerphones) != len(_correct_indicator):
            print('\n', np.array(remove_sil_from_phonelist(_alignerphones)), '\n', manual_phones, '\n', transcript, '\n', _correct_indicator)

            print('')
            nmismatch +=1


        else:
            matched_indicator.extend(_correct_indicator)
            matched_phones.extend(_alignerphones)
            correct_indicator_matched.append(_correct_indicator)

            # correct_indicator_matched.append(_correct_indicator)

            # running_acc_matched = np.mean(np.concatenate(correct_indicator_matched))

        # if ii>25:
            # break
        # print(manualdf)
        # print(alignerdf)
        # print(correct_indicator)
        # except:
            # print('error')
            # if not os.path.exists(estimated_textgridpath):
                # print('Textgrid file', estimated_textgridpath, 'not found, skipping this file')
    # print(correct_indicator)

    # return 0
    correct_indicator = np.concatenate(correct_indicator)
    acc = np.mean(correct_indicator)
    numcorrect = np.sum(correct_indicator)
    numpredicted = len(correct_indicator)

    correct_indicator_matched = np.concatenate(correct_indicator_matched)
    acc_matched = np.mean(correct_indicator_matched)
    numcorrect_matched = np.sum(correct_indicator_matched)
    numpredicted_matched = len(correct_indicator_matched)

    if verbose:
        print('============ Total Accuracy ============')
        print('Accuracy:\t', np.mean(correct_indicator))
        print('Num Correct:\t', numcorrect)
        print('Num Predicted Phones:\t', numpredicted)
        print('============ Matched Accuracy ============')
        print('Matched Accuracy:\t', np.mean(correct_indicator_matched))
        print('Num Correct Matched:\t', numcorrect_matched)
        print('Num Predicted Phones Matched:\t', numpredicted_matched)

    return acc, acc_matched, numcorrect, numcorrect_matched, numpredicted, numpredicted_matched, \
        correct_indicator, allphonelist, all_timingdfs





g2p = G2p()
satdf = pd.read_csv('./results_sat/results.csv').set_index('Unnamed: 0')

ALL_SPEAKERS = list(satdf.index)

results_dir_frame = './results_frame_10epochs'
results_dir_ivec = './results_sat'
# results_dir_xvec = './results_sat_xvector'
results_dir_xvec = './phone_matched_xvec_proj_textgrids'

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

method_to_dir = {

    'MFA_TrainNew': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_trained/',

    'xVector': './phone_matched_xvec_proj_textgrids',

    'W2V2Base': '/home/prad/github/DNN-PLLR/child_results_no_train/',
    'xvectorNewNoLDA': '/home/prad/github/DNN-PLLR/results_sat_xvector_nolda/',
    'xvectorNewWithLDA':'/home/prad/github/DNN-PLLR/results_sat_xvector_librilda/',
    'MFA_AlignOld': './results_mfa_base',
    'MFA_AlignNew': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_aligned/',
    'MFA_AdaptNew': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_adapted_alignments/',
    'MFA_Align_noSAT': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_noSAT/',
    'MFA_TrainOld':'./results_mfa_trained',
    'iVector': './results_sat',
    'W2V2TrainedOld': './results_frame_10epochs',
    'W2V2TrainedNew': '/home/prad/github/DNN-PLLR/textgrids/frame/',
    'MFA_noSAT': './results_mfa_trained',
    'gt': '/media/prad/data/datasets/ChildSpeechDataset/manually-aligned-text-grids/',
    'interrater': '/media/prad/data/datasets/ChildSpeechDataset/interrater-check/'}


speakers = ALL_SPEAKERS
collapse_shortphones=True
acc_results_dct = {}


'''
================================================================================
RUN PARAMETERS
================================================================================

'''
rerun = False
overwrite_results = True
'''
Generate aggregate accuracy across all speakers
'''
aggregate_acc_csv = './interspeech_results/acc_resultsJSLHR.csv'
timing_dfs = {}

method_to_dir = {

    'MFA_TrainNew': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_trained/',

    'xVector': './phone_matched_xvec_proj_textgrids',

    'W2V2Base': '/home/prad/github/DNN-PLLR/child_results_no_train/',
    'xvectorNewNoLDA': '/home/prad/github/DNN-PLLR/results_sat_xvector_nolda/',
    'xvectorNewWithLDA':'/home/prad/github/DNN-PLLR/results_sat_xvector_librilda/',
    'MFA_AlignOld': './results_mfa_base',
    'MFA_AlignNew': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_aligned/',
    'MFA_AdaptNew': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_adapted_alignments/',
    'MFA_Align_noSAT': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_noSAT/',
    'MFA_TrainOld':'./results_mfa_trained',
    'iVector': './results_sat',
    'W2V2TrainedOld': './results_frame_10epochs',
    'W2V2TrainedNew': '/home/prad/github/DNN-PLLR/textgrids/frame/',
    'MFA_noSAT': './results_mfa_trained',
    'gt': '/media/prad/data/datasets/ChildSpeechDataset/manually-aligned-text-grids/',
    'interrater': '/media/prad/data/datasets/ChildSpeechDataset/interrater-check/'}


if not os.path.exists(aggregate_acc_csv) or rerun:
    for methodname in method_to_dir.keys():
        if methodname=='gt' or methodname=='interrater':
            continue
        print('-------------------------------------------------------')
        print('Methodname:', methodname)
        correct_indicator_matched = []
        correct_indicator = []

        all_gt_tgs = []
        all_method_tgs = []

        for speakerid in speakers:
            tgs_manualpath = os.path.join(method_to_dir['gt'], speakerid)
            tgs_methodpath = os.path.join(method_to_dir[methodname], speakerid)
            tgs_manual = get_all_textgrids_in_directory(tgs_manualpath, verbose=False)
            tgs_method = get_all_textgrids_in_directory(tgs_methodpath, verbose=False)
            tgs_manual = [get_gt_tgpath(tgs_manual, _tgmethod) for _tgmethod in tgs_method]
            all_gt_tgs.extend(tgs_manual)
            all_method_tgs.extend(tgs_method)

        acc, acc_matched, numcorrect, numcorrect_matched, numpredicted, numpredicted_matched, correct_indicator, allphones, timingdfs = \
            calc_acc_between_tg_lists(all_gt_tgs, all_method_tgs, collapse_shortphones=collapse_shortphones,
                                      manual_phonekey='ha phones', aligner_phonekey='phones')
        acc_results_dct[methodname] = {'Acc': acc,
                                       'NumCorrect':numcorrect,
                                       'NumPredicted':numpredicted,
                                       'AccMatched': acc_matched,
                                       'NumCorrectMatched':numcorrect_matched,
                                       'NumPredictedMatched':numpredicted_matched,
                                       }
        timing_dfs[methodname] = timingdfs
    acc_rslts_df = pd.DataFrame.from_dict(acc_results_dct).T

    if overwrite_results:
        acc_rslts_df.to_csv(aggregate_acc_csv)
        # acc_rslts_df['Timingdfs'] = timingdfs
        pkl.dump(timing_dfs, open('./jslhr_timingdfs.pkl', 'wb'))


'''
SAMPLE SIZE ACC DFS GNEERATION
'''
rerun = False
overwrite_results = True
sample_size_dirs ={0.5: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.5_xvec/run_0/trained_alignments/TestAudio',
 0.1: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.1_xvec/run_0/trained_alignments/TestAudio',
 0.45: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.45_xvec/run_0/trained_alignments/TestAudio',
 0.3: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.3_xvec/run_0/trained_alignments/TestAudio',
 0.15: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.15_xvec/run_0/trained_alignments/TestAudio',
 0.4: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.4_xvec/run_0/trained_alignments/TestAudio',
 0.2: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.2_xvec/run_0/trained_alignments/TestAudio',
 1.0: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction1.0_xvec/run_0/trained_alignments/TestAudio',
 0.05: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.05_xvec/run_0/trained_alignments/TestAudio',
 0.029: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.029_xvec/run_0/trained_alignments/TestAudio',
 0.015: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.015_xvec/run_0/trained_alignments/TestAudio',
 0.35: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.35_xvec/run_0/trained_alignments/TestAudio',
 0.25: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.25_xvec/run_0/trained_alignments/TestAudio',
 0.075: '/home/prad/github/DNN-PLLR/alignment_training/child_interrater_split_fraction0.075_xvec/run_0/trained_alignments/TestAudio',
'gt': '/media/prad/data/datasets/ChildSpeechDataset/manually-aligned-text-grids/',
                   }
aggregate_acc_csv = './interspeech_results/acc_resultsJSLHR_sample_size_acc.csv'
timing_dfs = {}
if not os.path.exists(aggregate_acc_csv) or rerun:
    for methodname in sample_size_dirs.keys():
        if methodname=='gt' or methodname=='interrater':
            continue
        print('-------------------------------------------------------')
        print('Methodname:', methodname)
        correct_indicator_matched = []
        correct_indicator = []

        all_gt_tgs = []
        all_method_tgs = []

        for speakerid in speakers:
            tgs_manualpath = os.path.join(sample_size_dirs['gt'], speakerid)
            tgs_methodpath = os.path.join(sample_size_dirs[methodname], speakerid)
            tgs_manual = get_all_textgrids_in_directory(tgs_manualpath, verbose=False)
            tgs_method = get_all_textgrids_in_directory(tgs_methodpath, verbose=False)
            tgs_manual = [get_gt_tgpath(tgs_manual, _tgmethod) for _tgmethod in tgs_method]
            all_gt_tgs.extend(tgs_manual)
            all_method_tgs.extend(tgs_method)

        acc, acc_matched, numcorrect, numcorrect_matched, numpredicted, numpredicted_matched, correct_indicator, allphones, timingdfs = \
            calc_acc_between_tg_lists(all_gt_tgs, all_method_tgs, collapse_shortphones=collapse_shortphones,
                                      manual_phonekey='ha phones', aligner_phonekey='phones')
        acc_results_dct[methodname] = {'Acc': acc,
                                       'NumCorrect':numcorrect,
                                       'NumPredicted':numpredicted,
                                       'AccMatched': acc_matched,
                                       'NumCorrectMatched':numcorrect_matched,
                                       'NumPredictedMatched':numpredicted_matched,
                                       }
        timing_dfs[methodname] = timingdfs
    acc_rslts_df = pd.DataFrame.from_dict(acc_results_dct).T

    if overwrite_results:
        acc_rslts_df.to_csv(aggregate_acc_csv)
        # acc_rslts_df['Timingdfs'] = timingdfs
        pkl.dump(timing_dfs, open('./jslhr_timingdfs_sample_size_vs_acc.pkl', 'wb'))

'''
Generate Inter-rater results

'''


'''
Generate Inter-rater
'''
timing_dfs = {}

interrater_path = '/media/prad/data/datasets/ChildSpeechDataset/interrater-check/'
manualalign_path = '/media/prad/data/datasets/ChildSpeechDataset/manually-aligned-text-grids'

import glob
method_to_dir = {

    'MFA_TrainNew': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_trained/',

    'xVector': './phone_matched_xvec_proj_textgrids',

    'W2V2Base': '/home/prad/github/DNN-PLLR/child_results_no_train/',
    'xvectorNewNoLDA': '/home/prad/github/DNN-PLLR/results_sat_xvector_nolda/',
    'xvectorNewWithLDA':'/home/prad/github/DNN-PLLR/results_sat_xvector_librilda/',
    'MFA_AlignOld': './results_mfa_base',
    'MFA_AlignNew': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_aligned/',
    'MFA_AdaptNew': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_adapted_alignments/',
    'MFA_Align_noSAT': '/media/prad/data/datasets/ChildSpeechDataset/JSLHR_alignments/mfa_noSAT/',
    'MFA_TrainOld':'./results_mfa_trained',
    'iVector': './results_sat',
    'W2V2TrainedOld': './results_frame_10epochs',
    'W2V2TrainedNew': '/home/prad/github/DNN-PLLR/textgrids/frame/',
    'MFA_noSAT': './results_mfa_trained',
    'gt': '/media/prad/data/datasets/ChildSpeechDataset/manually-aligned-text-grids/',
    'interrater': '/media/prad/data/datasets/ChildSpeechDataset/interrater-check/'}


# for method in method_to_dir.keys():

    # methodpath = method_to_dir[method]

    # for speakerpath in glob.glob(interrater_path + '*'):
    #     speaker = speakerpath.split('/')[-1]

        # human1path = os.path.join(methodpath, speaker)
        # human1_interratertgs.extend([file for file in glob.glob(f'{human1path}/*')])

interrater_speakers = [speakerpath.split('/')[-1] for speakerpath in glob.glob(interrater_path + '*')]

interrater_acc_csvpath = './interspeech_results/acc_results_interraterJSLHR.csv'
if not os.path.exists(interrater_acc_csvpath):
    for methodname in method_to_dir.keys():
        # methodname='xvector'
        if methodname=='gt':
            continue
        print('-------------------------------------------------------')
        print('Methodname:', methodname)
        correct_indicator_matched = []
        correct_indicator = []

        all_gt_tgs = []
        all_method_tgs = []

        for speakerid in interrater_speakers:
            tgs_manualpath = os.path.join(method_to_dir['gt'], speakerid)
            tgs_methodpath = os.path.join(method_to_dir[methodname], speakerid)
            tgs_manual = get_all_textgrids_in_directory(tgs_manualpath, verbose=False)
            tgs_method = get_all_textgrids_in_directory(tgs_methodpath, verbose=False)
            tgs_manual = [get_gt_tgpath(tgs_manual, _tgmethod) for _tgmethod in tgs_method]
            all_gt_tgs.extend(tgs_manual)
            all_method_tgs.extend(tgs_method)


        phonekey = 'ha phones' if methodname=='interrater' else 'phones'
        acc, acc_matched, numcorrect, numcorrect_matched, numpredicted, numpredicted_matched, correct_indicator, allphones, timingdfs  = \
            calc_acc_between_tg_lists(all_gt_tgs, all_method_tgs, collapse_shortphones=collapse_shortphones,
                                      manual_phonekey='ha phones', aligner_phonekey=phonekey)
        acc_results_dct[methodname] = {'Acc': acc,
                                       'NumCorrect':numcorrect,
                                       'NumPredicted':numpredicted,
                                       'AccMatched': acc_matched,
                                       'NumCorrectMatched':numcorrect_matched,
                                       'NumPredictedMatched':numpredicted_matched,
                                       'timingdfs':timingdfs}
        timing_dfs[methodname] = timingdfs


    interrater_acc_df = pd.DataFrame.from_dict(acc_results_dct).T

    if overwrite_results:
        interrater_acc_df.to_csv(interrater_acc_csvpath)
        pkl.dump(timing_dfs, open('./jslhr_timingdfsINTERRATER.pkl', 'wb'))



'''
Generate per-speaker accuracy before and after fine tuning
'''
speaker_acc_results = {}

# usemethods = ['frame', 'ivector', 'xvector']
usemethods = method_to_dir.keys()
# per_speaker_csv = './interspeech_results/acc_results_speakerwise.csv'
per_speaker_csv = './interspeech_results/speakerwise_acc_resultsJSLHR.csv'
phonelist = []
acc_indicator = []
if not os.path.exists(per_speaker_csv) or rerun:
    for speakerid in tqdm.tqdm(speakers):
        print('\n-------------------------------------------')
        print(f'Running Speaker {speakerid}')
        results = {}
        for methodname in usemethods:
            if methodname=='gt':
                continue
            tgs_manualpath = os.path.join(method_to_dir['gt'], speakerid)
            tgs_methodpath = os.path.join(method_to_dir[methodname], speakerid)
            tgs_manual = get_all_textgrids_in_directory(tgs_manualpath, verbose=False)
            tgs_method = get_all_textgrids_in_directory(tgs_methodpath, verbose=False)
            tgs_manual = [get_gt_tgpath(tgs_manual, _tgmethod) for _tgmethod in tgs_method]

            #fix error in line below
            acc, acc_matched, numcorrect, numcorrect_matched, numpredicted, numpredicted_matched, correct_indicator, allphones = \
                calc_acc_between_tg_lists(manual_tg_list = tgs_manual, estimated_tg_list = tgs_method,
                                          collapse_shortphones = collapse_shortphones,  manual_phonekey='ha phones',
                                          aligner_phonekey='phones', ignore_numbers=True, ignore_extras=True,
                                          ignore_silence=False, verbose=False)
            # phonelist


            results[f'Acc_{methodname}'] = acc
            results[f'NumCorrect_{methodname}'] = numcorrect,
            results[f'NumPredicted_{methodname}'] = numpredicted,
            results[f'AccMatched_{methodname}'] = acc_matched,
            results[f'NumCorrectMatched_{methodname}'] = numcorrect_matched,
            results[f'NumPredictedMatched_{methodname}'] = numpredicted_matched

        age = float(speakerid[:2]) + float(speakerid[2:4])/12
        results['Age'] = age
        speaker_acc_results[speakerid] = results

    if overwrite_results:
        speakerwise_acc_df = pd.DataFrame.from_dict(speaker_acc_results).T
        speakerwise_acc_df.to_csv('./interspeech_results/speakerwise_acc_resultsJLSHR.csv')
else:
    speakerwise_acc_df = pd.read_csv('./interspeech_results/speakerwise_acc_resultsJSLHR.csv')


''' generate accuracy per phoneme class '''


''' generate interrater accuracy '''
interrater_path = '/media/prad/data/datasets/ChildSpeechDataset/interrater-check/'
manualalign_path = '/media/prad/data/datasets/ChildSpeechDataset/manually-aligned-text-grids'

import glob

method_to_dir = {
    'xvector': './phone_matched_xvec_proj_textgrids',
    'xvectorNewNoLDA': '/home/prad/github/DNN-PLLLR/',
    # 'xvector':,
    # 'mfa'
    'ivector': './results_sat',
    'frame': './results_frame_10epochs',
    'mfa_base':'./results_mfa_adapted_english_us_arpa',
    'mfa_train': './results_mfa_trained',
    'gt': '/media/prad/data/datasets/ChildSpeechDataset/manually-aligned-text-grids/',
    'interrater': '/media/prad/data/datasets/ChildSpeechDataset/interrater-check/'}

# for method in method_to_dir.keys():

    # methodpath = method_to_dir[method]

    # for speakerpath in glob.glob(interrater_path + '*'):
    #     speaker = speakerpath.split('/')[-1]

        # human1path = os.path.join(methodpath, speaker)
        # human1_interratertgs.extend([file for file in glob.glob(f'{human1path}/*')])

interrater_speakers = [speakerpath.split('/')[-1] for speakerpath in glob.glob(interrater_path + '*')]

interrater_acc_csvpath = './interspeech_results/acc_results_interrater.csv'
if not os.path.exists(interrater_acc_csvpath):
    for methodname in method_to_dir.keys():
        # methodname='xvector'
        if methodname=='gt':
            continue
        print('-------------------------------------------------------')
        print('Methodname:', methodname)
        correct_indicator_matched = []
        correct_indicator = []

        all_gt_tgs = []
        all_method_tgs = []

        for speakerid in interrater_speakers:
            tgs_manualpath = os.path.join(method_to_dir['gt'], speakerid)
            tgs_methodpath = os.path.join(method_to_dir[methodname], speakerid)
            tgs_manual = get_all_textgrids_in_directory(tgs_manualpath, verbose=False)
            tgs_method = get_all_textgrids_in_directory(tgs_methodpath, verbose=False)
            tgs_manual = [get_gt_tgpath(tgs_manual, _tgmethod) for _tgmethod in tgs_method]
            all_gt_tgs.extend(tgs_manual)
            all_method_tgs.extend(tgs_method)


        phonekey = 'ha phones' if methodname=='interrater' else 'phones'
        acc, acc_matched, numcorrect, numcorrect_matched, numpredicted, numpredicted_matched, correct_indicator, allphones, timingdfs  = \
            calc_acc_between_tg_lists(all_gt_tgs, all_method_tgs, collapse_shortphones=collapse_shortphones,
                                      manual_phonekey='ha phones', aligner_phonekey=phonekey)
        acc_results_dct[methodname] = {'Acc': acc,
                                       'NumCorrect':numcorrect,
                                       'NumPredicted':numpredicted,
                                       'AccMatched': acc_matched,
                                       'NumCorrectMatched':numcorrect_matched,
                                       'NumPredictedMatched':numpredicted_matched,
                                       'timingdfs':timingdfs}


    interrater_acc_df = pd.DataFrame.from_dict(acc_results_dct).T

    if overwrite_results:
        interrater_acc_df.to_csv(interrater_acc_csvpath)
print()
''' '''
# human2path = os.path.join(methodpath, speaker)
# human2_interratertgs.extend([file for file in glob.glob(f'{human2path}/*')])
#
# frametgpath = os.path.join(methodpath, speaker)
# interrater_frametgs.extend([file for file in glob.glob(f'{frametgpath}/*')])
#
# sattgpath = os.path.join(methodpath, speaker)
# interrater_ivectgs.extend([file for file in glob.glob(f'{sattgpath}/*')])
#
# xvectgpath = os.path.join(methodpath, speaker)
# interrater_xvectgs.extend([file for file in glob.glob(f'{xvectgpath}/*')])


