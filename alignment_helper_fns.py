import copy
import pandas as pd
import numpy as np
from g2p_en import G2p
from praatio.data_classes.textgrid import Textgrid
from praatio import textgrid
from phoneme_info import *
import tqdm
import re
import os

g2p = G2p()

def contains_same_phones(phone_list1, phone_list2):
    '''
        simply checks that two phoneme lists are the same
        for example that a ground truth textgrid and an estimated textgrid have the same list of phonemes
    '''
    try:
        # return np.logical_and.reduce(np.unique(phone_list1)==np.unique(phone_list2))
        return np.all(np.array(phone_list1)==np.array(phone_list2))
    except:
        return False

def remove_sil_from_phonelist(phonelist):
    res = list(filter(('[SIL]').__ne__, phonelist))
    res = list(filter(('sil').__ne__, res))
    return res


def get_df_rows_as_tuples(inpdf):
    return [tuple(row) for row in inpdf.itertuples(index=False)]


def add_before_after_silence(tgdf, manualdf):
    '''

    '''
    if tgdf.start.iloc[0] > 0:
        silence_row_df = pd.DataFrame({'start': tgdf.end.iloc[-1], 'end': manualdf.end.iloc[-1], 'phone': 'sil'},
                                      index=[0])
        tgdf = pd.concat([silence_row_df, tgdf]).reset_index(drop=True)

    if tgdf.end.iloc[-1] < manualdf.end.iloc[-1] and tgdf.phone.iloc[-1] != 'sil':
        silence_row_df = pd.DataFrame({'start': tgdf.end.iloc[-1], 'end': manualdf.end.iloc[-1], 'phone': 'sil'},
                                      index=[max(tgdf.index) + 1])
        tgdf = pd.concat([tgdf, silence_row_df]).reset_index(drop=True)
    return tgdf

def get_transcript_from_tgfile(tgfilepath, datapath = '/media/prad/data/datasets/ChildSpeechDataset/child_speech_16_khz'):
    '''
        each phoneme is associated with words in the transcript
        This function returns which phonemes in the textgrid occur at the start of a word

        Example:

        transcript - 'good day'
        phonelist = ['G', 'UH', 'D', ' ', 'D', 'AE', 'Y']

        1. the first phoneme at [0] is a start phone
        2a. find the locations of the spaces ' '
        2b. any phonemes occuring after a space are also start phones

    '''
    try:
        speaker_id = tgfilepath.split('/')[-2]
    except:
        print(tgfilepath)
    scriptfname = tgfilepath.split('/')[-1].split('.')[-2] + '.lab'
    scriptfpath = os.path.join(datapath,  speaker_id, scriptfname)
    f = open(scriptfpath)
    transcript = f.read().replace('\n', ' ')
    f.close()
    return transcript.strip()

def is_start_phone(phn, phonelist):
    '''
        each phoneme is associated with words in the transcript
        This function returns which phonemes in the textgrid occur at the start of a word

        Example:

        transcript - 'good day'
        phonelist = ['G', 'UH', 'D', ' ', 'D', 'AE', 'Y']

        1. the first phoneme at [0] is a start phone
        2a. find the locations of the spaces ' '
        2b. any phonemes occuring after a space are also start phones

    '''
    space_locs = np.argwhere(phonelist==' ').ravel()
    start_idxs = np.concatenate(([0], space_locs+1))
    phn_idx = np.argwhere(phonelist==phn)
    return any([_idx in start_idxs for _idx in phn_idx])

def is_end_phone(phn, phonelist):
    '''
    find the location of end phonemes
    does nearly the same thing as the is_start_phone function, except for end phones
    '''
    space_locs = np.argwhere(phonelist==' ').ravel()
    end_idxs = np.concatenate((space_locs-1, [len(phonelist)]))
    phn_idx = np.argwhere(phonelist==phn)
    return any([_idx in end_idxs for _idx in phn_idx])

def collapse_repeated_phones(input_df, phonekey='phone'):
    '''
    input:
        input_df - the input textgrid

    logic:
        - just concatenates consecutive identical phonemes

        for each line in the input textgrid df, check if the next phoneme is the same as the current phoneme
        if they are, simply

    '''
    # keepdata = []
    inp_df = copy.deepcopy(input_df)
    ii=0
    while ii<len(inp_df)-1:
        if inp_df.at[ii, phonekey]==inp_df.at[ii+1, phonekey]:
            newend = inp_df.at[ii+1, 'end']
            inp_df.at[ii, 'end'] = newend
            inp_df = inp_df.drop(ii+1, axis=0).reset_index(drop=True)
        else:
            ii+=1
    return inp_df

def process_silences(inp_df, transcript: str, silphone='sil'):
    phonelist = g2p(transcript)
    tgdf = copy.deepcopy(inp_df)
    ''' flag silences in the middle of a word with nan'''
    for ii in range(len(tgdf)):
        if ii < len(tgdf) - 1 and ii > 0:
            # print(tgdf[phonekey][ii])
            if tgdf['phone'][ii] == silphone:
                prevphone = tgdf['phone'].iloc[ii - 1]
                nextphone = tgdf['phone'].iloc[ii + 1]

                if prevphone == nextphone and not (
                        is_end_phone(prevphone, phonelist) and is_start_phone(nextphone, phonelist)):
                    tgdf.at[ii, 'phone'] = np.nan

    ''' remove the silences'''
    # tgdf = tgdf[~pd.isna(tgdf[phonekey])].reset_index(drop=True).drop(columns=['index'])
    tgdf = tgdf[~pd.isna(tgdf['phone'])].reset_index(drop=True)
    ''' collapse the repeated phonemes '''
    return collapse_repeated_phones(tgdf, phonekey='phone')



def get_all_textgrids_in_directory(directory, verbose=True):
    textgrid_files = []
    if verbose:
        print('Extracting all textgrids in directory:\t', directory)

    if verbose:
        for ii, (path, subdirs, files) in tqdm.tqdm(enumerate(os.walk(directory))):
            for name in files:
                if 'TextGrid' in name:
                    _textgridfile = os.path.join(path, name)
                    textgrid_files.append(_textgridfile)
    else:
        for ii, (path, subdirs, files) in enumerate(os.walk(directory)):
            for name in files:
                if 'TextGrid' in name:
                    _textgridfile = os.path.join(path, name)
                    textgrid_files.append(_textgridfile)

    return textgrid_files

def get_all_filetype_in_directory(directory, filetype):
    audiofiles = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if filetype in fname:
                _fpath = os.path.join(root, fname)
                print(_fpath)
                audiofiles.append(_fpath)

def textgridpath_to_phonedf(txtgrid_path: str, phone_key: str, remove_numbers=False, replace_silence=True, includeEmptyIntervals=False):
    '''
    txtgrid_path - the path to the textgridfile
    phone_key - the key in the textgrid for the phoneme column
    '''
    txtgrid = textgrid.openTextgrid(txtgrid_path, includeEmptyIntervals=includeEmptyIntervals)
    phndf = extract_phone_df_from_textgrid(txtgrid=txtgrid, phone_key=phone_key, remove_numbers=remove_numbers)
    if includeEmptyIntervals:
        phndf = phndf.replace('', 'sil')
    if replace_silence:
        phndf = phndf.replace('[SIL]', 'sil')
        phndf = phndf.replace('sp', 'sil')
        # phndf.iloc[phndf.iloc[:, 2] == '[SIL]', 2] = 'sil'
        # phndf.iloc[phndf.iloc[:, 2] == '[SIL]', 2] = 'sp'

    return phndf


def extract_phone_df_from_textgrid(txtgrid: Textgrid, phone_key, remove_numbers=False, silchar='[SIL]', replace_SP=True):
    '''
        txtgrid - praatio textgrid
        phone_key - the key for the phonemes
    '''
    phonelist = txtgrid.tierDict[phone_key].entryList
    phonedf = []
    for interval in phonelist:
        _phone = interval.label
        if remove_numbers:
            _phone = re.sub(r'[0-9]+', '', _phone)
        phonedf.append([interval.start, interval.end, _phone])

    phonedf = pd.DataFrame(phonedf, columns=['start', 'end', 'phone'])
    phonedf = phonedf.replace('sil', '[SIL]')
    if replace_SP:
        phonedf = phonedf.replace('sp', '[SIL]')
    return phonedf


'''

Accuracy calculation function

'''
def unique_phonemes_in_tgs(tglist, phone_key, remove_numbers):
    allphones = []

    for tg in tglist:
        tgdf = textgridpath_to_phonedf(tg, phone_key, remove_numbers)
        allphones.extend(list(tgdf.iloc[:, 2].values))

    return list(np.unique(allphones))

def calc_accuracy(predphn_df, annotated_midpoints_dict, ignore_extras=False):
    correct_preds = []

    annotated_phns = list(annotated_midpoints_dict.keys())
    #     not_in = False
    #     if any([_phn not in annotated_phns for _phn in list(predphn_df.iloc[:,2])]):
    #         print('Mismatched Transcripts')

    #     print(predphn_df)
    #     print(annotated_midpoints_dict)
    for ii in range(len(predphn_df)):
        _phone = predphn_df.iloc[ii, 2]
        _start = predphn_df.iloc[ii, 0]
        _end = predphn_df.iloc[ii, 1]
        if _phone in annotated_phns:
            midpoints = annotated_midpoints_dict[_phone]
            # for midpt in midpoints:
                #                 print('Phoneme:\t', _phone)
                #                 print('Annotated Midpoint:\t', midpt)
                #                 print('Alinger Start:\t', _start)
                #                 print('Alinger End:\t', _end)
            if any([midpt < _end and midpt > _start for midpt in midpoints]):
                contains_midpoint = 'Correct'
            else:
                contains_midpoint = 'Incorrect'

        elif ignore_extras:
            # if ignore_extras is True, the
            #             print('extra')
            contains_midpoint = 'Extra'

        else:
            contains_midpoint = 'Incorrect'

        if contains_midpoint == 'Correct':
            correct_preds.append(1)
        elif contains_midpoint == 'Incorrect':
            correct_preds.append(0)
    #         elif contains_midpoint=='Extra':
    #             correct_preds.append(-1)

    return np.array(correct_preds)


# phn_midpointds_df = extract_phn_midpoint_dict_from_df(manualdf)
# calc_accuracy(predphn_df = mfadf, annotated_midpoints_dict=phn_midpointds_df)
def extract_phn_midpoint_dict_from_df(phoneme_df):
    midpoint_dict = {}
    for ii in range(len(phoneme_df)):
        _phone = phoneme_df.iloc[ii, 2]

        _midpoint = np.mean(phoneme_df.iloc[ii, 1] + phoneme_df.iloc[ii, 0]) / 2
        #         print('\nPhoneme:\t', _phone)
        #         print('Annotated Midpoint:\t', _midpoint)
        #         print('Alinger Start:\t', phoneme_df.iloc[ii,0])
        #         print('Alinger End:\t', phoneme_df.iloc[ii, 1])
        if _phone in midpoint_dict.keys():
            midpoint_dict[_phone].append(_midpoint)
        else:
            midpoint_dict[_phone] = [_midpoint]
    return midpoint_dict


def calc_alignment_accuracy_between_textgrids(manual_textgridpath: str, aligner_textgridpath: str, manual_phonekey: str,
                                              aligner_phonekey: str, collapse_shortphones: bool,
                                              remove_numbers=True, ignore_extras=True,
                                              replace_silence=True, ):
    #TODO: return the label along with whether it was correct so that you can figure out which phonemes were wrong

    manualdf = textgridpath_to_phonedf(manual_textgridpath, phone_key=manual_phonekey, remove_numbers=True,
                                       replace_silence=replace_silence, includeEmptyIntervals=True)
    alignerdf = textgridpath_to_phonedf(aligner_textgridpath, phone_key=aligner_phonekey, remove_numbers=True,
                                        replace_silence=replace_silence, includeEmptyIntervals=True)
    manual_phones = np.array(remove_sil_from_phonelist(manualdf.phone.values))
    aligner_phones = np.array(remove_sil_from_phonelist(alignerdf.phone.values))
    ismatch_pre = contains_same_phones(manual_phones, aligner_phones)

    transcript = get_transcript_from_tgfile(aligner_textgridpath)
    if collapse_shortphones:
        alignerdf = process_silences(alignerdf, transcript)

    ismatch = contains_same_phones(manual_phones, aligner_phones)
    timingdf = pd.merge(alignerdf.reset_index(), manualdf.reset_index(), left_on=['index', 'phone'],
                 right_on=['index', 'phone'], suffixes=('_pred', '_gt')).drop(columns='index')

    if len(timingdf)>len(alignerdf) or len(timingdf)>len(alignerdf) or len(timingdf)==0:
        print()
        pass
    #     timingdf=None
    # else:

    # if not ismatch:
    #     print()
    #     pass
    # phn_midpoints_dict = extract_phn_midpoint_dict_from_df(manualdf)
    # correct_indicator = calc_accuracy(predphn_df=alignerdf, annotated_midpoints_dict=phn_midpoints_dict,
    #                                   ignore_extras=ignore_extras)
    if not ismatch:
        print()
        pass

    phn_midpoints_dict = extract_phn_midpoint_dict_from_df(manualdf)
    correct_indicator = calc_accuracy(predphn_df=alignerdf, annotated_midpoints_dict=phn_midpoints_dict,
                                      ignore_extras=ignore_extras)
    #     if any([c==-1 for c in correct_indicator]):
    if type(correct_indicator) == int:
        print('---------------------------------------')
        print('**********Manual************\n', manualdf)
        print('**********Aligned***********\n', alignerdf)



    return correct_indicator, ismatch, timingdf

def calc_accuracy_between_textgrid_lists(manual_textgrid_list, estimated_textgrid_list, manual_phonekey='ha phones',
                                         aligner_phonekey='phones', ignore_extras=True, ignore_silence=False,
                                         verbose=True, colapse_shortphones=True):
    if verbose:
        print('Caclulating alignment accuracy...')

    correct_indicator = []
    correct_indicator_matched = []
    matched_indicator = []
    for ii, (manual_textgridpath, estimated_textgridpath) in tqdm.tqdm(enumerate(zip(manual_textgrid_list, estimated_textgrid_list))):
        try:
            _correct_indicator, ismatch = calc_alignment_accuracy_between_textgrids(manual_textgridpath=manual_textgridpath,
                                                                           aligner_textgridpath=estimated_textgridpath,
                                                                           manual_phonekey=manual_phonekey,
                                                                           aligner_phonekey=aligner_phonekey,
                                                                           ignore_extras=ignore_extras,
                                                                           ignore_silence=ignore_silence,
                                                                           collapse_shortphones=colapse_shortphones)
            correct_indicator.append(_correct_indicator)
            if ismatch:
                correct_indicator_matched.append(_correct_indicator)
        except:
            if not os.path.exists(estimated_textgridpath):
                print('Textgrid file ', estimated_textgridpath, ' not found, skippping this file')
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

    return acc, acc_matched, numcorrect, numcorrect_matched, numpredicted, numpredicted_matched, matched_indicator

