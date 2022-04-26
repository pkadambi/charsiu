import pandas as pd
import numpy as np
from praatio.data_classes.textgrid import Textgrid
from praatio import textgrid
import tqdm
import re
import os

def get_all_textgrids_in_directory(directory):
    textgrid_files = []
    print('Extracting all textgrids in directory:\t', directory)
    for ii, (path, subdirs, files) in tqdm.tqdm(enumerate(os.walk(directory))):
        for name in files:
            if 'TextGrid' in name:
                _textgridfile = os.path.join(path, name)
                textgrid_files.append(_textgridfile)
                # print(_textgridfile)
    return textgrid_files

def textgridpath_to_phonedf(txtgrid_path: str, phone_key: str, remove_numbers=False, replace_silence=True):
    '''
    txtgrid_path - the path to the textgridfile
    phone_key - the key in the textgrid for the phoneme column
    '''
    txtgrid = textgrid.openTextgrid(txtgrid_path, False)
    phndf = extract_phone_df_from_textgrid(txtgrid=txtgrid, phone_key=phone_key, remove_numbers=remove_numbers)
    if replace_silence:
        phndf.iloc[phndf.iloc[:, 2] == '[SIL]', 2] = 'sil'

    return phndf


def extract_phone_df_from_textgrid(txtgrid: Textgrid, phone_key, remove_numbers=False):
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

    phonedf = pd.DataFrame(phonedf)
    return phonedf


'''

Accuracy calculation function

'''
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
            for midpt in midpoints:
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
                                              aligner_phonekey: str, remove_numbers=True, ignore_extras=True,
                                              ignore_silence=False):
    #TODO: return the label along with whether it was correct so that you can figure out which phonemes were wrong

    manualdf = textgridpath_to_phonedf(manual_textgridpath, phone_key=manual_phonekey, remove_numbers=True)
    alignerdf = textgridpath_to_phonedf(aligner_textgridpath, phone_key=aligner_phonekey, remove_numbers=True,
                                        replace_silence=ignore_silence)
    phn_midpoints_dict = extract_phn_midpoint_dict_from_df(manualdf)
    correct_indicator = calc_accuracy(predphn_df=alignerdf, annotated_midpoints_dict=phn_midpoints_dict,
                                      ignore_extras=ignore_extras)
    #     if any([c==-1 for c in correct_indicator]):
    if type(correct_indicator) == int:
        print('---------------------------------------')
        print('**********Manual************\n', manualdf)
        print('**********Aligned***********\n', alignerdf)
    return correct_indicator

def calc_accuracy_between_textgrid_lists(manual_textgrid_list, estimated_textgrid_list, manual_phonekey='ha phones',
                                         aligner_phonekey='phones', ignore_extras=True, ignore_silence=False):
    print('Caclulating alignment accuracy...')
    phoneme_correct_indicator = []
    for ii, (manual_textgridpath, estimated_textgridpath) in tqdm.tqdm(enumerate(zip(manual_textgrid_list, estimated_textgrid_list))):
        try:
            _correct_indicator = calc_alignment_accuracy_between_textgrids(manual_textgridpath=manual_textgridpath,
                                                                           aligner_textgridpath=estimated_textgridpath,
                                                                           manual_phonekey='ha phones',
                                                                           aligner_phonekey='phones',
                                                                           ignore_extras=ignore_extras,
                                                                           ignore_silence=ignore_silence)
            phoneme_correct_indicator.append(_correct_indicator)
        except:
            if not os.path.exists(estimated_textgridpath):
                print('Textgrid file ', estimated_textgridpath, ' not found, skippping this file')

    phoneme_correct_indicator = np.concatenate(phoneme_correct_indicator)
    acc = np.mean(phoneme_correct_indicator)
    numcorrect = np.sum(phoneme_correct_indicator)
    numpredicted = len(phoneme_correct_indicator)

    print('Accuracy Excluding Extra Phonemes')
    print('Accuracy:\t', np.mean(phoneme_correct_indicator))
    print('Num Correct:\t', numcorrect)
    print('Num Predicted Phones:\t', numpredicted)
    return acc, numcorrect, numpredicted
