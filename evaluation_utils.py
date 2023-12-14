import numpy as np
import pandas as pd
from scipy.stats import iqr
from alignment_helper_fns import *

def phone_in_tg(tg, phone, count=False):
    '''
    check if a phoneme is in a textgrid dataframe
    (assumes that phones are at iloc column 2)
    '''
    tgphones = np.unique(tg.iloc[:, 2].values).ravel()
    return phone in tgphones


def get_phonecount(tg, phone, key=None, loc=None):
    ''' count the number of predicted phones in a textgrid '''
    return sum(tg.iloc[:, 2].values == phone)


def get_phone_startend(tg, phone, key=None, loc=None):
    '''
    input:
        tg: textgrid dataframe
        phone: the phoneme in question

    get all starting and ending times for a given phoneme in a textgrid
    '''
    if key is not None:
        loc = np.argwhere(tg.columns == key).ravel()[0]

    tgphonedf = tg[tg.iloc[:, 2] == phone]
    starttimes = tgphonedf.iloc[:, 0].values
    endtimes = tgphonedf.iloc[:, 1].values
    return starttimes, endtimes


import numpy as np


def get_onoff_error(tg_gt, tg_estim, phone, verbose=False):
    '''
    assumption:
        phonemes at iloc column 2 of the dataframe
        *** ASSUMES the phoneme occurs the same number of times in each textgrid ***
        *** will return nan for error metrics otherwise ***
    input:
        tg_gt: ground truth textgrid
        tg_estim: estimated textgrid
        phone: phone in question to calculate alignment accuracy for

    returns:
        for all occurrences of a given phoneme:
            - onset error, offset error, duration error, ground truth duration, estimated duration

    '''
    gtstart, gtend = get_phone_startend(tg_gt, phone, loc=2)
    start, end = get_phone_startend(tg_estim, phone, loc=2)

    if len(gtstart) != len(start):
        if verbose:
            print('---------------------------------------------------------------------------------------------')
            print('Exception for phone [%s]: Ground truth has %d %ss, estimate has %d %ss' % (
            phone, len(gtstart), phone, len(start), phone))
            print('------ Estimated ------')
            print(tg_estim)
            print('------ Ground Truth ------')
            print(tg_gt)
        duration_gt = np.array([np.nan])
        duration_est = np.array([np.nan])

        duration_err = np.array([np.nan])
        onset_err = np.array([np.nan])
        offset_err = np.array([np.nan])
        error_indicator = True
    else:
        duration_gt = gtend - gtstart
        duration_est = end - start
        duration_err = duration_gt - duration_est
        onset_err = gtstart - start
        offset_err = gtend - end

    return onset_err, offset_err, duration_err, duration_gt, duration_est


def get_phone_durations(tg, phone, key=None, loc=None):
    '''
    inputs:
        tg: textgrid dataframe
        phone: the phoneme to get durations for

        key: key for the phoneme
        loc: column loccation where the phonemes occur (only specify if key is not passed)

    '''
    if key is not None:
        loc = np.argwhere(tg.columns == key).ravel()[0]

    start, end = get_phone_startend(tg, phone, loc=2)

    return end - start


def tg_error(tg_gt, tg_estim, phone, loc=2, verbose=False):
    '''
        returns a dictionary of [onset_error, offset_error, duration_error, duration_gt, duration_est]
        between a ground truth textgrid and an estimated textgrid
        *for a specific phoneme*

        NOTE: REQUIRES THAT THE PHONEME OCCURS THE SAME NUMBER OF TIMES IN BOTH TEXTGRIDS OR AN ERROR IS RETURNED
    '''

    stats = {}

    onset_err, offset_err, duration_error, duration_gt, duration_est = get_onoff_error(tg_gt, tg_estim, phone,
                                                                                       verbose=verbose)

    stats['durations_gt'] = duration_gt
    stats['durations_est'] = duration_est
    stats['onset_error'] = onset_err
    stats['offset_error'] = offset_err
    stats['duration_error'] = duration_error

    return stats

def get_gt_tgpath(manual_tgpaths, target_tgpath):
    fname = target_tgpath.split('/')[-1]
    idx = np.argwhere([fname in mtg for mtg in manual_tgpaths]).ravel()[0]
    return manual_tgpaths[idx]

def get_gt_tg(manual_tgpaths, target_tgpath, phone_key='ha phones'):
    '''
        gets the corresponding manual textgrid for a target textgrid
    '''
    corresponding_gt_tgpath = get_gt_tgpath(manual_tgpaths, target_tgpath)
    return textgridpath_to_phonedf(corresponding_gt_tgpath, phone_key=phone_key, remove_numbers=True)


def evaluate_tg_results(method, phone, tglist, tgs_manual, durations_est, onset_err, offset_err, phone_key='phones',
                        return_iqr = False):
    est_dur = []
    off_err = []
    on_err = []
    nerr = 0
    on_err_pct = []
    off_err_pct = []
    for tgpath in tglist:
        transcript = get_transcript_from_tgfile(tgpath)
        tg = textgridpath_to_phonedf(tgpath, phone_key=phone_key, remove_numbers=True)
        if not 'mfa' in method:
            tg = process_silences(tg, transcript)
        tg_gt = get_gt_tg(tgs_manual, target_tgpath=tgpath)



        tgp = textgrid.Textgrid()
        tgtup = [tuple(val) for val in list(tg.values)]
        phoneTier = textgrid.IntervalTier('phones', tgtup, 0, tgtup[-1][1])
        tgp.addTier(phoneTier)
        outpath = tgpath.replace('textgrids', 'processed')
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        tgp.save(tgpath.replace('textgrids', 'processed'), format="short_textgrid", includeBlankSpaces=False)

        if phone_in_tg(tg, phone) and phone_in_tg(tg_gt, phone):
            metrics = tg_error(tg_gt, tg, phone)
            # print('here')
            # print(metrics)
            # durations_est[method].extend(list(metrics['durations_est'] * 1000))
            # onset_err[method].extend(list(metrics['onset_error'] * 1000))
            # offset_err[method].extend(list(metrics['offset_error'] * 1000))
            on_err_pct.extend(list(100 * metrics['onset_error'] / metrics['durations_gt']))
            off_err_pct.extend(list(100 * metrics['offset_error'] / metrics['durations_gt']))
            est_dur.extend(list(metrics['durations_est'] * 1000))
            on_err.extend(list(metrics['onset_error'] * 1000))
            off_err.extend(list(metrics['offset_error'] * 1000))

            if any(np.isnan(metrics['onset_error'])):
                # nfiles_error['frame']+=1
                nerr += 1
            # print(on_err_pct)
            # print(on_err)
    return est_dur, on_err, on_err_pct, off_err, off_err_pct, nerr


