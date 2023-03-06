from Bio import pairwise2
import functools
import math
MATCH_SCORE = 0
MISMATCH_SCORE = 2
GAP_START_SCORE = -2
GAP_CONTINUE_SCORE = -2
import numpy as np

EXCLUDE_FILES = ['0505_M_EKs4T10', '0411_M_LMwT32']

def silence_check(phone):
    return phone in {'sp', '<p:>', 'sil', '', None}

def _compare_labels(ref, test):
    if ref == test:
        return MATCH_SCORE
    ref = ref.lower()
    test = test.lower()
    if ref == test:
        return MATCH_SCORE
    return MISMATCH_SCORE

def print_alignments(alignment):
    for a in alignment:
        for i, sa in enumerate(a.seqA):
            print(a.seqA[i], ' --> ', a.seqB[i])

def compare_labels(ref, test, silence_phone='sil'):
    if ref == test:
        return 0
    # if ref == silence_phone or test == silence_phone:
    #     return 10
    ref = ref.lower()
    test = test.lower()
    if ref == test:
        return 0
    return 2

def overlap_scoring(firstElement, secondElement, silence_phone='sil'):
    if firstElement==['-'] or secondElement == ['-']:
        begin_diff = abs(firstElement[0] - secondElement[0])
        end_diff = abs(firstElement[1] - secondElement[1])
        label_diff = compare_labels(firstElement[2], secondElement[2], silence_phone)
        return -1 * (begin_diff + end_diff + label_diff)
    else:
        return -5

def calc_alignment_cost(ref, test, silence_phone='sil', return_all_metrics=False):
    score_func = functools.partial(overlap_scoring)
    ref_len = len(ref)
    alignment = pairwise2.align.globalcs(ref, test, score_func, GAP_START_SCORE, GAP_CONTINUE_SCORE, gap_char=['-'], one_alignment_only=True)
    overlap_count = 0
    overlap_cost = 0
    num_insertions = 0
    num_deletions = 0
    num_substitutions = 0
    num_matched = 0
    matched_overlap_cost = 0
    for a in alignment:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            # print(sa, sb)
            if sa == '-':
                if not silence_check(sb[2]):
                    num_insertions += 1
                else:
                    continue
            elif sb == '-':
                if not silence_check(sa[2]):
                    num_deletions += 1
                else:
                    continue
            else:
                item_cost = abs(sa[0] - sb[0]) + abs(sa[1] - sb[1])
                overlap_cost += item_cost
                overlap_count += 1

                if compare_labels(sa[2], sb[2], silence_phone) > 0:
                    num_substitutions += 1
                elif compare_labels(sa[2], sb[2], silence_phone)==0:
                    matched_overlap_cost += item_cost
                    num_matched += 1

        overlap_cost = -overlap_cost if overlap_cost!=0 else np.nan
        matched_overlap_cost = -matched_overlap_cost if matched_overlap_cost!=0 else np.nan

        alignment_score = overlap_cost/2
        phone_error_rate = (num_insertions + num_deletions + 2 * num_substitutions)/ref_len
        matched_alignment_score = math.nan if num_matched==0 else matched_overlap_cost/num_matched
        rslts = {'OverlapCost': overlap_cost, 'MatchedOverlapCost': matched_overlap_cost,
                 'AlignmentScore': alignment_score, 'MatchedAlignmentScore': matched_alignment_score,
                 'NumInsertions': num_insertions, 'NumDeletions': num_deletions,
                 'NumSubstitutions': num_substitutions, 'NumMatched': num_matched,
                 'PhoneErrorRate': phone_error_rate}
        return rslts
        # if return_all_metrics:
        #     return overlap_cost, matched_overlap_cost, num_insertions, num_deletions, num_substitutions, alignment_score, phone_error_rate
        # else:
        #     return alignment_score, phone_error_rate, matched_overlap_cost, num_matched

def evaluate_aligner_metrics(ground_truth_textgrids, estimated_textgrids, FILTER_SILENCE = True):
    # params: ground_truth_textgrids, estimated_textgrids
    per_file_rslts = []
    for gt_tgpth, est_tgpth in tqdm.tqdm(zip(ground_truth_textgrids, estimated_textgrids)):
        transcript = get_transcript_from_tgfile(gt_tgpth)
        _gttg = textgridpath_to_phonedf(gt_tgpth, phone_key='ha phones', remove_numbers=True)

        _esttg = textgridpath_to_phonedf(est_tgpth, phone_key='phones', remove_numbers=True)
        _esttg = add_before_after_silence(tgdf=_esttg, manualdf=_gttg)
        if FILTER_SILENCE:
            _esttg = process_silences(_esttg, transcript)
        _gt = get_df_rows_as_tuples(_gttg)
        _est = get_df_rows_as_tuples(_esttg)
        _rslts = calc_alignment_cost(_gt, _est)
        per_file_rslts.append(_rslts)
    return pd.DataFrame(per_file_rslts)

if __name__=='__main__':
    '''
    Runs a simple test case
    '''
    from alignment_helper_fns import *
    # esttg = './results_sat_xvector/0505_M_EK/0505_M_EKs7T01.TextGrid'
    # gttg = '/home/prad/datasets/ChildSpeechDataset/manually-aligned-text-grids/0505_M_EK/0505_M_EKs7T01.TextGrid'


    esttg = './results_mfa_adapted/0407_M_SJ/0407_M_SJs2T05.TextGrid'
    gttg = '/home/prad/datasets/ChildSpeechDataset/manually-aligned-text-grids/0407_M_SJ/0407_M_SJs2T05.TextGrid'
    gt = get_df_rows_as_tuples(textgridpath_to_phonedf(gttg, phone_key='ha phones', remove_numbers=True))
    estim = get_df_rows_as_tuples(textgridpath_to_phonedf(esttg, phone_key='phones', remove_numbers=True))

    rslts = calc_alignment_cost(gt, estim, return_all_metrics=True)

    # lenref = len(_gtrows)
    print('======================')
    print('Alignment Score:\t', rslts['AlignmentScore'])
    print('Matched Overlap Cost:\t', rslts['MatchedOverlapCost'])
    print('Overlap Cost:\t', rslts['OverlapCost'])
    print('Phone Error Rate:\t', rslts['PhoneErrorRate'])

    print('---------------------')
    print('Num substitutions:\t', rslts['NumSubstitutions'])
    print('Num Deletions:\t', rslts['NumDeletions'])
    print('Num Insertions:\t', rslts['NumInsertions'])
    print('Num NumMatched:\t', rslts['NumMatched'])
    print('======================')

    allmanual_tgs = [pth for pth in get_all_textgrids_in_directory(
        '/home/prad/datasets/ChildSpeechDataset/manually-aligned-text-grids/') if '.TextGrid' in pth]
    allmanual_tgs = [tg for tg in allmanual_tgs if all([_excludefile not in tg for _excludefile in EXCLUDE_FILES])]

    xvector_tgs = [pth for pth in get_all_textgrids_in_directory('./results_sat_xvector') if '.TextGrid' in pth]
    xvector_tgs = [tg for tg in xvector_tgs if all([_excludefile not in tg for _excludefile in EXCLUDE_FILES])]

    matchedxvec_tgs = [pth for pth in get_all_textgrids_in_directory('./phone_matched_xvec_proj_textgrids') if '.TextGrid' in pth]
    matchedxvec_tgs = [tg for tg in matchedxvec_tgs if all([_excludefile not in tg for _excludefile in EXCLUDE_FILES])]

    frame_tgs = [pth for pth in get_all_textgrids_in_directory('./results_frame_10epochs') if '.TextGrid' in pth]
    frame_tgs = [tg for tg in frame_tgs if all([_excludefile not in tg for _excludefile in EXCLUDE_FILES])]

    mfa_tgs = [pth for pth in get_all_textgrids_in_directory('./results_mfa_adapted') if '.TextGrid' in pth]
    mfa_tgs = [tg for tg in mfa_tgs if all([_excludefile not in tg for _excludefile in EXCLUDE_FILES])]


    nmanual, nxvec, nmfa = len(allmanual_tgs), len(xvector_tgs), len(mfa_tgs)

    xvecrsltsdf = evaluate_aligner_metrics(allmanual_tgs, xvector_tgs)
    matchedxvecrsltsdf = evaluate_aligner_metrics(allmanual_tgs, matchedxvec_tgs)
    mfarsltsdf = evaluate_aligner_metrics(allmanual_tgs, mfa_tgs)
    framersltsdf = evaluate_aligner_metrics(allmanual_tgs, frame_tgs)

    def print_avg_results(rsltsdf):
        print(f"AlignmentScore:\t{rsltsdf['AlignmentScore'].mean():.3f}")
        print(f"OverlapCost:\t{rsltsdf['OverlapCost'].mean():.3f}")
        print(f"MatchedOverlapCost:\t{rsltsdf['MatchedOverlapCost'].mean():.3f}")
        print(f"PhoneErrorRate:\t{rsltsdf['PhoneErrorRate'].mean():.3f}")
        print(f"NumSubstitutions:\t{rsltsdf['NumSubstitutions'].mean():.3f}")
        print(f"NumInsertions:\t{rsltsdf['NumInsertions'].mean():.3f}")
        print(f"NumDeletions:\t{rsltsdf['NumDeletions'].mean():.3f}")
        print(f"NumMatched:\t{rsltsdf['NumMatched'].mean():.3f}")

    def print_median_results(rsltsdf):
        print(f"AlignmentScore:\t{rsltsdf['AlignmentScore'].median():.3f}")
        print(f"OverlapCost:\t{rsltsdf['OverlapCost'].median():.3f}")
        print(f"MatchedOverlapCost:\t{rsltsdf['MatchedOverlapCost'].median():.3f}")
        print(f"PhoneErrorRate:\t{rsltsdf['PhoneErrorRate'].median():.3f}")
        print(f"NumSubstitutions:\t{rsltsdf['NumSubstitutions'].median():.3f}")
        print(f"NumInsertions:\t{rsltsdf['NumInsertions'].median():.3f}")
        print(f"NumDeletions:\t{rsltsdf['NumDeletions'].median():.3f}")
        print(f"NumMatched:\t{rsltsdf['NumMatched'].median():.3f}")

    print()


