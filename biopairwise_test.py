from Bio import pairwise2
import functools

MATCH_SCORE = 0
MISMATCH_SCORE = 2

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

def compare_labels(ref, test, silence_phone='sil'):
    if ref == test:
        return 0
    if ref == silence_phone or test == silence_phone:
        return 10
    ref = ref.lower()
    test = test.lower()
    if ref == test:
        return 0
    return 2

def overlap_scoring(firstElement, secondElement, silence_phone='sil'):
    # print(firstElement)
    # print(secondElement)
    if firstElement==['-'] or secondElement == ['-']:
        begin_diff = abs(firstElement[0] - secondElement[0])
        end_diff = abs(firstElement[1] - secondElement[1])
        label_diff = compare_labels(firstElement[2], secondElement[2], silence_phone)
        return -1 * (begin_diff + end_diff + label_diff)
    else:
        return -2

def calc_alignment_cost(alignment, ref_len, silence_phone='sil', return_all_metrics=False):
    overlap_count = 0
    overlap_cost = 0
    num_insertions = 0
    num_deletions = 0
    num_substitutions = 0
    for a in alignment:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            print(sa, sb)
            if sa == '-':
                if not silence_check(sb[2]):
                    num_insertions += 1
                else:
                    continue
            elif sb == '-':
                if not silence_check(sb[2]):
                    num_deletions += 1
                else:
                    continue
            else:
                overlap_cost += abs(sa[0] - sb[0]) + abs(sa[1] - sb[1])
                overlap_count += 1

                if compare_labels(sa[2], sb[2], silence_phone) > 0:
                    num_substitutions += 1
        overlap_cost = -overlap_cost
        alignment_score = overlap_cost/2
        phone_error_rate = (num_insertions + num_deletions + 2 * num_substitutions)/ref_len
        if return_all_metrics:
            return overlap_cost, num_insertions, num_deletions, num_substitutions, alignment_score, phone_error_rate
        else:
            return alignment_score, phone_error_rate


if __name__=='__main__':
    _gtphn = ['sil', 'B', 'IY', 'sil']
    _prphn = ['sil', 'B', 'IY', 'sil', 'IY', 'sil']

    _gtrows = [(0, 0.3, 'sil'), (0.3, .5, 'B'), (.5, 3, 'IY'), (3, 4, 'sil')]
    _prrows = [(0, 0.5, 'sil'), (0.35, .5, 'B'), (.5, 1.25, 'IY'), (1.25, 2, 'sil'), (2, 3, 'IY'),  (3, 4, 'sil')]
    # _gtrows = [(0, 1, 'B'), (1, 3, 'IY'), (3, 4, 'sil')]
    # _prrows = [(0, 1, 'B'), (1, 3, 'IY'), (3, 4, 'sil')]
    score_func = functools.partial(overlap_scoring)
    alignment = pairwise2.align.globalcs(_gtrows, _prrows, score_func, -5, -5, gap_char=['-'], one_alignment_only=True)
    ref_len = len(_gtrows)
    # overlap_cost, num_insertions, num_deletions, num_substitutions, alignment_score, phone_error_rate = calc_alignment_cost(alignment, ref_len)
    alignment_score, phone_error_rate = calc_alignment_cost(alignment, ref_len)

    lenref = len(_gtrows)
    print('======================')
    print('Alignment Score:\t', alignment_score)
    print('Phone Error Rate:\t', phone_error_rate)
    print('======================')
    print(alignment)
    print(calc_alignment_cost(alignment, ref_len))
    print('======================')
    alignment = pairwise2.align.globalcs(_gtphn, _prphn, score_func, -5, -5, gap_char=['-'], one_alignment_only=True)
    print(alignment)

