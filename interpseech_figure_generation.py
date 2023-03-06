import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from phoneme_info import *
from scipy.stats import iqr

# methodnames = ['frame', 'ivector', 'xvector', 'mfa_base', 'mfa_train']
# label_dict = {'frame': 'Wav2Vec2-Frame', 'ivector': 'Wav2Vec2-iVec', 'xvector': 'Wav2Vec2-xVec',
#               'mfa_base': 'MFA no SAT', 'mfa_train': 'MFA with SAT'}

methodnames = ['frame', 'ivector', 'xvector', 'mfa_train']
label_dict = {'frame': 'Wav2Vec2-frame', 'ivector': 'Wav2Vec2-iVec',
              'xvector': 'Wav2Vec2-xVec', 'mfa_train': 'MFA with SAT'}
color_dct = {'frame': 'k', 'ivector': 'm',
              'xvector': 'g', 'mfa_train': 'tab:red'}
''' 

Average accuracy vs Age 

'''

df = pd.read_csv('./interspeech_results/speakerwise_acc_results.csv')
fig, ax = plt.subplots()
from sklearn.linear_model import LinearRegression
ages = df.Age.values.reshape(-1, 1)
age_range = np.linspace(3, 7.15).reshape(-1, 1)
predacc = LinearRegression().fit(ages, 100*df.Acc_frame.values.reshape(-1,1)).predict(age_range)
ax.scatter(ages, 100*df.Acc_frame.values, color='k', label=label_dict['frame'], alpha=.4,)
ax.plot(age_range, predacc, color='k', linewidth=2)

predacc = LinearRegression().fit(ages, 100*df.Acc_ivector.values.reshape(-1,1)).predict(age_range)
ax.scatter(ages, 100*df.Acc_ivector.values, color='m', label=label_dict['ivector'], alpha=.4)
ax.plot(age_range, predacc,'m', linewidth=2)

predacc = LinearRegression().fit(ages, 100*df.Acc_xvector.values.reshape(-1,1)).predict(age_range)
ax.scatter(ages, 100*df.Acc_xvector.values, color='g', label=label_dict['xvector'], alpha=.4)
ax.plot(age_range, predacc, linewidth=2, color='g')
import matplotlib.ticker as mticker
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

predacc = LinearRegression().fit(ages, 100*df.Acc_mfa_train.values.reshape(-1,1)).predict(age_range)
ax.scatter(ages, 100*df.Acc_mfa_train.values, color='tab:red', label=label_dict['mfa_train'], alpha=.4)
ax.plot(age_range, predacc, linewidth=2, color='tab:red')
import matplotlib.ticker as mticker
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

ax.set_title('Alignment Accuracy vs Age', size=14, fontweight='bold')
ax.set_xlabel('Age', size=12, fontweight='bold')
ax.set_ylabel('Alignment Accuracy', size=12, fontweight='bold')
ax.legend()
ax.grid()
fig.savefig('./interspeech_results/age_vs_align_acc.pdf', bbox_inches='tight')

'''

CDF of Onset/Offset Error (aggregate)

'''

phone_dfs = pkl.load(open('./outputs/phone_accuract_dfs_final.pkl', 'rb'))
# methodnames = ['frame', 'ivector', 'xvector', 'mfa_base', 'mfa_train']

onset_errors = {}
offset_errors = {}
onoff_err_all_dfs = {}
onoff_err_all_rslts = {}
for method in methodnames:
    onset_errors[method] = []
    offset_errors[method] = []
    onoff_err_all_dfs[method] = []
    onoff_err_all_rslts[method] = {} # SHOULD BE A DICTIONARY

for method in methodnames:
    for ii, phone in enumerate(ENGLISH_PHONEME_LIST):
        _df = phone_dfs[phone][method]
        onoff_err_all_dfs[method].append(_df)

    onoff_err_all_dfs[method] = pd.concat(onoff_err_all_dfs[method])
    _onoff_err_df = onoff_err_all_dfs[method]
    for col in _onoff_err_df.columns:
        if col!='phone' and col!='speakerid' and 'duration' not in col:
            descriptor = []
            if 'off' in col:
                descriptor.append('offset_err')
            if 'on' in col:
                descriptor.append('onset_err')
            if 'pct' in col:
                descriptor.append('pct')
            descriptor = '_'.join(descriptor)

            errvals = np.abs(_onoff_err_df[col].values)

            errmean = np.mean(errvals)
            errmed = np.nanmedian(errvals)
            errstd = np.std(errvals)

            iqrval = iqr(errvals[~np.isnan(errvals)])

            errlt20 = sum(errvals<20)/len(errvals)
            errlt50 = sum(errvals<50)/len(errvals)
            errlt100 = sum(errvals<100)/len(errvals)

            onoff_err_all_rslts[method][descriptor + '_mean'] = errmean
            onoff_err_all_rslts[method][descriptor + '_median'] = errmed
            onoff_err_all_rslts[method][descriptor + '_std'] = errstd

            onoff_err_all_rslts[method][descriptor + '_IQR'] = iqrval

            onoff_err_all_rslts[method][descriptor + '_lt20'] = errlt20
            onoff_err_all_rslts[method][descriptor + '_lt50'] = errlt50
            onoff_err_all_rslts[method][descriptor + '_lt100'] = errlt100

onoff_err_all_rslts_df = pd.DataFrame.from_dict(onoff_err_all_rslts)
onoff_err_all_rslts_df.to_csv('./interspeech_results/onoff_err_table.csv')

for ii, phone in enumerate(ENGLISH_PHONEME_LIST):
    for method in methodnames:
        values_on = np.abs(phone_dfs[phone][method][f'onset_err_{method}'].values)
        # values_on = values_on[~np.isnan(values_on)]
        onset_errors[method].extend(values_on)

        values_off = np.abs(phone_dfs[phone][method][f'offset_err_{method}'].values)
        # values_off = values_off[~np.isnan(values_off)]
        offset_errors[method].extend(values_off)

plt.figure(figsize=(5, 8.5))
plt.subplot(2, 1, 1)
# plt.figure()
for method in methodnames:
    _values = onset_errors[method]
    _values = np.clip(_values, 0, 500)
    thresholds = np.linspace(0, 500, 21)
    yvals = np.array([sum(_values<thresh)/len(_values) for thresh in thresholds])

    # histvals, bin_edges = np.histogram(_values, density=True, bins=20)
    # bin_width = bin_edges[1] - bin_edges[0]
    # cdf = np.cumsum(histvals) * bin_width

    # plt.plot(bin_edges[:-1], cdf, '-o', label=label_dict[method])
    plt.plot(thresholds, yvals, '-o',color=color_dct[method], label=label_dict[method])
    plt.ylim([.5, 1.04])
    plt.xlim([0, 300])
    plt.grid(True)

    plt.title('Onset Error Cumulative Distribution\n (All Phonemes)', size=14, fontweight='bold')
    plt.ylabel('Fraction of Phonemes', size=12, fontweight='bold')
    # plt.xlabel('Onset Error Threshold', size=12, fontweight='bold')
    plt.legend()
# plt.savefig('./interspeech_results/onset_errors.pdf', bbox_inches='tight')

# plt.figure()
plt.subplot(2, 1, 2)
for method in methodnames:
    _values = offset_errors[method]
    _values = np.clip(_values, 0, 500)
    # histvals, bin_edges = np.histogram(_values, density=True, bins=20)
    # bin_width = bin_edges[1] - bin_edges[0]
    # cdf = np.cumsum(histvals) * bin_width
    # plt.plot(bin_edges[:-1], cdf, '-o', label=label_dict[method])

    thresholds = np.linspace(0, 500, 21)
    yvals = np.array([sum(_values<thresh)/len(_values) for thresh in thresholds])
    plt.plot(thresholds, yvals, '-o', color=color_dct[method], label=label_dict[method])

    plt.ylim([.5, 1.04])
    plt.xlim([0, 300])
    plt.grid(True)

    plt.title('Offset Error Cumulative Distribution', size=14, fontweight='bold')
    plt.ylabel('Fraction of Phonemes', size=12, fontweight='bold')
    plt.xlabel('Error Threshold (ms)', size=12, fontweight='bold')
    plt.legend()

# plt.show()
# plt.savefig('./interspeech_results/offset_errors.pdf', bbox_inches='tight')
plt.savefig('./interspeech_results/onoff_errors.pdf', bbox_inches='tight')
'''

Per phoneme CDF of onset error

'''

import seaborn as sns
plt.figure()
idx = 0
import matplotlib.gridspec as gridspec
fig, ax = plt.subplots(6, 7, figsize=(50, 40), constrained_layout=True)
pct_short_gt = []
for ii, phone in enumerate(ENGLISH_PHONEME_LIST):
    plt.subplot(6, 7, ii + 1)
    row = int(ii/7)
    col = ii % 7
    for method in methodnames:
        # values = np.abs(np.clip(phone_dfs[ENGLISH_PHONEME_LIST[idx]][method][f'onset_err_pct_{method}'].values, 0, 100))
        values = np.abs(np.clip(phone_dfs[phone][method][f'onset_err_{method}'].values, 0, 500))
        values = values[~np.isnan(values)]
        histvals, bin_edges = np.histogram(values, density=True, bins=20)
        bin_width = bin_edges[1] - bin_edges[0]
        cdf = np.cumsum(histvals) * bin_width
        print(row, col, phone)
        ax[row, col].plot(bin_edges[:-1], cdf, '-o', label=label_dict[method])
        ax[row, col].set_ylim([.7, 1.04])
        ax[row, col].set_xlim([0, 100])
        ax[row, col].grid(True)

    ax[row, col].set_title(f'Onset Error for {phone}\nCumulative Distribution')
    ax[row, col].set_ylabel('Fraction of Phonemes')
    ax[row, col].set_xlabel('Onset Error Threshold')
    plt.legend()

print()


''' Generate on/off error per phoneme class '''
categories = np.array(['Vowel', 'Plosive', 'Fricative', 'Other'])
phone_category_onoff = {}

for category in categories:
    phone_category_onoff[category] = {}
    allphones_onoff = {}
    for method in methodnames:
        phone_category_onoff[category][method] = []
for phn in ENGLISH_PHONEME_LIST:
    phncategory = PHONEME_INFO_DF[PHONEME_INFO_DF['phoneme']==phn].type.values[0]
    category = [cat for cat in categories if cat.lower() in phncategory]
    category = 'Other' if len(category)==0 else category[0]

    for method in methodnames:
        phone_category_onoff[category][method].append(phone_dfs[phn][method])

onoff_err_by_category = {}
onoff_err_all = {}

for category in categories:

    onoff_err_by_category[category] = {}

    for method in methodnames:
        onoff_err_by_category[category][method] = {}
        _onoff_err_df = pd.concat(phone_category_onoff[category][method])

        for col in _onoff_err_df.columns:
            if col!='phone' and col!='speakerid' and 'duration' not in col:
                descriptor = []
                if 'off' in col:
                    descriptor.append('offset_err')
                if 'on' in col:
                    descriptor.append('onset_err')
                if 'pct' in col:
                    descriptor.append('pct')
                descriptor = '_'.join(descriptor)

                errvals = np.abs(_onoff_err_df[col].values)

                errmean = np.nanmean(errvals)
                errmed = np.nanmedian(errvals)

                errstd = np.std(errvals)
                iqrval = iqr(errvals[~np.isnan(errvals)])

                errlt20 = sum(errvals < 20) / len(errvals)
                errlt50 = sum(errvals < 50) / len(errvals)
                errlt100 = sum(errvals < 100) / len(errvals)

                onoff_err_by_category[category][method][descriptor + '_mean'] = errmean
                onoff_err_by_category[category][method][descriptor + '_std'] = errstd

                onoff_err_by_category[category][method][descriptor + '_lt20'] = errlt20
                onoff_err_by_category[category][method][descriptor + '_lt50'] = errlt50
                onoff_err_by_category[category][method][descriptor + '_lt100'] = errlt100

                onoff_err_all_rslts[method][descriptor + '_median'] = errmed
                onoff_err_all_rslts[method][descriptor + '_std'] = errstd
                onoff_err_all_rslts[method][descriptor + '_IQR'] = iqrval

for category in categories:
    category_dict = onoff_err_by_category[category]
    _df = pd.DataFrame.from_dict(category_dict)
    _df.to_csv(f'./interspeech_results/{category}_onoff_results.csv')

print()
def return_category(category_info):
    if 'plosive' in category_info:
        return 'plosive'
    elif 'vowel' in category_info:
        return 'vowel'
    elif 'fricative' in category_info:
        return 'fricative'
    else:
        return 'other'

runmethodnames = ['frame', 'ivec', 'xvec', 'mfa_train']
runmethodnames = ['xvec', 'mfa_train', 'ivec', 'xvec']

for methodname in runmethodnames:
    df1 = pd.read_csv(f'./interspeech_results/correct_indicator/{methodname}_correct_indicator.csv', index_col='Phones')
    category = []
    for ind in df1.index:
        try:
            _category = return_category(PHONEME_INFO_DF[PHONEME_INFO_DF['phoneme'] == ind]['type'].values[0])
        except:
            _category = 'other'
        category.append(_category)
    # category = [return_category(PHONEME_INFO_DF[PHONEME_INFO_DF['phoneme'] == ind]['type'].values[0]) if ind != 'sil' else 'sil'
    #     for ind in df1.index]
    df1['category'] = category

    print('---------------------------------------------------------------------')
    print('METHOD:\t', methodname)
    print('---------------------------------------------------------------------')
    for _category in np.unique(category):
        corr_ind = df1[df1['category']==_category]['CorrectIndicator']
        print('Subset', _category)
        print('Accuracy:\t', np.mean(corr_ind.values))

    print()
''' Generate onoff error accuracy by phoneme class df'''




