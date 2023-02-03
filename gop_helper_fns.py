from src.Charsiu import charsiu_forced_aligner
from alignment_helper_fns import *
from scipy.io import wavfile
import torch
import soundfile as sf
from praatio.data_classes.textgrid import Textgrid
from praatio import textgrid
#TODO: re-implement as an ARTP class that will take the files and
if not torch.cuda.is_available():
    DEVICE='cpu'
else:
    DEVICE='cuda'
CHARSIU_MODEL = charsiu_forced_aligner('charsiu/en_w2v2_fc_10ms', device=DEVICE)
PHONE2IND = CHARSIU_MODEL.charsiu_processor.processor.tokenizer.encoder
IND2PHONE = CHARSIU_MODEL.charsiu_processor.processor.tokenizer.decoder

# MIN_PHONE_LENGTH_DF = pd.read_csv('./min_phone_lengths.csv')
MIN_PHONE_LENGTH_DF = []

''' get logprobas from healthy aligner'''

def get_logprobas(audio_path, charsiu):
    audio_signal, fs = sf.read(audio_path)
    assert fs == 16000
    # charsiu.aligner._get_feat_extract_output_lengths(len(audio_signal))
    audio = charsiu.charsiu_processor.audio_preprocess(audio_signal)
    inputs = torch.tensor(audio).float().unsqueeze(0).to(charsiu.device)
    with torch.no_grad():
        out = charsiu.aligner(inputs)
    logits = out[0]
    logprobas = torch.log_softmax(logits, dim=-1).detach().cpu().numpy().squeeze()
    return logprobas

''' get aligned phone ids from aligned df'''


def return_aligned_phns_and_ids(aligned_df, seqlen, phone2id_dict=PHONE2IND):
    timesteps = .01 * np.arange(seqlen) + .01
    # timesteps = .01 * np.arange(530)

    aligned_phns = []



    for timestep in timesteps:
        gt_start_indicator = timestep > aligned_df.iloc[:, 0].values
        lt_end_indicator = timestep <= aligned_df.iloc[:, 1].values
        match_indicator = np.logical_and(gt_start_indicator, lt_end_indicator)
        aligned_ind = np.argwhere(match_indicator).ravel()
        # print(aligned_ind)
        # framewise_phns.append()
        if len(aligned_ind)==0:
            _phn = '[SIL]'
        elif len(aligned_ind)>1:
            Exception('Error found more than 1 matching phone index!')
        else:
            _phn = aligned_df.iloc[aligned_ind, 2].values[0]
            _phn = '[SIL]' if _phn=='sil' else _phn
            # print(_phn)
            aligned_phns.append(_phn)
    aligned_phn_idxs = [phone2id_dict[phone] for phone in aligned_phns]
    return np.array(aligned_phns), np.array(aligned_phn_idxs)

def phone_occurance_too_short(phone_idx, nframes, id2phone_dict=IND2PHONE, frame_step=.01, phone_minlen=.02):
    # frame_step
    phone = id2phone_dict[phone_idx]

    # phone_minlen = MIN_PHONE_LENGTH_DF[phone]
    phonelen = nframes * frame_step
    return phone_minlen>=phonelen

def _calc_GOP(logprobas, aligned_phn_idxs, sil_phone='[SIL]', phone_subset=None, ignore_short=True,
              phone2id_dict=PHONE2IND, id2phone_dict=IND2PHONE, frame_step=.01):
    sil_phn_idx = phone2id_dict[sil_phone]

    max_phn_idxs = np.argmax(logprobas, axis=1)
    framewise_gop = np.array(
        [logprobas[ii, align_phn_idx] - logprobas[ii, max_phn_idx] for ii, (align_phn_idx, max_phn_idx) in
         enumerate(zip(aligned_phn_idxs, max_phn_idxs))])
    unique_phone_idxs = np.unique(aligned_phn_idxs)


    def get_phonelen_from_idxs(phone_frame_idxs):
        '''
        gets the length in frames from each occurance of the phoneme

        Input: a sorted array of all the indexes at which the phoneme occurs

        output

        example:
        input [0, 1, 2, 3, 4, 99, 149, 150, 160, 170, 487, 488]
        output [5, 1, 4, 2]
        -------

        '''
        previdx = -1
        lengths_of_phone = []
        frame_length_count = 0
        for ii in range(len(phone_frame_idxs)):
            curridx = phone_frame_idxs[ii]
            if previdx == -1:
                frame_length_count += 1
            elif previdx + 1 == curridx:
                frame_length_count += 1
            else:
                lengths_of_phone.append(frame_length_count)
                frame_length_count = 1
            previdx = curridx

            if ii == len(phone_frame_idxs) - 1:
                lengths_of_phone.append(frame_length_count)

        return lengths_of_phone

    def filter_out_short_ids(phone_idx, phone_frame_idxs, lengths_of_phone):
        ind_pointer = 0
        keepinds = np.ones_like(phone_frame_idxs).astype('bool')
        for ii, phoneframelen in enumerate(lengths_of_phone):
            if phone_occurance_too_short(phone_idx, phoneframelen):
                keepinds[ind_pointer: ind_pointer + phoneframelen] = False
            ind_pointer += phoneframelen
        return phone_frame_idxs[keepinds]

    def get_gop_for_phone(phone_idx, framewise_gop, aligned_phn_idxs, ignore_short=False):
        phone_frame_idxs = np.argwhere(aligned_phn_idxs == phone_idx).ravel()

        if ignore_short:

            lengths_of_phone = get_phonelen_from_idxs(phone_frame_idxs)
            phone_frame_idxs = filter_out_short_ids(phone_idx, phone_frame_idxs, lengths_of_phone)

        phone_framewise_gop = framewise_gop[phone_frame_idxs]

                # length_of_phone.append()
        if len(phone_framewise_gop)==0:
            return np.nan, 0

        return np.mean(phone_framewise_gop), len(phone_frame_idxs)

    phonewise_gops = []
    phone_durations = []
    for _phone_id in unique_phone_idxs:
        if not _phone_id==sil_phn_idx:
            phone_name = id2phone_dict[_phone_id]
            phone_gop, phone_frameduration = get_gop_for_phone(phone_idx=_phone_id, framewise_gop=framewise_gop,
                                                               aligned_phn_idxs=aligned_phn_idxs,
                                                               ignore_short=ignore_short)
            phonewise_gops.append((phone_name, phone_gop, phone_frameduration))
            phone_durations.append(phone_frameduration)

    phonewise_gop_df = pd.DataFrame.from_records(phonewise_gops, columns=['Phone', 'GOP', 'NumFramesDuration'])
    duration_weights = np.array(phone_durations)/sum(phone_durations)
    nosil_idxs = np.argwhere(aligned_phn_idxs != sil_phn_idx).ravel()
    framewise_gop_nosil = framewise_gop[nosil_idxs]
    output = {'framewise_gop': framewise_gop_nosil, 'gopmean': np.mean(framewise_gop_nosil),
              'gop_withsil': np.nanmean(framewise_gop), 'phonewise_gop':phonewise_gop_df,
              'phonewise_gop_mean': np.nanmean(phonewise_gop_df['GOP'].mean()),
              'phonewise_gop_mean_weighted': np.average(list(phonewise_gop_df['GOP'].values), weights=duration_weights)}

    if phone_subset is not None:
        assert len(
            phone_subset) > 0, 'Cannot pass empty list for argument `phone_subset.` Must specify more than one phoneme in `phone_subset`'
        framewise_subset_gops = np.array([])
        for phone in phone_subset:
            _phn_idx = phone2id_dict[phone]
            phone_idxs = np.argwhere(aligned_phn_idxs == _phn_idx).ravel()
            framewise_subset_gops = np.concatenate([framewise_subset_gops, framewise_gop[
                phone_idxs]])  # don't need to use framewise_gops_nosil since we're filtering phones anyways
            output['framewise_gop_subset'] = framewise_subset_gops
            output['gopmean_subset'] = np.mean(framewise_subset_gops)
    return output


def get_aligner_frame_seq_len(audio_filepath, fs, charsiu):
    audio = charsiu.charsiu_processor.audio_preprocess(audio_filepath, sr=fs)
    audio = torch.Tensor(audio).unsqueeze(0).to(charsiu.device)
    print(len(audio))
    return charsiu.aligner._get_feat_extract_output_lengths(audio)


def calculate_GOP_e2e(audio, transcript, charsiu_model=CHARSIU_MODEL, textgrid_alignment=None, phone_subset=None,
                      TEMPERATURE=1, ignore_short=False):
    #TODO: provide an input alignment textgrid and hav the GOP compute using the provided alignment
    if type(audio) == str:
        audio_signal, fs = sf.read(audio)
    elif isinstance(audio, np.ndarray):
        audio_signal = audio

    USE_MANUAL_ALIGNMENT = False
    if textgrid_alignment is not None:
        #read in the textgrid and relabel the columns as required
        USE_MANUAL_ALIGNMENT = True
        # print('using textgrid')

        if type(textgrid_alignment) == str:
            print('here')
            input_alignment_df = extract_phone_df_from_textgrid(textgrid.openTextgrid(textgrid_alignment, False),
                                                                phone_key='phones', remove_numbers=True)
        elif type(textgrid_alignment) == pd.DataFrame:
            input_alignment_df = textgrid_alignment

    with torch.no_grad():
        aligned_phones, words, logits = charsiu_model.align(audio_signal, text=transcript, return_logits=True,
                                                            TEMPERATURE=TEMPERATURE)
        # aligned_phones = process_silences(aligned_phones, transcript)
        aligned_phones = [list(algn) + [algn[1] - algn[0]] for algn in aligned_phones]

        w2v_alignment_phone_df = pd.DataFrame.from_records(aligned_phones, columns=['start', 'end', 'phone', 'duration'])
        if USE_MANUAL_ALIGNMENT:
            # print(input_alignment_df)
            # print(input_alignment_df.iloc[:, 1])
            # print(input_alignment_df.iloc[:, 0])
            if 'duration' not in input_alignment_df.columns:
                input_alignment_df['duration'] = input_alignment_df['end'] - input_alignment_df['end']
            aligned_phone_df = input_alignment_df
        else:
            aligned_phone_df = w2v_alignment_phone_df

        seqlen = CHARSIU_MODEL.aligner._get_feat_extract_output_lengths(len(audio_signal))
        # print(seqlen)
        aligned_phns, aligned_phn_idxs = return_aligned_phns_and_ids(aligned_phone_df, seqlen=seqlen)
        logprobas = torch.log_softmax(logits, dim=-1).numpy()
        gopoutput = _calc_GOP(logprobas, aligned_phn_idxs, phone_subset=phone_subset,
                              ignore_short=ignore_short)
        return gopoutput['phonewise_gop'], gopoutput['phonewise_gop_mean']


if __name__ == "__main__":
    """
    test gop
    """
    audio_filepath = '/home/prad/datasets/als_at_home/als_at_home_audio_files/1002-20170522-1917-3.wav'
    transcript = 'Much more money must be donated to make this department succeed'
    per_phone_gop, gop = calculate_GOP_e2e(audio = audio_filepath, transcript = transcript, TEMPERATURE=1)
    print(per_phone_gop, '\t Normal phonewise gop\t')
    print('Normal mean gop', gop)

    per_phone_gop, gop = calculate_GOP_e2e(audio = audio_filepath, transcript = transcript, TEMPERATURE=1,
                                           ignore_short=True)
    print(per_phone_gop, '\t FilterShort phonewise gop\t')
    print('Filtershort mean gop', gop)

    per_phone_gop, gop = calculate_GOP_e2e(audio = audio_filepath, transcript = transcript, TEMPERATURE=2)
    print(gop, per_phone_gop, '\t Temp 2:\t')

    TRANSCRIPT = 'The chairman decided to pave over the shopping center garden'

    ''' Healthy file'''
    filepath = './data/artp_files/chairman_healthy.wav'
    print(os.path.isfile(filepath))
    _, ap1 = calculate_GOP_e2e(audio=filepath, transcript=TRANSCRIPT, charsiu_model=CHARSIU_MODEL)
    print('Healthy File1 Temp=1:\t', ap1)
    _, ap1 = calculate_GOP_e2e(audio=filepath, transcript=TRANSCRIPT, charsiu_model=CHARSIU_MODEL, TEMPERATURE=10)
    print('Healthy File1 Temp=.75:\t', ap1)

    ''' Healthy file older male'''
    filepath = './data/artp_files/chairman_healthy_older_male.wav'
    print(os.path.isfile(filepath))
    _, ap2 = calculate_GOP_e2e(audio=filepath, transcript=TRANSCRIPT, charsiu_model=CHARSIU_MODEL)
    print('Healthy File2 Older Temp=1:\t', ap2)
    _, ap2 = calculate_GOP_e2e(audio=filepath, transcript=TRANSCRIPT, charsiu_model=CHARSIU_MODEL, TEMPERATURE=10)
    print('Healthy File2 Older Temp=.75:\t', ap2)

    ''' ALS Speaker'''
    filepath = './data/artp_files/chairman_als.wav'
    print(os.path.isfile(filepath))
    _, ap3 = calculate_GOP_e2e(audio=filepath, transcript=TRANSCRIPT, charsiu_model=CHARSIU_MODEL)
    print('ALS File1 Temp=1:\t', ap3)
    _, ap3 = calculate_GOP_e2e(audio=filepath, transcript=TRANSCRIPT, charsiu_model=CHARSIU_MODEL, TEMPERATURE=10)
    print('ALS File1 Temp=.75:\t', ap3)


    ''' ALS Speaker'''
    print('Testing with provided textgrid')
    filepath = './data/artp_files/chairman_healthy.wav'
    textgrid_file = './data/artp_files/chairman_healthy.TextGrids'
    print(os.path.isfile(filepath))
    _, ap3 = calculate_GOP_e2e(audio=filepath, transcript=TRANSCRIPT, charsiu_model=CHARSIU_MODEL,
                               textgrid_alignment=textgrid_file)
    print('ALS File1 Temp=1:\t', ap3)
    _, ap3 = calculate_GOP_e2e(audio=filepath, transcript=TRANSCRIPT, charsiu_model=CHARSIU_MODEL,
                               textgrid_alignment=textgrid_file, TEMPERATURE=10)
    print('ALS File1 Temp=.75:\t', ap3)



