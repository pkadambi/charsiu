from src.Charsiu import charsiu_forced_aligner
from alignment_helper_fns import *
from scipy.io import wavfile
import torch
import soundfile as sf

#TODO: re-implement as an ARTP class that will take the files and
if not torch.cuda.is_available():
    DEVICE='cpu'
else:
    DEVICE='cuda'
CHARSIU_MODEL = charsiu_forced_aligner('charsiu/en_w2v2_fc_10ms', device=DEVICE)
PHONE2IND = CHARSIU_MODEL.charsiu_processor.processor.tokenizer.encoder
IND2PHONE = CHARSIU_MODEL.charsiu_processor.processor.tokenizer.decoder

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


def return_aligned_phns_and_ids(aligned_df, seqlen, phone2id_dict=PHONE2IND, id2phone_dict=IND2PHONE):
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


def _calc_GOP(logprobas, aligned_phn_idxs, sil_phone='[SIL]', phone_subset=None, ignore_sil=True,
              phone2id_dict=PHONE2IND, id2phone_dict=IND2PHONE):
    sil_phn_idx = phone2id_dict[sil_phone]

    max_phn_idxs = np.argmax(logprobas, axis=1)
    framewise_gop = np.array(
        [logprobas[ii, align_phn_idx] - logprobas[ii, max_phn_idx] for ii, (align_phn_idx, max_phn_idx) in
         enumerate(zip(aligned_phn_idxs, max_phn_idxs))])
    unique_phone_idxs = np.unique(aligned_phn_idxs)

    def get_gop_for_phone(phone_idx, framewise_gop, aligned_phn_idxs):
        phone_frame_idxs = np.argwhere(aligned_phn_idxs == phone_idx).ravel()
        phone_framewise_gop = framewise_gop[phone_frame_idxs]
        return np.mean(phone_framewise_gop)

    phonewise_gops = []
    for _phone_id in unique_phone_idxs:
        if not _phone_id==sil_phn_idx:
            phone_name = id2phone_dict[_phone_id]
            phone_gop = get_gop_for_phone(phone_idx=_phone_id,
                                          framewise_gop=framewise_gop, aligned_phn_idxs=aligned_phn_idxs)
            phonewise_gops.append((phone_name, phone_gop))

    phonewise_gop_df = pd.DataFrame.from_records(phonewise_gops, columns=['Phone', 'GOP'])

    nosil_idxs = np.argwhere(aligned_phn_idxs != sil_phn_idx).ravel()
    framewise_gop_nosil = framewise_gop[nosil_idxs]
    output = {'framewise_gop': framewise_gop_nosil, 'gopmean': np.mean(framewise_gop_nosil),
              'gop_withsil': np.mean(framewise_gop), 'phonewise_gop':phonewise_gop_df,
              'phonewise_gop_mean': np.mean(phonewise_gop_df['GOP'].mean())}

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


# TODO implement for phone subset

def get_aligner_frame_seq_len(audio_filepath, fs, charsiu):
    audio = charsiu.charsiu_processor.audio_preprocess(audio_filepath, sr=fs)
    audio = torch.Tensor(audio).unsqueeze(0).to(charsiu.device)
    print(len(audio))
    return charsiu.aligner._get_feat_extract_output_lengths(audio)


def calculate_GOP_e2e(audio, transcript, charsiu_model=CHARSIU_MODEL, aligned_phones=None, phone_subset=None):

    if type(audio) == str:
        audio_signal, fs = sf.read(audio)
    elif isinstance(audio, np.ndarray):
        audio_signal = audio

    with torch.no_grad():
        phones, words, logits = charsiu_model.align(audio_signal, text=transcript, return_logits=True)
        aligned_phones = aligned_phones if aligned_phones is not None else phones
        aligned_phone_df = pd.DataFrame.from_records(aligned_phones, columns=['start', 'end', 'phone'])
        seqlen = CHARSIU_MODEL.aligner._get_feat_extract_output_lengths(len(audio_signal))
        # print(seqlen)
        aligned_phns, aligned_phn_idxs = return_aligned_phns_and_ids(aligned_phone_df, seqlen=seqlen)
        logprobas = torch.log_softmax(logits, dim=-1).numpy()
        gopoutput = _calc_GOP(logprobas, aligned_phn_idxs, phone_subset=phone_subset)
        return gopoutput['phonewise_gop_mean'], gopoutput['phonewise_gop']

if __name__ == "__main__":
    """
    test gop
    """
    audio_filepath = '/home/prad/datasets/als_at_home/als_at_home_audio_files/1002-20170522-1917-3.wav'
    transcript = 'Much more money must be donated to make this department succeed'
    gop, per_phone_gop = calculate_GOP_e2e(audio_filepath = audio_filepath, transcript = transcript)
    print(gop, per_phone_gop)





