
import tqdm
import torch
import librosa
import pandas as pd
from alignment_helper_fns import *
from datasets import Dataset, DatasetDict
from src.utils import word2textgrid
# librosa.core.load(path,sr=16000)


class PhonationDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths: list, lables_df: pd.DataFrame, textgrids_dir: str):
        self.textgrids_dir = textgrids_dir
        self.audio_paths = audio_paths
        self.label_df = lables_df
        # Extract Audios
        print('Loading audio files')
        self.audios = self.extract_audio()
        self.audio_lens = [len(_audio) for _audio in self.audios]
        # Extract speaker ids and file ids
        self.speaker_ids = [self.speaker_id_from_tgpath(_path) for _path in self.audio_paths]
        # for _path in self.audio_paths:
        #     if '5022' in _path:
        #         self.speaker_ids.append('-'.join(_path.split('/')[-1].split('-')[:3]))
        #     else:
        #         self.speaker_ids.append(_path.split('/')[-1].split('-')[0])

        self.sessionids = [_path.split('/')[-1].split('.')[0] for _path in self.audio_paths]

        #Step 1: extract word transcripts
        print('\nExtracting transcripts, phone and word bounds')
        # self.text_transcripts = [self.extract_text_transcript(_path) for _path in tqdm.tqdm(self.audio_paths)]
        self.text_transcripts = ['ah' for _path in tqdm.tqdm(self.audio_paths)]

        #Step 2: extract phone transcripts/alignments
        self.phone_alignments = self.extract_phone_bounds()
        self.word_alignments = self.phone_alignments

        #step3: extract frame labels
        print('Extracting Framewise Labels')
        self.frame_phn_labels, self.frame_times = self.gen_frame_labels_from_alignment()
        # assumes that the directory is in the file

    @staticmethod
    def speaker_id_from_tgpath(tg_path):
        # for _path in tg_path:
        if '5022' in tg_path:
            return '-'.join(tg_path.split('/')[-1].split('-')[:3])
        else:
            return tg_path.split('/')[-1].split('-')[0]
    def extract_audio(self):
        return [librosa.load(_audio_path, sr=16000)[0] for _audio_path in tqdm.tqdm(self.audio_paths)]

    def extract_phone_bounds(self, sessid_key='sessionStepId', write_textgrids=True):
        phone_alignments = []
        # phone_df = pd.DataFrame(columns=['start', 'stop', 'utterance'])

        for ii, sessid in enumerate(self.sessionids):
            row = self.label_df[self.label_df[sessid_key]==sessid]
            start_time = float(row['start_time_groundtruth'])
            end_time = float(row['end_time_groundtruth'])

            start_times = []
            end_times = []
            phones = []
            if start_time>0:
                start_times.append(0)
                end_times.append(start_time)
                phones.append('[SIL]')

            start_times.append(start_time)
            end_times.append(end_time)
            phones.append('AA')

            if end_time<30:
                start_times.append(end_time)
                end_times.append(30)
                phones.append('[SIL]')

            phonedict = {'start': start_times,
                         'stop': end_times,
                         'utterance': phones}
            phone_alignments.append(phonedict)
            if write_textgrids:
                phone_df = pd.DataFrame(phonedict)
                pre_tg_list = list(phone_df.to_records(index=False))
                tg = textgrid.Textgrid()
                audio_path = self.audio_paths[ii]
                textgrid_filename =  audio_path.split('/')[-1][:-3]+'TextGrid'
                textgrid_full_filepath = os.path.join(self.textgrids_dir, textgrid_filename)
                phoneTier = textgrid.IntervalTier('phones', pre_tg_list, 0, pre_tg_list[-1][1])
                tg.addTier(phoneTier)
                # print(pre_tg_list)
                tg.save(textgrid_full_filepath, format="short_textgrid", includeBlankSpaces=False)
        return phone_alignments

    def gen_frame_labels_from_alignment(self):
        '''
        RUN AFTER 'extract_phone_words_bounds'

        Returns
        -------

        '''
        frame_phn_labels = []
        frame_times = []
        w2v2_time_step = .02001/2

        def _find_framewise_phn_labels(timestep, phn_dict):
            framewise_phn_labels = []
            for jj, _time in enumerate(timestep):
                gt_start_indicator = _time>np.array(phn_dict['start'])
                lt_end_indicator = _time<np.array(phn_dict['stop'])
                match_indicator = np.logical_and(gt_start_indicator, lt_end_indicator)
                phone_ind = np.argwhere(match_indicator).ravel()
                if len(phone_ind)==0:
                    _phn='[SIL]'
                    # np.concatenate([np.array(phn_dict['start']).reshape(-1, 1), np.array(phn_dict['stop']).reshape(-1, 1), np.array(phn_dict['utterance']).reshape(-1, 1)], axis=1)
                elif len(phone_ind)>1:
                    Exception('Error found more than 1 matching phone index!\n Manual Transcript dictionary:\n', phn_dict)
                else:
                    _phn = phn_dict['utterance'][phone_ind[0]]
                framewise_phn_labels.append(_phn)
            return framewise_phn_labels

        for ii, _path in enumerate(tqdm.tqdm(self.audio_paths)):
            _audiolen = len(self.audios[ii])/16000
            timesteps = np.arange(0, _audiolen, step=w2v2_time_step)[1:] #No embedding for t=0
            _phn_dct = self.phone_alignments[ii]

            frame_times.append(timesteps)
            frame_phn_labels.append(_find_framewise_phn_labels(timesteps, _phn_dct))

            # phone_df = textgridpath_to_phonedf(manual_tg_path, phone_key='ha phones', remove_numbers=True)

        return frame_phn_labels, frame_times

    def return_as_dict(self):
        dct = DatasetDict({'file': self.audio_paths, 'audio':self.audios, 'phone_alignments': self.phone_alignments,
               'word_alignments': self.word_alignments, 'speaker_id': self.speaker_ids, 'audio_len': self.audio_lens,
                           'sessionid': self.sessionids, 'sentence': self.text_transcripts, 'frame_phones': self.frame_phn_labels,
                           'frame_times': self.frame_times})
        return dct

    def return_as_datsets(self):
        self.dct = self.return_as_dict()
        self.dataset = Dataset.from_dict(self.dct)
        return self.dataset

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.audio_paths)
