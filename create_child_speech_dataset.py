
import tqdm
import torch
import librosa
from alignment_helper_fns import *
from datasets import Dataset
# librosa.core.load(path,sr=16000)


class ChildSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths: list):
        self.audio_paths = audio_paths

        # Extract Audios
        print('Loading audio files')
        self.audios = self.extract_audio()

        # Extract speaker ids and file ids
        self.speaker_ids = [_path.split('/')[-2] for _path in self.audio_paths]
        self.ids = [_path.split('/')[-1].split('.')[0] for _path in self.audio_paths]

        #Step 1: extract word transcripts
        print('\nExtracting transcripts, phone and word bounds')
        self.text_transcripts = [self.extract_text_transcript(_path) for _path in tqdm.tqdm(self.audio_paths)]

        #Step 2: extract phone transcripts/alignments
        self.phone_alignments, self.word_alignments = self.extract_phone_words_bounds()

        # assumes that the directory is in the file
    def extract_text_transcript(self, path):
        transcript_path = path.split('.')[0] + '.lab'
        f = open(transcript_path, 'rb')
        transcript = f.read()
        return transcript

    def extract_audio(self):
        return [librosa.load(_audio_path, sr=16000)[0] for _audio_path in tqdm.tqdm(self.audio_paths)]

    def extract_phone_transcripts(self):
        transcripts = []
        for _path in self.audio_paths:
            _transcript_path = _path.split('.') + 'lab'
            f = open(_transcript_path, 'rb')
            transcripts.append(f.read())
            f.close()
        return transcripts

    def extract_phone_words_bounds(self):
        phone_alignments = []
        word_alignments = []
        phones = []
        for _path in tqdm.tqdm(self.audio_paths):
            # _tgpath = _path[:-3] + 'TextGrid'
            tgfilename = _path.split('/')[-1][:-3] + 'TextGrid'
            splitpath = _path.split('/')[:5]
            speaker_dir = _path.split('/')[-2]
            splitpath.append('manually-aligned-text-grids')
            splitpath.append(speaker_dir)
            splitpath.append(tgfilename)
            manual_tg_path = '/'.join(splitpath)
            phone_df = textgridpath_to_phonedf(manual_tg_path, phone_key='ha phones', remove_numbers=True)
            words_df = textgridpath_to_phonedf(manual_tg_path, phone_key='ha words', remove_numbers=True)
            phone_df = phone_df.replace('sil', '[SIL]')
            phonedict = {'start': list(phone_df.iloc[:,0]),
                         'stop':list(phone_df.iloc[:,1]),
                         'utterance': list(phone_df.iloc[:,2])}

            worddict = {'start': list(words_df.iloc[:,0]),
                         'stop': list(words_df.iloc[:,1]),
                         'utterance': list(words_df.iloc[:,2])}
            phone_alignments.append(phonedict)
            word_alignments.append(worddict)
            phones.extend(list(phone_df.iloc[:, 2]))
            self.unique_phones = list(np.unique(phones))
        return phone_alignments, word_alignments

    def return_as_dict(self):
        dct = {'file': self.audio_paths, 'phone_alignments': self.phone_alignments,
               'word_alignments': self.word_alignments, 'speaker_id': self.speaker_ids, 'id': self.ids, 'sentence': self.text_transcripts}
        return dct

    def return_as_datsets(self):
        self.dct = self.return_as_dict()
        self.dataset = Dataset.from_dict(self.dct)
        return self.dataset

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.audio_paths)
