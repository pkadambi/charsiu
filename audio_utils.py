from playsound import playsound
import os

def play_audio_given_textgrid(textgridpath,
                              child_speech_dir='/home/prad/datasets/ChildSpeechDataset/child_speech_16_khz/'):
    '''
    This will only work if the textgrid is in format .../speaker/file.TextGrid
    :textgridpath:

    Returns
    -------

    '''
    speaker = textgridpath.split('/')[-2]
    filename = textgridpath.split('/')[-1][:-8]
    audiopath = os.path.join(child_speech_dir, speaker, filename) + 'wav'
    print('Playing file: ', audiopath)
    playsound(audiopath)

# play_audio_given_textgrid('/home/prad/datasets/ChildSpeechDataset/manually-aligned-text-grids/0407_M_SJ/0407_M_SJwT40.TextGrid')
