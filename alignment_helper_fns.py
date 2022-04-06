import pandas as pd
import numpy as np
from praatio.data_classes.textgrid import Textgrid
from praatio import textgrid
import re
import os

def get_all_textgrids_in_directory(directory):
    textgrid_files = []
    for ii, (path, subdirs, files) in enumerate(os.walk(directory)):
        for name in files:
            if 'TextGrid' in name:
                _textgridfile = os.path.join(path, name)
                textgrid_files.append(_textgridfile)
    return textgrid_files

def textgridpath_to_phonedf(txtgrid_path: str, phone_key: str, remove_numbers=False):
    '''
    txtgrid_path - the path to the textgridfile
    phone_key - the key in the textgrid for the phoneme column
    '''
    txtgrid = textgrid.openTextgrid(txtgrid_path, False)
    return extract_phone_df_from_textgrid(txtgrid=txtgrid, phone_key=phone_key, remove_numbers=remove_numbers)


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

