import torch
import pandas as pd
import numpy as np
import os
from gop_helper_fns import *


''' Compute mean artp for each session'''
STEP_TO_TRANSCRIPT = {
'3': 'Much more money must be donated to make this department succeed',
'4': 'In this famous coffee shop they serve the best donuts in town',
'5': 'The chairman decided to pave over the shopping center garden', 
'6': 'The standards committee met this afternoon in an open meeting'}

alsathome_dir = '/home/prad/datasets/als_at_home'
als_datadir = os.path.join(alsathome_dir, 'als_at_home_audio_files')
alsathome_df = pd.read_csv(os.path.join(alsathome_dir, 'als_at_home_primary_covariates_existing_files_only.csv'))


''' Run all sessionIDs'''
import time
sessionIDs = alsathome_df['session'].values
steps = list(STEP_TO_TRANSCRIPT.keys())

step_artp_dict = {}
for step in steps:
    step_artp_dict[step] = np.zeros_like(sessionIDs)

step3artp = np.zeros_like(sessionIDs)
for ii, sessid in enumerate(tqdm.tqdm(sessionIDs)):    

    for step in steps:
        step_filepath = os.path.join(als_datadir, sessid) + '-%s.wav' % step
        TRANSCRIPT = STEP_TO_TRANSCRIPT[step]
        try:
            step_artp = calculate_GOP_e2e(audio_filepath=step_filepath, transcript=TRANSCRIPT)
            step_artp_dict[step][ii] = step_artp
            time.sleep(.05)
        except:
            step_artp_dict[step][ii] = np.nan
            print('Exception for:\t', step_filepath)



