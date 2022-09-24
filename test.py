from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Config
from src.models import Wav2Vec2ForFrameClassification
import torch
import numpy as np

tok = Wav2Vec2CTCTokenizer.from_pretrained('charsiu/tokenizer_en_cmu')
model = Wav2Vec2ForFrameClassification.from_pretrained('charsiu/en_w2v2_fc_10ms')
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tok)
model.config.gradient_checkpointing = True,
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = len(processor.tokenizer)
model.config.ivector_size = 64
import pdb
pdb.set_trace()
print('')

