
from alignment_helper_fns import *
from experiments.common_voice_pretraining import *
from alignment_helper_fns import *
import argparse
#TODO: repackage this into a wav2vec2 training script

parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', default='/home/prad/datasets/ChildSpeechDataset/child_speech_16_khz')
parser.add_argument('--manual_textgrids_dir', default='/home/prad/datasets/ChildSpechDataset/manually-aligned-text-grids/')
parser.add_argument('--model_output_dir', default='./child_speech_pretrained')
parser.add_argument('--tokenizer_name', default='charsiu/tokenizer_en_cmu')
parser.add_argument('--model_name', default='charsiu/en_w2v2_fc_10ms')
parser.add_argument('--device', default='cuda')
# parser.add_argument('--')
args = parser.parse_args()
audio_dir = args.audio_dir
manual_textgrids_dir = args.manual_textgrids_dir
output_dir = args.model_output_dir
device = args.device
tokenizer_name = args.tokenizer_name
model_name = args.model_name

# if __name__ == "__main__":

unmatched_manual_textgrid_files = get_all_textgrids_in_directory(manual_textgrids_dir)

manual_textgrid_files = []
audio_files = ['/'.join([audio_dir,_path.split('/')[-2], _path.split('/')[-1][:-8]+'wav'])
               for _path in unmatched_manual_textgrid_files]

from create_child_speech_dataset import *
csd = ChildSpeechDataset(audio_paths=audio_files)
data_prepared = csd.return_as_datsets()

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_name)

# tokenizer = Wav2Vec2CTCTokenizer('~/datasets/ChildSpeechDataset/huggingface_dataset')
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                             do_normalize=True, return_attention_mask=False, device=device)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

data_collator = DataCollatorWithPadding(processor=processor, padding=True)

config = Wav2Vec2Config()
config.num_attention_heads = 6
config.hidden_size = 384
config.num_hidden_layers = 6
config.num_negatives = 20
config.device = device
model = Wav2Vec2ForPreTraining(config).to(device) #, device=device)

pre_trained_model = Wav2Vec2ForPreTraining.from_pretrained(model_name, device=device)

layers = {'wav2vec2.feature_extractor.conv_layers.0.conv.weight',
          'wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight',
          'wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias',
          'wav2vec2.feature_extractor.conv_layers.1.conv.weight',
          'wav2vec2.feature_extractor.conv_layers.2.conv.weight',
          'wav2vec2.feature_extractor.conv_layers.3.conv.weight',
          'wav2vec2.feature_extractor.conv_layers.4.conv.weight',
          'wav2vec2.feature_extractor.conv_layers.5.conv.weight',
          'wav2vec2.feature_extractor.conv_layers.6.conv.weight', 'quantizer.codevectors',
          'quantizer.weight_proj.weight', 'quantizer.weight_proj.bias', 'project_q.weight', 'project_q.bias'}
pretrained_dict = {k: v for k, v in pre_trained_model.state_dict().items() if k in layers}
print('Loaded Wav2Vec2 English')

training_args = TrainingArguments(output_dir=output_dir,
                                  group_by_length=True,
                                  per_device_train_batch_size=4,
                                  gradient_accumulation_steps=40,
#                                      evaluation_strategy="steps",
                                  num_train_epochs=4,
                                  fp16=True,
                                  save_steps=1000,
#                                      eval_steps=1000,
                                  logging_steps=1000,
                                  learning_rate=5e-4,
                                  weight_decay=1e-6,
                                  warmup_steps=1000,
                                  save_total_limit=2,
                                  ignore_data_skip=True,
                                 )

trainer = Trainer(
                    model=model,
                    data_collator=data_collator,
                    args=training_args,
#                            compute_metrics=compute_metrics,
                    train_dataset=data_prepared,
#                            eval_dataset=libris_train_prepared,
                    tokenizer=processor.feature_extractor,
                    )

trainer.train()

