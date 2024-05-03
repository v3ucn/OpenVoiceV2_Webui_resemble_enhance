import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter,BaseSpeakerTTS

from melo.api import TTS

ckpt_converter = './checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = './outputs_v2'

ckpt_base = './checkpoints_v2/base_speakers/EN'
ckpt_converter = './checkpoints_v2/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs'

base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

