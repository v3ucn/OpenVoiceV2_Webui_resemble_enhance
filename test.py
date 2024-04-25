import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

from melo.api import TTS

ckpt_converter = './checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = './outputs_v2'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

reference_speaker = './gakki.wav' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)


src_path = f'{output_dir}/tmp.wav'


texts = {
    # 'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
    # 'EN': "Did you ever hear a folk tale about a giant turtle?",
    # 'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    # 'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    'ZH': "想成为股票经纪人有两个诀窍，首先你得放松，你撸管儿吗？一周多少次？您这频率只能算是个菜鸟，我个人来讲，一天至少两次，早上来一发，吃完午饭再来一发。我这么玩儿不是因为我想，而是因为我需要，这很重要。想想吧，你和数字打交道，那些该死的数字，你开始不堪重负，太让人抓狂了不是吗？你必须多来几发，促进血液循环，必须保持好胯下的节奏，这不是什么建议，这是疗法，相信我，否则你会失去平衡的，差之毫厘，你就完蛋了，还有更惨的，我亲眼见过有人猝死，猝死是最操蛋的了，所以只要有空就去厕所来一发，等你熟练了，你撸的时候，脑子里想的都是钱。",
    # 'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
    # 'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
}

# Speed is adjustable
speed = 1.0

for language, text in texts.items():
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        source_se = torch.load(f'./checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
        save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)

