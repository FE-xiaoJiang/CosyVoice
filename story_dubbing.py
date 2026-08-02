# 分角色故事配音脚本
#
# 用法:
#   conda run --no-capture-output -n cosyvoice python story_dubbing.py \
#       --model_dir pretrained_models/Fun-CosyVoice3-0.5B \
#       --voices dubbing_example/voices.json \
#       --story dubbing_example/story.txt \
#       --output output/story.wav
#
# 剧本格式(story.txt): 每行 "角色：台词"，全角/半角冒号均可，空行和 # 开头的行忽略。
# 音色配置(voices.json):
#   {
#     "旁白": {"prompt_wav": "asset/zero_shot_prompt.wav", "prompt_text": "希望你以后能够做的比我还好呦。"},
#     "老和尚": {"prompt_wav": "voices/monk.wav", "prompt_text": "...", "instruct": "请用低沉平缓的语气说。", "speed": 0.9}
#   }
#   prompt_wav: 该角色的参考音频(5~15s, 采样率>=16k)
#   prompt_text: 参考音频的逐字转写
#   instruct: 可选，风格指令(方言/情感/语速等)，走 instruct2 推理
#   speed: 可选，语速倍率(0.5~2.0, 仅非流式)
import argparse
import json
import re
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging

ENDOFPROMPT = '<|endofprompt|>'
DEFAULT_SYSTEM = 'You are a helpful assistant.'
LINE_RE = re.compile(r'^\s*([^：:]+?)\s*[：:]\s*(.+)$')


def parse_story(path):
    segments = []
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = LINE_RE.match(line)
            if m is None:
                raise ValueError('第{}行格式错误，应为 "角色：台词": {}'.format(lineno, line))
            segments.append((m.group(1), m.group(2)))
    return segments


def main():
    parser = argparse.ArgumentParser(description='多角色故事配音')
    parser.add_argument('--model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--voices', required=True, help='角色音色配置 json')
    parser.add_argument('--story', required=True, help='剧本 txt，每行 角色：台词')
    parser.add_argument('--output', default='output/story.wav', help='输出音频路径')
    parser.add_argument('--pause', type=float, default=0.4, help='段间停顿秒数')
    parser.add_argument('--save_segments', action='store_true', help='同时保存每段音频')
    args = parser.parse_args()

    with open(args.voices, encoding='utf-8') as f:
        voices = json.load(f)
    segments = parse_story(args.story)

    missing = {role for role, _ in segments} - set(voices.keys())
    if missing:
        raise ValueError('剧本中的角色未在音色配置中定义: {}'.format('、'.join(missing)))

    cosyvoice = AutoModel(model_dir=args.model_dir)
    is_v3 = cosyvoice.__class__.__name__ == 'CosyVoice3'

    # 注册各角色音色，注册后合成时无需重复提取参考音频特征
    for role, cfg in voices.items():
        prompt_text = cfg['prompt_text']
        if is_v3 and ENDOFPROMPT not in prompt_text:
            prompt_text = DEFAULT_SYSTEM + ENDOFPROMPT + prompt_text
        assert cosyvoice.add_zero_shot_spk(prompt_text, cfg['prompt_wav'], role) is True
        logging.info('已注册角色音色: {}'.format(role))

    sr = cosyvoice.sample_rate
    silence = torch.zeros(1, int(sr * args.pause))
    pieces = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for idx, (role, text) in enumerate(segments):
        cfg = voices[role]
        speed = cfg.get('speed', 1.0)
        instruct = cfg.get('instruct')
        logging.info('[{}/{}] {}: {}'.format(idx + 1, len(segments), role, text))
        if instruct:
            if is_v3 and ENDOFPROMPT not in instruct:
                instruct = DEFAULT_SYSTEM + ' ' + instruct + ENDOFPROMPT
            outputs = cosyvoice.inference_instruct2(text, instruct, cfg['prompt_wav'], stream=False, speed=speed)
        else:
            outputs = cosyvoice.inference_zero_shot(text, '', '', zero_shot_spk_id=role, stream=False, speed=speed)
        speech = torch.concat([o['tts_speech'] for o in outputs], dim=1)
        if args.save_segments:
            seg_path = '{}.seg{:03d}.{}.wav'.format(os.path.splitext(args.output)[0], idx, role)
            torchaudio.save(seg_path, speech, sr)
        pieces.append(speech)
        pieces.append(silence)

    final = torch.concat(pieces[:-1], dim=1)
    torchaudio.save(args.output, final, sr)
    logging.info('完成: {} (时长 {:.1f}s, 共 {} 段)'.format(args.output, final.shape[1] / sr, len(segments)))


if __name__ == '__main__':
    main()
