# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮\n(或直接选择已保存音色，无需上传prompt)',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮\n(或直接选择已保存音色，无需上传prompt)',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


# ---------- 音色保存/加载 ----------
# 音色特征与模型版本绑定(v2/v3 的 speech tokenizer 不同)，按模型名分子目录存储
def voice_store_dir():
    d = os.path.join(ROOT_DIR, 'voices', os.path.basename(os.path.normpath(args.model_dir)))
    os.makedirs(d, exist_ok=True)
    return d


def list_saved_voices():
    return sorted(x[:-3] for x in os.listdir(voice_store_dir()) if x.endswith('.pt'))


def voice_choices():
    return [('不使用', '')] + [(x, x) for x in list_saved_voices()]


def load_saved_voices():
    for name in list_saved_voices():
        cosyvoice.frontend.spk2info[name] = torch.load(os.path.join(voice_store_dir(), name + '.pt'),
                                                       map_location=cosyvoice.frontend.device)
        logging.info('loaded saved voice {}'.format(name))


def create_voice(voice_name, prompt_text, prompt_wav_upload, prompt_wav_record):
    prompt_wav = prompt_wav_upload if prompt_wav_upload is not None else prompt_wav_record
    voice_name = voice_name.strip()
    if voice_name == '' or '/' in voice_name or '\\' in voice_name:
        gr.Warning('请输入合法的音色名(不能为空或包含路径分隔符)')
        return gr.update()
    if prompt_wav is None:
        gr.Warning('请先上传或录制prompt音频')
        return gr.update()
    if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
        gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
        return gr.update()
    if prompt_text == '':
        gr.Warning('请输入prompt文本(与prompt音频内容一致)')
        return gr.update()
    if cosyvoice.__class__.__name__ == 'CosyVoice3' and '<|endofprompt|>' not in prompt_text:
        prompt_text = 'You are a helpful assistant.<|endofprompt|>' + prompt_text
    cosyvoice.add_zero_shot_spk(prompt_text, prompt_wav, voice_name)
    torch.save(cosyvoice.frontend.spk2info[voice_name], os.path.join(voice_store_dir(), voice_name + '.pt'))
    logging.info('voice {} saved to {}'.format(voice_name, voice_store_dir()))
    gr.Info('音色「{}」已保存，已在下拉框中选中'.format(voice_name))
    return gr.update(choices=voice_choices(), value=voice_name)


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed, seg_pause, saved_voice):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # 选中已保存音色时，复刻模式无需 prompt 音频/文本
    use_saved = saved_voice != '' and saved_voice is not None and mode_checkbox_group in ['3s极速复刻', '跨语种复刻']
    if use_saved and saved_voice not in cosyvoice.frontend.spk2info:
        gr.Warning('音色「{}」不存在，请重新创建'.format(saved_voice))
        yield (cosyvoice.sample_rate, default_data)
        return
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None and not use_saved:
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻'] and not use_saved:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
        if sft_dropdown == '':
            gr.Warning('没有可用的预训练音色！')
            yield (cosyvoice.sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '' and not use_saved:
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    # 优先使用已保存音色；否则把 prompt 音频特征注册为临时音色，逐行合成时复用，避免每行重复提取
    spk_id = ''
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if use_saved:
            spk_id = saved_voice
        elif prompt_wav is not None:
            spk_id = 'webui_prompt_spk'
            cosyvoice.add_zero_shot_spk(prompt_text, prompt_wav, spk_id)

    def line_inference(line):
        if mode_checkbox_group == '预训练音色':
            return cosyvoice.inference_sft(line, sft_dropdown, stream=stream, speed=speed)
        if mode_checkbox_group == '3s极速复刻':
            return cosyvoice.inference_zero_shot(line, '', '', zero_shot_spk_id=spk_id, stream=stream, speed=speed)
        if mode_checkbox_group == '跨语种复刻':
            return cosyvoice.inference_cross_lingual(line, '', zero_shot_spk_id=spk_id, stream=stream, speed=speed)
        return cosyvoice.inference_instruct(line, sft_dropdown, instruct_text, stream=stream, speed=speed)

    def trim_tail(audio):
        # 裁掉段尾低能量拖尾并淡出，消除拼接处的尾音
        sr = cosyvoice.sample_rate
        frame = int(0.02 * sr)
        end = len(audio)
        while end > frame and np.sqrt(np.mean(audio[end - frame:end] ** 2)) < 0.005:
            end -= frame
        audio = audio[:end]
        fade_len = min(int(0.1 * sr), len(audio))
        if fade_len > 0:
            audio[-fade_len:] = audio[-fade_len:] * np.linspace(1.0, 0.0, fade_len)
        return audio

    # 按行切分合成文本，每行独立合成，行间插入指定时长静音。
    # 行边界是真实的语义边界，流式/非流式下插静音都安全。
    # 引号是罕见字符，模型会在其位置发出填充元音(哎音)，合成前剥掉(保留英文撇号)
    # 弯撇号'保留，英文缩写(don't)会用到
    quote_table = str.maketrans('', '', '"“”‘「」『』')
    lines = [x.strip().translate(quote_table) for x in tts_text.split('\n') if x.strip() != '']
    silence = np.zeros(int(cosyvoice.sample_rate * seg_pause), dtype=np.float32)
    set_all_random_seed(seed)
    for line_idx, line in enumerate(lines):
        logging.info('get {} inference request, line {}/{}: {}'.format(mode_checkbox_group, line_idx + 1, len(lines), line))
        if line_idx > 0 and seg_pause > 0:
            yield (cosyvoice.sample_rate, silence)
        for i in line_inference(line):
            audio = i['tts_speech'].numpy().flatten()
            # 流式时 chunk 是句内片段，裁尾/淡出会造成断续，只在非流式下处理
            if stream is False:
                audio = trim_tail(audio)
            yield (cosyvoice.sample_rate, audio)


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(label="输入合成文本(按行切分，每行独立合成，行间插入停顿)", lines=5,
                              value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            seg_pause = gr.Number(value=0.5, label="行间停顿秒数", minimum=0.0, maximum=3.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        with gr.Row():
            saved_voice_dropdown = gr.Dropdown(choices=voice_choices(), value='',
                                               label='选择已保存音色(3s极速复刻/跨语种复刻可用，选中后无需上传prompt)')
            voice_name = gr.Textbox(label='音色名', placeholder='用上方prompt音频+prompt文本创建音色，重名覆盖')
            create_voice_button = gr.Button('创建/覆盖音色')

        generate_button = gr.Button("生成音频")

        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        create_voice_button.click(create_voice,
                                  inputs=[voice_name, prompt_text, prompt_wav_upload, prompt_wav_record],
                                  outputs=[saved_voice_dropdown])
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed, seg_pause, saved_voice_dropdown],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    # 注意先取预训练音色列表，再加载本地保存的克隆音色，避免混入预训练音色下拉框
    sft_spk = cosyvoice.list_available_spks()
    load_saved_voices()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
