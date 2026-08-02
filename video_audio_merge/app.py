# 视频多轨剪辑工具（简易合并 + 剪映风格多轨）
#
# 用法:
#   conda run --no-capture-output -n cosyvoice python video_audio_merge/app.py --port 7861
import argparse
import os

import gradio as gr

from composer import (
    AudioTrack, SubtitleTrack, TimelineProject, VideoTrack,
    compose, simple_merge,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_FORMATS = ['mp4', 'webm', 'mov', 'wav']


def do_simple_merge(video_file, audio_file, export_format):
    if video_file is None:
        raise gr.Error('请先上传视频文件')
    if audio_file is None:
        raise gr.Error('请先上传音频文件')
    video_path = video_file if isinstance(video_file, str) else getattr(video_file, 'name', video_file)
    audio_path = audio_file if isinstance(audio_file, str) else getattr(audio_file, 'name', audio_file)
    try:
        preview, download, info = simple_merge(video_path, audio_path, export_format)
    except Exception as e:
        raise gr.Error(str(e))
    return preview, download, info


def do_timeline_compose(
    video_file, video_loop, mute_original, video_volume,
    voice_file, voice_start, voice_volume,
    bgm_file, bgm_start, bgm_volume, bgm_loop, bgm_fade_in, bgm_fade_out,
    subtitle_file, subtitle_text, subtitle_enable, font_size, margin_v,
    timeline_duration, export_format, width, height,
):
    video = None
    if video_file is not None:
        vpath = video_file if isinstance(video_file, str) else getattr(video_file, 'name', video_file)
        video = VideoTrack(
            path=vpath,
            loop=bool(video_loop),
            mute_original=bool(mute_original),
            volume=float(video_volume),
        )

    voice = None
    if voice_file is not None:
        apath = voice_file if isinstance(voice_file, str) else getattr(voice_file, 'name', voice_file)
        voice = AudioTrack(
            path=apath,
            start=float(voice_start or 0),
            volume=float(voice_volume),
            name='voice',
        )

    bgm = None
    if bgm_file is not None:
        bpath = bgm_file if isinstance(bgm_file, str) else getattr(bgm_file, 'name', bgm_file)
        bgm = AudioTrack(
            path=bpath,
            start=float(bgm_start or 0),
            volume=float(bgm_volume),
            fade_in=float(bgm_fade_in or 0),
            fade_out=float(bgm_fade_out or 0),
            loop=bool(bgm_loop),
            name='bgm',
        )

    subtitle = None
    srt_path = None
    if subtitle_file is not None:
        srt_path = subtitle_file if isinstance(subtitle_file, str) else getattr(subtitle_file, 'name', subtitle_file)
    if subtitle_enable and (srt_path or (subtitle_text and subtitle_text.strip())):
        subtitle = SubtitleTrack(
            srt_path=srt_path,
            srt_text=subtitle_text or '',
            font_size=int(font_size),
            margin_v=int(margin_v),
            enabled=True,
        )

    if video is None and voice is None and bgm is None:
        raise gr.Error('请至少上传：视频 / 人声音频 / 背景音乐 其中之一')

    project = TimelineProject(
        video=video,
        voice=voice,
        bgm=bgm,
        subtitle=subtitle,
        duration=float(timeline_duration or 0),
        export_format=export_format,
        width=int(width),
        height=int(height),
    )
    try:
        preview, download, info = compose(project)
    except Exception as e:
        raise gr.Error(str(e))
    return preview, download, info


def build_ui():
    with gr.Blocks(title='多轨视频剪辑') as demo:
        gr.Markdown('## 多轨视频剪辑（简易合并 / 剪映风格多轨）')
        gr.Markdown(
            '支持视频轨、人声轨、背景音乐轨、字幕轨混剪导出。'
            '时间轴默认按人声时长对齐；视频/BGM 可循环铺满。'
        )

        with gr.Tabs():
            # -------- Tab 1: 简易合并（兼容旧功能） --------
            with gr.Tab('简易合并'):
                gr.Markdown('视频按音频长度循环拼接后合成（原逻辑）。')
                with gr.Row():
                    s_video = gr.Video(label='视频', sources=['upload'])
                    s_audio = gr.Audio(label='音频', type='filepath', sources=['upload'])
                with gr.Row():
                    s_fmt = gr.Dropdown(choices=EXPORT_FORMATS, value='mp4', label='导出格式')
                    s_btn = gr.Button('开始合并', variant='primary')
                s_info = gr.Textbox(label='处理信息', interactive=False)
                with gr.Row():
                    s_preview = gr.Video(label='预览')
                    s_download = gr.File(label='下载')
                s_btn.click(
                    fn=do_simple_merge,
                    inputs=[s_video, s_audio, s_fmt],
                    outputs=[s_preview, s_download, s_info],
                )

            # -------- Tab 2: 多轨编辑 --------
            with gr.Tab('多轨编辑'):
                gr.Markdown(
                    '### 轨道说明\n'
                    '- **视频轨**：画面；可循环铺满时间轴，可静音原片\n'
                    '- **人声轨**：旁白/配音主音轨（默认决定时间轴长度）\n'
                    '- **BGM 轨**：背景音乐，可调音量、淡入淡出、循环\n'
                    '- **字幕轨**：上传 `.srt` 或粘贴字幕文本（纯文本会按每行 3 秒自动生成时间轴）'
                )

                with gr.Accordion('🎬 视频轨', open=True):
                    with gr.Row():
                        t_video = gr.Video(label='视频素材', sources=['upload'])
                    with gr.Row():
                        t_video_loop = gr.Checkbox(value=True, label='循环铺满时间轴')
                        t_mute = gr.Checkbox(value=True, label='静音原片音轨')
                        t_video_vol = gr.Slider(0, 2, value=1.0, step=0.05, label='原片音量（未静音时生效）')

                with gr.Accordion('🎙️ 人声 / 旁白轨', open=True):
                    with gr.Row():
                        t_voice = gr.Audio(label='人声音频', type='filepath', sources=['upload'])
                    with gr.Row():
                        t_voice_start = gr.Number(value=0, label='入点偏移(秒)', minimum=0)
                        t_voice_vol = gr.Slider(0, 2, value=1.0, step=0.05, label='音量')

                with gr.Accordion('🎵 背景音乐轨', open=True):
                    with gr.Row():
                        t_bgm = gr.Audio(label='BGM', type='filepath', sources=['upload'])
                    with gr.Row():
                        t_bgm_start = gr.Number(value=0, label='入点偏移(秒)', minimum=0)
                        t_bgm_vol = gr.Slider(0, 2, value=0.35, step=0.05, label='音量')
                        t_bgm_loop = gr.Checkbox(value=True, label='循环铺满')
                    with gr.Row():
                        t_bgm_fade_in = gr.Number(value=1.0, label='淡入(秒)', minimum=0)
                        t_bgm_fade_out = gr.Number(value=1.5, label='淡出(秒)', minimum=0)

                with gr.Accordion('字幕轨', open=True):
                    with gr.Row():
                        t_sub_file = gr.File(label='上传 SRT 字幕文件', file_types=['.srt', '.txt'])
                        t_sub_enable = gr.Checkbox(value=True, label='启用字幕烧录')
                    t_sub_text = gr.Textbox(
                        label='或直接粘贴字幕（SRT 格式，或每行一句纯文本）',
                        lines=8,
                        placeholder='1\n00:00:00,000 --> 00:00:03,000\n第一句字幕\n\n2\n00:00:03,000 --> 00:00:06,000\n第二句字幕',
                    )
                    with gr.Row():
                        t_font_size = gr.Slider(16, 72, value=32, step=1, label='字号')
                        t_margin_v = gr.Slider(10, 200, value=48, step=2, label='距底部边距')

                with gr.Accordion('时间轴 / 导出', open=True):
                    with gr.Row():
                        t_duration = gr.Number(
                            value=0, label='时间轴时长(秒)，0=自动（优先人声）', minimum=0)
                        t_fmt = gr.Dropdown(choices=EXPORT_FORMATS, value='mp4', label='导出格式')
                        t_width = gr.Number(value=1280, label='宽度', precision=0)
                        t_height = gr.Number(value=720, label='高度', precision=0)

                t_btn = gr.Button('开始合成', variant='primary')
                t_info = gr.Textbox(label='处理信息', interactive=False, lines=2)
                with gr.Row():
                    t_preview = gr.Video(label='预览')
                    t_download = gr.File(label='下载')

                t_btn.click(
                    fn=do_timeline_compose,
                    inputs=[
                        t_video, t_video_loop, t_mute, t_video_vol,
                        t_voice, t_voice_start, t_voice_vol,
                        t_bgm, t_bgm_start, t_bgm_vol, t_bgm_loop, t_bgm_fade_in, t_bgm_fade_out,
                        t_sub_file, t_sub_text, t_sub_enable, t_font_size, t_margin_v,
                        t_duration, t_fmt, t_width, t_height,
                    ],
                    outputs=[t_preview, t_download, t_info],
                )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    demo = build_ui()
    demo.queue(max_size=4, default_concurrency_limit=1)
    demo.launch(server_name='0.0.0.0', server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
