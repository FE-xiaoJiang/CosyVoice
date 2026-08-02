# 多轨合成引擎（视频 / 人声 / BGM / 字幕）
# 用 ffmpeg filter_complex 实现类似剪映的基础多轨混剪
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2500:] if proc.stderr else '命令执行失败')
    return proc.stdout.strip()


def probe_duration(path: str) -> float:
    out = run_cmd([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path,
    ])
    return float(out)


def probe_has_video(path: str) -> bool:
    out = run_cmd([
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'csv=p=0',
        path,
    ])
    return 'video' in out


def probe_has_audio(path: str) -> bool:
    out = run_cmd([
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'csv=p=0',
        path,
    ])
    return 'audio' in out


@dataclass
class VideoTrack:
    path: str
    loop: bool = True                 # 是否循环铺满时间轴
    mute_original: bool = True        # 是否静音原片音轨
    start: float = 0.0                # 入点（秒）
    volume: float = 1.0               # 保留原音时的音量


@dataclass
class AudioTrack:
    path: str
    start: float = 0.0                # 相对时间轴起点的偏移
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    loop: bool = False                # BGM 可循环铺满
    name: str = 'audio'


@dataclass
class SubtitleTrack:
    # 二选一：srt_path 或 srt_text
    srt_path: Optional[str] = None
    srt_text: str = ''
    font_size: int = 28
    font_color: str = 'white'
    outline_color: str = 'black'
    margin_v: int = 40                # 距底部像素
    enabled: bool = True


@dataclass
class TimelineProject:
    video: Optional[VideoTrack] = None
    voice: Optional[AudioTrack] = None      # 人声/旁白主音轨
    bgm: Optional[AudioTrack] = None        # 背景音乐
    extra_audios: List[AudioTrack] = field(default_factory=list)
    subtitle: Optional[SubtitleTrack] = None
    duration: float = 0.0                   # 0 表示自动取最长素材
    export_format: str = 'mp4'
    width: int = 1280
    height: int = 720
    fps: int = 25


def _escape_filter_path(path: str) -> str:
    # ffmpeg filter 路径转义：\ : '
    return path.replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")


def _write_srt_file(text: str, dest: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError('字幕内容为空')
    # 若用户只贴了纯文本行，自动转成简单 SRT（每行 3 秒）
    if '-->' not in text:
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        blocks = []
        t = 0.0
        for i, line in enumerate(lines, 1):
            start, end = t, t + 3.0
            blocks.append(
                '{}\n{} --> {}\n{}\n'.format(
                    i, _sec_to_ts(start), _sec_to_ts(end), line)
            )
            t = end
        text = '\n'.join(blocks)
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(text)
        if not text.endswith('\n'):
            f.write('\n')
    return dest


def _sec_to_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return '{:02d}:{:02d}:{:02d},{:03d}'.format(h, m, s, ms)


def resolve_duration(project: TimelineProject) -> float:
    if project.duration and project.duration > 0:
        return float(project.duration)

    candidates = []
    if project.voice:
        d = probe_duration(project.voice.path)
        candidates.append(project.voice.start + d)
    if project.bgm and not project.bgm.loop:
        d = probe_duration(project.bgm.path)
        candidates.append(project.bgm.start + d)
    for a in project.extra_audios:
        if not a.loop:
            d = probe_duration(a.path)
            candidates.append(a.start + d)
    if project.video and not project.video.loop:
        d = probe_duration(project.video.path)
        candidates.append(project.video.start + d)

    # 默认优先用人声时长；都没有则用视频
    if project.voice:
        return max(candidates) if candidates else probe_duration(project.voice.path)
    if project.video:
        return probe_duration(project.video.path)
    if candidates:
        return max(candidates)
    raise ValueError('无法推断时间轴时长，请上传素材或手动指定时长')


def compose(project: TimelineProject) -> Tuple[str, str, str]:
    """合成多轨工程，返回 (preview_path, download_path, info)"""
    if project.video is None and project.voice is None and project.bgm is None:
        raise ValueError('至少需要视频、人声或背景音乐其中之一')

    duration = resolve_duration(project)
    if duration <= 0.1:
        raise ValueError('时间轴时长过短')

    job_id = uuid.uuid4().hex[:10]
    export_format = project.export_format or 'mp4'
    out_path = os.path.join(OUTPUT_DIR, 'timeline_{}.{}'.format(job_id, export_format))

    with tempfile.TemporaryDirectory(dir=OUTPUT_DIR) as tmpdir:
        # ---- 仅导出音频 ----
        if export_format == 'wav':
            return _compose_audio_only(project, duration, out_path, tmpdir, job_id)

        # ---- 视频 + 多音轨 + 字幕 ----
        inputs: List[str] = []
        filter_parts: List[str] = []
        audio_labels: List[str] = []
        input_idx = 0

        # 视频轨
        if project.video:
            vpath = project.video.path
            vdur = probe_duration(vpath)
            if project.video.loop:
                loops = max(0, math.ceil(duration / max(vdur, 0.01)) - 1)
                inputs += ['-stream_loop', str(loops), '-i', vpath]
            else:
                inputs += ['-i', vpath]
            # 统一分辨率、帧率，裁到时间轴
            filter_parts.append(
                '[{}:v]scale={}:{}:force_original_aspect_ratio=decrease,'
                'pad={}:{}:(ow-iw)/2:(oh-ih)/2,fps={},setsar=1,'
                'trim=duration={},setpts=PTS-STARTPTS[vout]'.format(
                    input_idx, project.width, project.height,
                    project.width, project.height, project.fps, duration)
            )
            # 原片音轨
            if (not project.video.mute_original) and probe_has_audio(vpath):
                vol = max(0.0, float(project.video.volume))
                filter_parts.append(
                    '[{}:a]atrim=duration={},asetpts=PTS-STARTPTS,volume={}[va]'.format(
                        input_idx, duration, vol)
                )
                audio_labels.append('[va]')
            video_in_idx = input_idx
            input_idx += 1
            video_label = '[vout]'
        else:
            # 无视频时生成纯色底图，方便字幕预览
            filter_parts.append(
                'color=c=black:s={}x{}:d={}:r={}[vout]'.format(
                    project.width, project.height, duration, project.fps)
            )
            video_label = '[vout]'

        # 人声轨
        if project.voice:
            input_idx, label = _add_audio_input(
                project.voice, inputs, filter_parts, input_idx, duration, tmpdir)
            audio_labels.append(label)

        # BGM
        if project.bgm:
            input_idx, label = _add_audio_input(
                project.bgm, inputs, filter_parts, input_idx, duration, tmpdir)
            audio_labels.append(label)

        # 额外音轨
        for extra in project.extra_audios:
            input_idx, label = _add_audio_input(
                extra, inputs, filter_parts, input_idx, duration, tmpdir)
            audio_labels.append(label)

        # 混音
        if len(audio_labels) == 0:
            # 无音频则补静音
            filter_parts.append(
                'anullsrc=r=44100:cl=stereo,atrim=duration={},asetpts=PTS-STARTPTS[aout]'.format(duration)
            )
            aout = '[aout]'
        elif len(audio_labels) == 1:
            filter_parts.append(
                '{}aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[aout]'.format(
                    audio_labels[0])
            )
            aout = '[aout]'
        else:
            filter_parts.append(
                '{}amix=inputs={}:duration=longest:dropout_transition=0,'
                'aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[aout]'.format(
                    ''.join(audio_labels), len(audio_labels))
            )
            aout = '[aout]'

        # 字幕烧录
        final_v = video_label
        if project.subtitle and project.subtitle.enabled:
            srt_file = None
            if project.subtitle.srt_path:
                srt_file = project.subtitle.srt_path
            elif project.subtitle.srt_text and project.subtitle.srt_text.strip():
                srt_file = os.path.join(tmpdir, 'subs.srt')
                _write_srt_file(project.subtitle.srt_text, srt_file)
            if srt_file:
                # 复制到纯 ASCII 临时名，避免 filter 路径编码问题
                srt_ascii = os.path.join(tmpdir, 'subs_{}.srt'.format(job_id))
                shutil.copy2(srt_file, srt_ascii)
                style = (
                    "FontSize={},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                    "BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV={}".format(
                        int(project.subtitle.font_size), int(project.subtitle.margin_v))
                )
                # 允许覆盖颜色（简单映射）
                if project.subtitle.font_color.lower() not in ('white', '#ffffff', 'fff'):
                    # 保持默认白色；复杂颜色留给后续扩展
                    pass
                esc = _escape_filter_path(os.path.abspath(srt_ascii))
                filter_parts.append(
                    "{}subtitles='{}':force_style='{}'[vsub]".format(final_v, esc, style)
                )
                final_v = '[vsub]'

        filter_complex = ';'.join(filter_parts)

        if export_format == 'mp4':
            vcodec, acodec, extra = 'libx264', 'aac', ['-movflags', '+faststart', '-crf', '23']
        elif export_format == 'webm':
            vcodec, acodec, extra = 'libvpx-vp9', 'libopus', ['-b:v', '1M', '-crf', '32']
        else:
            vcodec, acodec, extra = 'libx264', 'aac', ['-movflags', '+faststart', '-crf', '23']

        cmd = [
            'ffmpeg', '-y', *inputs,
            '-filter_complex', filter_complex,
            '-map', final_v, '-map', aout,
            '-c:v', vcodec, '-preset', 'veryfast',
            '-c:a', acodec, '-b:a', '192k',
            '-t', str(duration),
            *extra,
            out_path,
        ]
        run_cmd(cmd)

    preview_path = os.path.join(OUTPUT_DIR, 'preview_{}.{}'.format(job_id, export_format))
    shutil.copy2(out_path, preview_path)

    tracks = []
    if project.video:
        tracks.append('视频' + ('(循环)' if project.video.loop else ''))
    if project.voice:
        tracks.append('人声×{:.0%}'.format(project.voice.volume))
    if project.bgm:
        tracks.append('BGM×{:.0%}{}'.format(
            project.bgm.volume, '(循环)' if project.bgm.loop else ''))
    if project.subtitle and project.subtitle.enabled and (
            project.subtitle.srt_path or project.subtitle.srt_text.strip()):
        tracks.append('字幕')
    info = '导出 {} | 时长 {:.2f}s | 轨道: {}'.format(
        export_format, duration, ' + '.join(tracks) if tracks else '无')
    return preview_path, out_path, info


def _add_audio_input(
    track: AudioTrack,
    inputs: List[str],
    filter_parts: List[str],
    input_idx: int,
    timeline_dur: float,
    tmpdir: str,
) -> Tuple[int, str]:
    """把一条音频加入 inputs/filters，返回 (新 input_idx, 输出标签)"""
    adur = probe_duration(track.path)
    start = max(0.0, float(track.start))
    vol = max(0.0, float(track.volume))
    fade_in = max(0.0, float(track.fade_in))
    fade_out = max(0.0, float(track.fade_out))

    # 循环铺满时间轴（相对 start 之后的部分）
    remain = max(0.01, timeline_dur - start)
    if track.loop:
        loops = max(0, math.ceil(remain / max(adur, 0.01)) - 1)
        inputs += ['-stream_loop', str(loops), '-i', track.path]
        usable = remain
    else:
        inputs += ['-i', track.path]
        usable = min(adur, remain)

    label = 'a{}'.format(input_idx)
    chain = '[{}:a]atrim=duration={},asetpts=PTS-STARTPTS,volume={}'.format(
        input_idx, usable, vol)

    if fade_in > 0:
        chain += ',afade=t=in:st=0:d={}'.format(min(fade_in, usable))
    if fade_out > 0:
        fo_st = max(0.0, usable - fade_out)
        chain += ',afade=t=out:st={}:d={}'.format(fo_st, min(fade_out, usable))

    # 延时对齐到时间轴，并补齐到时间轴长度（尾部静音）
    delay_ms = int(round(start * 1000))
    chain += ',adelay={}|{},apad=whole_dur={}[{}]'.format(delay_ms, delay_ms, timeline_dur, label)
    filter_parts.append(chain)
    return input_idx + 1, '[{}]'.format(label)


def _compose_audio_only(project, duration, out_path, tmpdir, job_id):
    inputs: List[str] = []
    filter_parts: List[str] = []
    audio_labels: List[str] = []
    input_idx = 0

    if project.voice:
        input_idx, label = _add_audio_input(
            project.voice, inputs, filter_parts, input_idx, duration, tmpdir)
        audio_labels.append(label)
    if project.bgm:
        input_idx, label = _add_audio_input(
            project.bgm, inputs, filter_parts, input_idx, duration, tmpdir)
        audio_labels.append(label)
    for extra in project.extra_audios:
        input_idx, label = _add_audio_input(
            extra, inputs, filter_parts, input_idx, duration, tmpdir)
        audio_labels.append(label)

    if not audio_labels:
        raise ValueError('导出 wav 需要至少一条音频轨')

    if len(audio_labels) == 1:
        filter_parts.append('{}aformat=sample_rates=44100:channel_layouts=stereo[aout]'.format(
            audio_labels[0]))
    else:
        filter_parts.append(
            '{}amix=inputs={}:duration=longest:dropout_transition=0,'
            'aformat=sample_rates=44100:channel_layouts=stereo[aout]'.format(
                ''.join(audio_labels), len(audio_labels))
        )

    cmd = [
        'ffmpeg', '-y', *inputs,
        '-filter_complex', ';'.join(filter_parts),
        '-map', '[aout]',
        '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
        '-t', str(duration),
        out_path,
    ]
    run_cmd(cmd)
    info = '仅导出音频 wav | 时长 {:.2f}s'.format(duration)
    return None, out_path, info


# ---------- 简易合并（兼容旧逻辑） ----------
def simple_merge(video_path: str, audio_path: str, export_format: str = 'mp4'):
    project = TimelineProject(
        video=VideoTrack(path=video_path, loop=True, mute_original=True),
        voice=AudioTrack(path=audio_path, volume=1.0, name='voice'),
        export_format=export_format,
    )
    return compose(project)
