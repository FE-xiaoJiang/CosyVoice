# 火山方舟 Seedance 视频 / Seedream 图片生成 Gradio 界面
#
# 用法:
#   在 volce-engine/video/.env 中配置 ARK_API_KEY=xxx
#   conda run --no-capture-output -n cosyvoice python volce-engine/video/app.py --port 7862
#
# API 参考:
#   视频: https://console.volcengine.com/ark/region:cn-beijing/docs/82379/1520757
#   图片: https://www.volcengine.com/docs/82379/1541523
import argparse
import os
import uuid

import gradio as gr
from dotenv import load_dotenv

from client import (
    IMAGE_MODELS, IMAGE_MODES, IMAGE_SIZE_BY_MODEL, IMAGE_SIZES,
    MODELS, MODES, RATIOS, RESOLUTIONS,
    SeedanceClient, SeedreamClient, build_content,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 优先加载本目录 .env，再回退到仓库根目录 .env
load_dotenv(os.path.join(ROOT_DIR, '.env'))
load_dotenv(os.path.join(ROOT_DIR, '..', '..', '.env'), override=False)


def _resolve_api_key(api_key: str) -> str:
    key = (api_key or '').strip() or os.environ.get('ARK_API_KEY', '')
    if not key:
        raise gr.Error('请填写 ARK API Key，或在 volce-engine/video/.env 中配置 ARK_API_KEY')
    return key


# ===================== 视频生成 =====================

def on_mode_change(mode):
    # 文生视频：隐藏首尾帧；首帧：只显示首帧；首尾帧：都显示
    if mode == '文生视频':
        return (
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )
    if mode == '图生视频-首帧':
        return (
            gr.update(visible=True),
            gr.update(visible=False, value=None),
        )
    return (
        gr.update(visible=True),
        gr.update(visible=True),
    )


def on_model_change(model_name):
    model_id = MODELS[model_name]
    # Lite T2V 仅文生；Lite I2V 仅图生；其余两种模式都可用
    if 'lite-t2v' in model_id:
        return gr.update(choices=['文生视频'], value='文生视频')
    if 'lite-i2v' in model_id:
        return gr.update(choices=['图生视频-首帧', '图生视频-首尾帧'], value='图生视频-首帧')
    return gr.update(choices=MODES, value='文生视频')


def generate_video(
    api_key, model_name, mode, prompt,
    first_frame, last_frame,
    ratio, duration, resolution,
    generate_audio, watermark, camera_fixed, web_search, seed,
    progress=gr.Progress(track_tqdm=False),
):
    key = _resolve_api_key(api_key)

    model_id = MODELS.get(model_name)
    if not model_id:
        raise gr.Error('未知模型: {}'.format(model_name))

    if mode == '文生视频':
        if not prompt or not prompt.strip():
            raise gr.Error('文生视频模式需要填写提示词')
        if 'lite-i2v' in model_id:
            raise gr.Error('当前模型仅支持图生视频')
        first_frame, last_frame = None, None
    elif mode == '图生视频-首帧':
        if not first_frame:
            raise gr.Error('请上传首帧图片')
        if 'lite-t2v' in model_id:
            raise gr.Error('当前模型仅支持文生视频')
        last_frame = None
    elif mode == '图生视频-首尾帧':
        if not first_frame or not last_frame:
            raise gr.Error('首尾帧模式需要同时上传首帧和尾帧图片')
        if 'lite-t2v' in model_id:
            raise gr.Error('当前模型仅支持文生视频')
    else:
        raise gr.Error('未知模式: {}'.format(mode))

    # duration 范围提示（不同模型略有差异，交由服务端最终校验）
    duration = int(duration)
    if duration != -1 and not (2 <= duration <= 15):
        raise gr.Error('时长需在 2–15 秒，或填 -1 由模型智能决定（仅部分模型支持）')

    seed_val = None if seed is None or seed < 0 else int(seed)
    content = build_content(prompt or '', first_frame, last_frame)
    if not content:
        raise gr.Error('content 为空，请检查提示词或图片输入')

    client = SeedanceClient(key)
    progress(0.05, desc='提交任务...')
    task = client.create_task(
        model=model_id,
        content=content,
        ratio=ratio,
        duration=duration,
        resolution=resolution,
        generate_audio=generate_audio,
        watermark=watermark,
        seed=seed_val,
        camera_fixed=camera_fixed,
        web_search=web_search and mode == '文生视频',
    )
    task_id = task['id']

    def on_progress(status, elapsed):
        # 异步任务无精确进度，用时间做视觉反馈
        pct = min(0.9, 0.1 + elapsed / 600.0)
        progress(pct, desc='任务 {} | {} | 已等待 {}s'.format(task_id, status, elapsed))

    result = client.wait_task(task_id, interval=10, timeout=1800, progress_cb=on_progress)
    video_url = client.extract_video_url(result)
    if not video_url:
        raise gr.Error('任务成功但未返回 video_url: {}'.format(result))

    progress(0.95, desc='下载视频...')
    save_path = os.path.join(OUTPUT_DIR, '{}.mp4'.format(task_id.replace('/', '_')))
    client.download_video(video_url, save_path)
    progress(1.0, desc='完成')

    info = (
        '任务 ID: {}\n'
        '模型: {} ({})\n'
        '模式: {}\n'
        '分辨率/比例/时长: {} / {} / {}s\n'
        '本地文件: {}\n'
        '远程 URL(约24h有效): {}\n'
        '已清理: AIGC/C2PA 元数据标记'
    ).format(task_id, model_name, model_id, mode, resolution, ratio, duration, save_path, video_url)
    return save_path, save_path, info


# ===================== 图片生成 =====================

def on_image_mode_change(mode):
    if mode == '文生图':
        return gr.update(visible=False, value=None)
    return gr.update(visible=True)


def on_image_model_change(model_name):
    model_id = IMAGE_MODELS.get(model_name, '')
    sizes = IMAGE_SIZE_BY_MODEL.get(model_id, IMAGE_SIZES)
    default = '2K' if '2K' in sizes else sizes[0]
    # 仅 5.0 Lite 官方明确支持联网搜索
    web_search_visible = '5-0-260128' in model_id or '5.0 Lite' in model_name
    return (
        gr.update(choices=sizes, value=default),
        gr.update(visible=web_search_visible, value=False),
    )


def _normalize_image_files(files):
    """Gradio File/Image 可能返回 str / list / None。"""
    if not files:
        return []
    if isinstance(files, str):
        return [files]
    paths = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        elif isinstance(f, dict) and f.get('name'):
            paths.append(f['name'])
        elif hasattr(f, 'name'):
            paths.append(f.name)
    return [p for p in paths if p]


def generate_image(
    api_key, model_name, mode, prompt, ref_images,
    size, watermark, sequential, max_images, web_search, seed, optimize_prompt,
    progress=gr.Progress(track_tqdm=False),
):
    key = _resolve_api_key(api_key)

    model_id = IMAGE_MODELS.get(model_name)
    if not model_id:
        raise gr.Error('未知图片模型: {}'.format(model_name))
    if not prompt or not prompt.strip():
        raise gr.Error('请填写提示词')

    images = _normalize_image_files(ref_images)
    if mode == '图生图' and not images:
        raise gr.Error('图生图模式请至少上传 1 张参考图')
    if mode == '文生图':
        images = []
    if len(images) > 10:
        raise gr.Error('参考图最多 10 张')

    seed_val = None if seed is None or seed < 0 else int(seed)
    seq = 'auto' if sequential else 'disabled'
    # 联网搜索仅文生图有意义，且主要 5.0 Lite 支持
    use_web = bool(web_search) and mode == '文生图' and '5-0-260128' in model_id

    client = SeedreamClient(key)
    progress(0.1, desc='提交图片生成请求...')
    result = client.generate(
        model=model_id,
        prompt=prompt,
        images=images or None,
        size=size,
        watermark=watermark,
        sequential_image_generation=seq,
        max_images=int(max_images),
        web_search=use_web,
        seed=seed_val,
        optimize_prompt=optimize_prompt,
    )

    progress(0.7, desc='下载图片...')
    prefix = uuid.uuid4().hex[:12]
    paths = client.save_images(result, OUTPUT_DIR, prefix=prefix)
    urls = client.extract_image_urls(result)
    progress(1.0, desc='完成')

    info = (
        '模型: {} ({})\n'
        '模式: {}\n'
        '尺寸: {}\n'
        '组图: {}\n'
        '生成数量: {}\n'
        '本地文件:\n{}\n'
        '远程 URL(约24h有效):\n{}\n'
        '已清理: AIGC/C2PA 元数据标记'
    ).format(
        model_name, model_id, mode, size,
        seq, len(paths),
        '\n'.join(paths),
        '\n'.join(urls),
    )
    # Gallery 要 list；File 可接收多文件 list
    return paths, paths, info


def build_ui():
    default_key = os.environ.get('ARK_API_KEY', '')
    with gr.Blocks(title='火山方舟 视频/图片生成') as demo:
        gr.Markdown(
            '## 火山方舟 Seedance / Seedream\n'
            '视频：**Seedance** 文生/图生视频 · 图片：**Seedream** 文生图/图生图\n'
            '文档：[视频生成](https://console.volcengine.com/ark/region:cn-beijing/docs/82379/1520757) · '
            '[图片生成](https://www.volcengine.com/docs/82379/1541523)'
        )
        api_key = gr.Textbox(
            label='ARK API Key',
            value=default_key,
            type='password',
            placeholder='留空则读取 .env / 环境变量 ARK_API_KEY',
        )

        with gr.Tabs():
            # ---------- 视频 ----------
            with gr.Tab('视频生成 (Seedance)'):
                model_name = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value='Seedance 2.0',
                    label='视频模型（1.0–2.0）',
                )
                mode = gr.Radio(choices=MODES, value='文生视频', label='生成模式')
                prompt = gr.Textbox(
                    label='提示词 Prompt',
                    lines=4,
                    placeholder='主体 + 运动，背景 + 运动，镜头 + 运动。例如：一只橘猫在窗台伸懒腰，阳光洒进房间，镜头缓慢推进',
                )
                with gr.Row():
                    first_frame = gr.Image(
                        label='首帧图片', type='filepath', sources=['upload'], visible=False)
                    last_frame = gr.Image(
                        label='尾帧图片', type='filepath', sources=['upload'], visible=False)

                with gr.Accordion('生成参数', open=True):
                    with gr.Row():
                        ratio = gr.Dropdown(choices=RATIOS, value='16:9', label='画面比例')
                        duration = gr.Number(
                            value=5, label='时长(秒)，部分模型支持 -1 智能时长',
                            minimum=-1, maximum=15, step=1)
                        resolution = gr.Dropdown(choices=RESOLUTIONS, value='720p', label='分辨率')
                    with gr.Row():
                        generate_audio = gr.Checkbox(value=True, label='生成同步音频')
                        watermark = gr.Checkbox(value=False, label='加水印')
                        camera_fixed = gr.Checkbox(value=False, label='固定镜头(1.x)')
                        web_search = gr.Checkbox(value=False, label='联网搜索(2.0 文生)')
                        seed = gr.Number(value=-1, label='随机种子(-1 表示随机)', precision=0)

                generate_btn = gr.Button('开始生成视频', variant='primary')
                info_out = gr.Textbox(label='任务信息', lines=6, interactive=False)
                with gr.Row():
                    video_preview = gr.Video(label='结果预览')
                    download_out = gr.File(label='下载视频')

                mode.change(fn=on_mode_change, inputs=[mode], outputs=[first_frame, last_frame])
                model_name.change(fn=on_model_change, inputs=[model_name], outputs=[mode]).then(
                    fn=on_mode_change, inputs=[mode], outputs=[first_frame, last_frame]
                )
                generate_btn.click(
                    fn=generate_video,
                    inputs=[
                        api_key, model_name, mode, prompt,
                        first_frame, last_frame,
                        ratio, duration, resolution,
                        generate_audio, watermark, camera_fixed, web_search, seed,
                    ],
                    outputs=[video_preview, download_out, info_out],
                )

            # ---------- 图片 ----------
            with gr.Tab('图片生成 (Seedream)'):
                img_model = gr.Dropdown(
                    choices=list(IMAGE_MODELS.keys()),
                    value='Seedream 4.5',
                    label='图片模型',
                )
                img_mode = gr.Radio(choices=IMAGE_MODES, value='文生图', label='生成模式')
                img_prompt = gr.Textbox(
                    label='提示词 Prompt',
                    lines=4,
                    placeholder='主体 + 场景 + 风格/光影/构图。例如：一只橘猫趴在窗台，午后阳光，胶片感，浅景深',
                )
                ref_images = gr.File(
                    label='参考图（图生图 / 多图融合，最多 10 张）',
                    file_count='multiple',
                    file_types=['image'],
                    visible=False,
                )

                with gr.Accordion('生成参数', open=True):
                    with gr.Row():
                        img_size = gr.Dropdown(
                            choices=IMAGE_SIZE_BY_MODEL.get(
                                IMAGE_MODELS['Seedream 4.5'], IMAGE_SIZES),
                            value='2K',
                            label='输出尺寸',
                        )
                        max_images = gr.Number(
                            value=4, label='组图最大张数(auto 时生效)',
                            minimum=1, maximum=15, step=1, precision=0)
                        img_seed = gr.Number(value=-1, label='随机种子(-1 随机)', precision=0)
                    with gr.Row():
                        img_watermark = gr.Checkbox(value=False, label='加水印')
                        sequential = gr.Checkbox(value=False, label='组图生成(auto)')
                        optimize_prompt = gr.Checkbox(value=False, label='提示词优化')
                        img_web_search = gr.Checkbox(
                            value=False, label='联网搜索(5.0 Lite 文生)', visible=False)

                img_btn = gr.Button('开始生成图片', variant='primary')
                img_info = gr.Textbox(label='任务信息', lines=8, interactive=False)
                with gr.Row():
                    img_gallery = gr.Gallery(label='结果预览', columns=2, height=420)
                    img_download = gr.File(label='下载图片', file_count='multiple')

                img_mode.change(fn=on_image_mode_change, inputs=[img_mode], outputs=[ref_images])
                img_model.change(
                    fn=on_image_model_change,
                    inputs=[img_model],
                    outputs=[img_size, img_web_search],
                )
                img_btn.click(
                    fn=generate_image,
                    inputs=[
                        api_key, img_model, img_mode, img_prompt, ref_images,
                        img_size, img_watermark, sequential, max_images,
                        img_web_search, img_seed, optimize_prompt,
                    ],
                    outputs=[img_gallery, img_download, img_info],
                )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7862)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    demo = build_ui()
    demo.queue(max_size=4, default_concurrency_limit=1)
    demo.launch(server_name='0.0.0.0', server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
