# 火山方舟 Seedance 视频 / Seedream 图片生成客户端
# 视频 API: https://console.volcengine.com/ark/region:cn-beijing/docs/82379/1520757
# 图片 API: https://www.volcengine.com/docs/82379/1541523
import base64
import mimetypes
import os
import time
import uuid
from typing import List, Optional

import requests

ARK_VIDEO_BASE = 'https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks'
ARK_IMAGE_BASE = 'https://ark.cn-beijing.volces.com/api/v3/images/generations'

# ---------- Seedance 视频 ----------
# 1.0 ~ 2.0 系列模型（展示名 -> 模型 ID）
MODELS = {
    'Seedance 2.0': 'doubao-seedance-2-0-260128',
    'Seedance 2.0 Fast': 'doubao-seedance-2-0-fast-260128',
    'Seedance 1.5 Pro': 'doubao-seedance-1-5-pro-251215',
    'Seedance 1.0 Pro': 'doubao-seedance-1-0-pro-250528',
    'Seedance 1.0 Pro Fast': 'doubao-seedance-1-0-pro-fast-251015',
    'Seedance 1.0 Lite T2V（仅文生）': 'doubao-seedance-1-0-lite-t2v-250428',
    'Seedance 1.0 Lite I2V（仅图生）': 'doubao-seedance-1-0-lite-i2v-250428',
}

RATIOS = ['16:9', '4:3', '1:1', '3:4', '9:16', '21:9', 'adaptive']
RESOLUTIONS = ['480p', '720p', '1080p']
MODES = ['文生视频', '图生视频-首帧', '图生视频-首尾帧']

# ---------- Seedream 图片 ----------
IMAGE_MODELS = {
    'Seedream 5.0 Pro': 'doubao-seedream-5-0-pro-260628',
    'Seedream 5.0 Lite': 'doubao-seedream-5-0-260128',
    'Seedream 4.5': 'doubao-seedream-4-5-251128',
    'Seedream 4.0': 'doubao-seedream-4-0-250828',
}

IMAGE_MODES = ['文生图', '图生图']
# 分辨率关键词（不同模型支持范围略有差异，交由服务端校验）
IMAGE_SIZES = [
    '1K', '2K', '3K', '4K',
    '2048x2048', '2304x1728', '1728x2304',
    '2560x1440', '1440x2560', '2496x1664', '1664x2496',
]
IMAGE_SIZE_BY_MODEL = {
    'doubao-seedream-5-0-pro-260628': ['2K', '3K', '4K', '2048x2048', '2560x1440', '1440x2560'],
    'doubao-seedream-5-0-260128': ['2K', '3K', '2048x2048', '2560x1440', '1440x2560'],
    'doubao-seedream-4-5-251128': ['2K', '4K', '2048x2048', '2560x1440', '1440x2560'],
    'doubao-seedream-4-0-250828': ['1K', '2K', '4K', '2048x2048', '2560x1440', '1440x2560'],
}


def file_to_data_url(path_or_url: str) -> str:
    if path_or_url.startswith(('http://', 'https://', 'data:', 'asset://')):
        return path_or_url
    mime = mimetypes.guess_type(path_or_url)[0] or 'application/octet-stream'
    with open(path_or_url, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return 'data:{};base64,{}'.format(mime, b64)


def build_content(prompt: str, first_frame: Optional[str] = None, last_frame: Optional[str] = None) -> list:
    content = []
    if prompt and prompt.strip():
        content.append({'type': 'text', 'text': prompt.strip()})
    if first_frame:
        content.append({
            'type': 'image_url',
            'image_url': {'url': file_to_data_url(first_frame)},
            'role': 'first_frame',
        })
    if last_frame:
        content.append({
            'type': 'image_url',
            'image_url': {'url': file_to_data_url(last_frame)},
            'role': 'last_frame',
        })
    return content


class SeedanceClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError('请填写 ARK API Key')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': 'Bearer {}'.format(api_key),
            'Content-Type': 'application/json',
        })

    def create_task(
        self,
        model: str,
        content: list,
        ratio: str = '16:9',
        duration: int = 5,
        resolution: str = '720p',
        generate_audio: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        camera_fixed: bool = False,
        return_last_frame: bool = False,
        web_search: bool = False,
    ) -> dict:
        payload = {
            'model': model,
            'content': content,
            'ratio': ratio,
            'duration': int(duration),
            'resolution': resolution,
            'generate_audio': bool(generate_audio),
            'watermark': bool(watermark),
            'return_last_frame': bool(return_last_frame),
        }
        if seed is not None and seed >= 0:
            payload['seed'] = int(seed)
        if camera_fixed:
            payload['camera_fixed'] = True
        if web_search:
            payload['tools'] = [{'type': 'web_search'}]

        resp = self.session.post(ARK_VIDEO_BASE, json=payload, timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError('创建任务失败 HTTP {}: {}'.format(resp.status_code, resp.text[:1000]))
        data = resp.json()
        if not data.get('id'):
            raise RuntimeError('创建任务未返回 id: {}'.format(data))
        return data

    def get_task(self, task_id: str) -> dict:
        resp = self.session.get('{}/{}'.format(ARK_VIDEO_BASE, task_id), timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError('查询任务失败 HTTP {}: {}'.format(resp.status_code, resp.text[:1000]))
        return resp.json()

    def wait_task(self, task_id: str, interval: int = 10, timeout: int = 1800, progress_cb=None) -> dict:
        t0 = time.time()
        while True:
            data = self.get_task(task_id)
            status = data.get('status', 'unknown')
            elapsed = int(time.time() - t0)
            if progress_cb:
                progress_cb(status, elapsed)
            if status in ('succeeded', 'completed', 'complete', 'success'):
                return data
            if status in ('failed', 'error', 'cancelled', 'canceled', 'content_filter'):
                err = data.get('error') or data.get('message') or data
                raise RuntimeError('任务失败 status={}: {}'.format(status, err))
            if elapsed > timeout:
                raise TimeoutError('等待超时（{}s），task_id={}'.format(timeout, task_id))
            time.sleep(interval)

    @staticmethod
    def extract_video_url(task: dict) -> Optional[str]:
        content = task.get('content')
        if isinstance(content, dict):
            return content.get('video_url')
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('video_url'):
                    return item['video_url']
        return None

    @staticmethod
    def download_video(url: str, save_path: str, strip_meta: bool = True) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        if strip_meta:
            from strip_meta import strip_aigc_meta
            strip_aigc_meta(save_path, inplace=True)
        return save_path


class SeedreamClient:
    """火山方舟 Seedream 图片生成（文生图 / 图生图），同步接口。"""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError('请填写 ARK API Key')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': 'Bearer {}'.format(api_key),
            'Content-Type': 'application/json',
        })

    def generate(
        self,
        model: str,
        prompt: str,
        images: Optional[List[str]] = None,
        size: str = '2K',
        watermark: bool = False,
        response_format: str = 'url',
        sequential_image_generation: str = 'disabled',
        max_images: int = 4,
        web_search: bool = False,
        seed: Optional[int] = None,
        optimize_prompt: bool = False,
    ) -> dict:
        payload = {
            'model': model,
            'prompt': prompt.strip(),
            'size': size,
            'watermark': bool(watermark),
            'response_format': response_format,
            'sequential_image_generation': sequential_image_generation,
        }
        if images:
            urls = [file_to_data_url(p) for p in images if p]
            if len(urls) == 1:
                payload['image'] = urls[0]
            elif len(urls) > 1:
                payload['image'] = urls
        if sequential_image_generation == 'auto':
            payload['sequential_image_generation_options'] = {
                'max_images': max(1, min(int(max_images), 15)),
            }
        if web_search:
            payload['tools'] = [{'type': 'web_search'}]
        if seed is not None and seed >= 0:
            payload['seed'] = int(seed)
        if optimize_prompt:
            payload['optimize_prompt_options'] = {'mode': 'standard'}

        # 图片生成通常比视频快，但组图/4K 可能较久
        resp = self.session.post(ARK_IMAGE_BASE, json=payload, timeout=300)
        if resp.status_code >= 400:
            raise RuntimeError('图片生成失败 HTTP {}: {}'.format(resp.status_code, resp.text[:1000]))
        return resp.json()

    @staticmethod
    def extract_image_urls(result: dict) -> List[str]:
        urls = []
        data = result.get('data')
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                if item.get('url'):
                    urls.append(item['url'])
                elif item.get('b64_json'):
                    urls.append('data:image/png;base64,{}'.format(item['b64_json']))
        return urls

    @staticmethod
    def download_image(url: str, save_path: str, strip_meta: bool = True) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if url.startswith('data:'):
            # data:image/png;base64,xxxx
            _, b64 = url.split(',', 1)
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(b64))
        else:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
        if strip_meta:
            from strip_meta import strip_aigc_meta
            strip_aigc_meta(save_path, inplace=True)
        return save_path

    def save_images(self, result: dict, output_dir: str, prefix: Optional[str] = None) -> List[str]:
        urls = self.extract_image_urls(result)
        if not urls:
            raise RuntimeError('未返回图片 URL: {}'.format(result))
        prefix = prefix or uuid.uuid4().hex[:12]
        paths = []
        for i, url in enumerate(urls):
            ext = '.png' if 'png' in url[:64] else '.jpg'
            path = os.path.join(output_dir, '{}_{}{}'.format(prefix, i, ext))
            self.download_image(url, path, strip_meta=True)
            paths.append(path)
        return paths
