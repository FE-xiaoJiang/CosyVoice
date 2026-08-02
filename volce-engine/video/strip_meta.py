# 移除火山方舟生成结果中的 AIGC / C2PA 元数据标记
# - JPEG: 删除含 c2pa/jumbf 的 APP11 等段（不重编码）
# - PNG: 删除可疑 tEXt/iTXt/zTXt 文本块
# - MP4: 将含 c2pa 的顶层 uuid box 清零为同尺寸 free（不改偏移、不重编码）
import os
import re
import struct
from typing import List, Tuple

# 匹配常见 AI 溯源 / 大模型注入标记
_MARKER_RE = re.compile(
    rb'(?i)c2pa|jumbf|jumb|volcengine|seedream|seedance|doubao|'
    rb'trainedAlgorithmicMedia|ContentCredentials|ai[_ -]?generated|'
    rb'"Label"\s*:\s*"\d"|AIGC|aigc_info|Content Propagator|'
    rb'digitalSourceType|softwareAgent'
)


def has_aigc_markers(data: bytes) -> bool:
    return _MARKER_RE.search(data) is not None


def _strip_jpeg(data: bytes) -> Tuple[bytes, List[str]]:
    if len(data) < 4 or data[:2] != b'\xff\xd8':
        return data, []
    out = bytearray(data[:2])
    removed = []
    i = 2
    while i + 1 < len(data):
        if data[i] != 0xff:
            out.extend(data[i:])
            break
        # skip fill bytes 0xFF
        while i + 1 < len(data) and data[i] == 0xff and data[i + 1] == 0xff:
            i += 1
        if i + 1 >= len(data):
            break
        marker = data[i + 1]
        # standalone markers (no length): RST0-7, SOI, EOI, TEM
        if marker in (0x00, 0x01) or (0xd0 <= marker <= 0xd9):
            out.extend(data[i:i + 2])
            i += 2
            if marker == 0xd9:  # EOI
                if i < len(data):
                    out.extend(data[i:])
                break
            continue
        if marker == 0xda:  # SOS: remainder is entropy + EOI
            out.extend(data[i:])
            break
        if i + 4 > len(data):
            out.extend(data[i:])
            break
        seglen = int.from_bytes(data[i + 2:i + 4], 'big')
        if seglen < 2 or i + 2 + seglen > len(data):
            out.extend(data[i:])
            break
        seg = data[i:i + 2 + seglen]
        drop = False
        # APP11 (0xEB) 常承载 C2PA JUMBF；APP1/2 可能有 XMP
        if marker == 0xeb:
            drop = True
            removed.append('JPEG:APP11(C2PA)')
        elif 0xe0 <= marker <= 0xef:
            if _MARKER_RE.search(seg):
                drop = True
                removed.append('JPEG:APP{}(marker)'.format(marker - 0xe0))
        elif marker in (0xed, 0xfe):  # APP13 / COM
            if _MARKER_RE.search(seg):
                drop = True
                removed.append('JPEG:0x{:02X}'.format(marker))
        if not drop:
            out.extend(seg)
        i += 2 + seglen
    return bytes(out), removed


def _strip_png(data: bytes) -> Tuple[bytes, List[str]]:
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        return data, []
    out = bytearray(data[:8])
    removed = []
    i = 8
    while i + 12 <= len(data):
        length = struct.unpack('>I', data[i:i + 4])[0]
        ctype = data[i + 4:i + 8]
        chunk_end = i + 12 + length
        if chunk_end > len(data):
            out.extend(data[i:])
            break
        chunk = data[i:chunk_end]
        drop = False
        if ctype in (b'tEXt', b'iTXt', b'zTXt', b'eXIf'):
            if _MARKER_RE.search(chunk):
                drop = True
                removed.append('PNG:{}'.format(ctype.decode('ascii', errors='ignore')))
        if not drop:
            out.extend(chunk)
        i = chunk_end
        if ctype == b'IEND':
            if i < len(data):
                # trailing junk with markers — drop
                if _MARKER_RE.search(data[i:]):
                    removed.append('PNG:trailing')
                else:
                    out.extend(data[i:])
            break
    return bytes(out), removed


def _iter_top_boxes(data: bytes):
    i = 0
    n = len(data)
    while i + 8 <= n:
        size = int.from_bytes(data[i:i + 4], 'big')
        typ = data[i + 4:i + 8]
        hdr = 8
        if size == 1 and i + 16 <= n:
            size = int.from_bytes(data[i + 8:i + 16], 'big')
            hdr = 16
        elif size == 0:
            size = n - i
        if size < hdr or i + size > n:
            break
        yield i, size, typ, hdr
        i += size


def _strip_mp4(data: bytes) -> Tuple[bytes, List[str]]:
    """将含 C2PA 的顶层 uuid 清为零填充 free box，保持后续偏移不变。"""
    if len(data) < 16:
        return data, []
    # 粗判：非 ISO BMFF
    if data[4:8] != b'ftyp' and b'ftyp' not in data[:64]:
        return data, []

    out = bytearray(data)
    removed = []
    for off, size, typ, hdr in _iter_top_boxes(data):
        box = data[off:off + size]
        if typ == b'uuid' and _MARKER_RE.search(box):
            # size(4) + 'free' + zero payload
            out[off:off + 4] = struct.pack('>I', size if size < 2**32 else 1)
            out[off + 4:off + 8] = b'free'
            if size >= 8:
                out[off + 8:off + size] = b'\x00' * (size - 8)
            removed.append('MP4:uuid(C2PA)')
        elif typ == b'free' and size > 8 and _MARKER_RE.search(box):
            out[off + 8:off + size] = b'\x00' * (size - 8)
            removed.append('MP4:free(markers)')
    # moov/udta 中若夹带可疑文本，尝试清零对应 leaf（少见；C2PA 主要在顶层 uuid）
    return bytes(out), removed


def strip_aigc_meta(path: str, inplace: bool = True) -> dict:
    """
    检查并移除文件中的 AIGC/C2PA 元数据标记。
    返回 {path, changed, removed, before_has_markers, after_has_markers}
    """
    path = os.path.abspath(path)
    with open(path, 'rb') as f:
        data = f.read()
    before = has_aigc_markers(data)
    ext = os.path.splitext(path)[1].lower()
    removed: List[str] = []
    new = data

    if ext in ('.jpg', '.jpeg'):
        new, removed = _strip_jpeg(data)
    elif ext == '.png':
        new, removed = _strip_png(data)
    elif ext in ('.mp4', '.mov', '.m4v'):
        new, removed = _strip_mp4(data)
    else:
        # 未知类型：若能当 JPEG/PNG/MP4 识别则处理
        if data[:2] == b'\xff\xd8':
            new, removed = _strip_jpeg(data)
        elif data.startswith(b'\x89PNG'):
            new, removed = _strip_png(data)
        elif data[4:8] == b'ftyp':
            new, removed = _strip_mp4(data)

    # 若仍残留标记且是图片，再走一次 PIL 重保存去掉残余 EXIF（可能损一点质量，仅兜底）
    after = has_aigc_markers(new)
    if after and ext in ('.jpg', '.jpeg', '.png'):
        try:
            from io import BytesIO
            from PIL import Image
            im = Image.open(BytesIO(new))
            buf = BytesIO()
            if ext == '.png' or im.format == 'PNG':
                im.save(buf, format='PNG', optimize=True)
            else:
                im.save(buf, format='JPEG', quality=95, optimize=True)
            new2 = buf.getvalue()
            if not has_aigc_markers(new2):
                new = new2
                removed.append('reencode:strip-residual')
                after = False
        except Exception:
            pass

    changed = new != data
    if changed and inplace:
        tmp = path + '.stripping'
        with open(tmp, 'wb') as f:
            f.write(new)
        os.replace(tmp, path)

    return {
        'path': path,
        'changed': changed,
        'removed': removed,
        'before_has_markers': before,
        'after_has_markers': has_aigc_markers(new) if changed else before,
        'bytes_before': len(data),
        'bytes_after': len(new),
    }


def strip_aigc_meta_bytes(data: bytes, hint_ext: str = '') -> Tuple[bytes, List[str]]:
    ext = (hint_ext or '').lower()
    if ext in ('.jpg', '.jpeg') or data[:2] == b'\xff\xd8':
        return _strip_jpeg(data)
    if ext == '.png' or data.startswith(b'\x89PNG'):
        return _strip_png(data)
    if ext in ('.mp4', '.mov', '.m4v') or (len(data) > 8 and data[4:8] == b'ftyp'):
        return _strip_mp4(data)
    return data, []
