"""固定三个输入槽位；缺失副视角使用同一张小尺寸黑色 RGB 图。"""

from PIL import Image

VIEW_POLICY = "main-first; missing auxiliary views use RGB black 64x64; v1"


def fill_views(images):
    if not 1 <= len(images) <= 3 or images[0] is None:
        raise ValueError("需要主视角及最多两个副视角；缺失副视角可用 None")
    images = list(images) + [None] * (3 - len(images))
    return [image if image is not None else Image.new("RGB", (64, 64), 0) for image in images]
