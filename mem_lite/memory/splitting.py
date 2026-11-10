"""Event boundary rules and their cache/checkpoint compatibility contract."""

import math

SPLIT_METHODS = ("min_similarity", "max_depth")
DEPTH_SPEC = {
    "version": 1,
    "method": "max_depth",
    "similarity": "adjacent_main_view_clip_cls_cosine",
    "depth": "exclusive_left_global_max+exclusive_right_global_max-2*similarity",
    "context": "first_event_split_window_gaps",
    "empty_side": "current_similarity",
    "ties": "earliest",
    "threshold": None,
}


def global_depth_scores(similarities):
    """O(L) exclusive prefix/suffix maxima; an empty side contributes s_i.

    Only observed gaps in the eligible split window participate. Negative
    scores are retained (no clipping, smoothing, peak filter or threshold).
    """
    values = [float(value) for value in similarities]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("事件相似度必须为有限数")
    if not values:
        return []
    left, peak = [values[0]], values[0]
    for value in values[1:]:
        left.append(peak)
        peak = max(peak, value)
    right, peak = [0.0] * len(values), values[-1]
    right[-1] = peak
    for index in range(len(values) - 2, -1, -1):
        right[index] = peak
        peak = max(peak, values[index])
    return [a + b - 2 * value for a, b, value in zip(left, right, values, strict=True)]


def split_method(memory_config):
    method = memory_config.get("event_split_method", "min_similarity")
    if method not in SPLIT_METHODS:
        raise ValueError(f"未知事件切分方式: {method}")
    return method


def validate_history_source(source, expected_method=None):
    """Legacy caches without a method remain the original Qwen/minimum rule."""
    method = split_method(source.get("memory_config", {}))
    if expected_method is not None and expected_method != method:
        raise ValueError("事件切分方式与缓存/检查点不一致，请重新生成对应缓存")
    metadata = source.get("event_split")
    if method == "max_depth":
        if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in DEPTH_SPEC.items()):
            raise ValueError("CLIP 深度切分缓存缺少或不匹配 event_split 元数据，请重新生成缓存")
        clip = metadata.get("clip", {})
        for key in ("weights_sha256", "config_sha256", "processor_sha256"):
            value = clip.get(key, "")
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"CLIP 缓存缺少有效 {key}")
        if (clip.get("feature") != "last_cls_post_layernorm_no_projection"
                or clip.get("dtype") != "float32" or clip.get("attention") != "eager"):
            raise ValueError("CLIP 特征定义或计算精度不匹配")
    elif metadata is not None:
        raise ValueError("原版切分缓存不应包含 CLIP 深度切分元数据")
    return method


def add_split_arguments(parser):
    parser.add_argument("--event-split-method", choices=SPLIT_METHODS,
                        help="原版最小相似度，或 CLIP 最大深度分数（无阈值）")
    parser.add_argument("--clip-model-path", help="max_depth 使用的本地 CLIP 权重目录")


def override_memory_config(config, method=None, clip_model_path=None):
    config = dict(config)
    if method is not None:
        config["event_split_method"] = method
        if method == "min_similarity" and clip_model_path is None:
            config.pop("clip_model_path", None)
    if clip_model_path is not None:
        config["clip_model_path"] = str(clip_model_path)
    if split_method(config) == "min_similarity" and config.get("clip_model_path"):
        raise ValueError("clip_model_path 仅用于 max_depth；原版模式请使用原版配置")
    return config
