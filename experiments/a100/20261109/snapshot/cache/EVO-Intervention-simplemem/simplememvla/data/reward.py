"""Native-video input for progress/success and paired preference supervision."""

import torch
from torch.nn.utils.rnn import pad_sequence


def build_reward_messages(instruction, paired=False, current_cameras=0):
    """Videos are supplied separately, in A/B order; images follow each video.

    Keep the task before the history so every causal video representation sees it.
    The caller samples history with messages.derive_video_sampling and supplies
    original frame indices/fps in video_metadata, as in upstream SimpleMem.
    """
    if not instruction.strip():
        raise ValueError("instruction must not be empty")
    content = [
        {
            "type": "text",
            "text": (
                f"The task is: {instruction}. Evaluate task progress and success from the timestamped video history. "
                "Progress is 0 at the start and 1 when the task is completed."
            ),
        }
    ]
    for name in ["A", "B"] if paired else ["A"]:
        content.extend([{"type": "text", "text": f"Trajectory {name}:"}, {"type": "video"}])
        for camera in range(current_cameras):
            content.extend([{"type": "text", "text": f"Current camera {camera + 1}:"}, {"type": "image"}])
    if paired:
        content.append(
            {"type": "text", "text": ("Which trajectory makes more progress towards the task, A or B? <|pref_token|>")}
        )
    return [{"role": "user", "content": content}]


class SimpleMemRewardCollator:
    """Features use upstream videos/video_metadata/images plus optional targets.

    One history camera per trajectory, one or two trajectories per sample.
    Targets are per temporal patch (normally two sampled video frames), in video
    order; sample the label at the LAST frame of each patch. Images are context,
    not additional progress targets. -100 masks missing labels.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        encoded = []
        for feature in features:
            paired = len(feature["videos"]) == 2
            if len(feature["videos"]) not in (1, 2):
                raise ValueError("Each sample must contain one history video, or two for preference")
            images = feature.get("images") or []
            if len(images) % len(feature["videos"]):
                raise ValueError("Paired trajectories must supply the same number of current cameras")
            messages = build_reward_messages(feature["instruction"], paired, len(images) // len(feature["videos"]))
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            item = self.processor(
                text=[text],
                videos=feature["videos"],
                images=images or None,
                video_metadata=feature["video_metadata"],
                do_sample_frames=False,
                return_tensors="pt",
            )
            groups = int(item["video_grid_thw"][:, 0].sum())
            for key in ("progress_targets", "success_targets"):
                if key in feature and torch.as_tensor(feature[key]).shape != (groups,):
                    raise ValueError(f"{key} must contain {groups} temporal-patch labels, not raw-frame labels")
            if "preference_targets" in feature and not paired:
                raise ValueError("Preference labels require two trajectories")
            encoded.append(item)
        batch = {}
        for key, padding in (
            ("input_ids", self.processor.tokenizer.pad_token_id),
            ("attention_mask", 0),
            ("mm_token_type_ids", 0),
        ):
            if key in encoded[0]:
                batch[key] = pad_sequence([item[key][0] for item in encoded], batch_first=True, padding_value=padding)
        for key in ("pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw"):
            values = [item[key] for item in encoded if key in item]
            if values:
                batch[key] = torch.cat(values)
        for key in ("progress_targets", "success_targets", "preference_targets"):
            present = [key in feature for feature in features]
            if any(present):
                if not all(present):
                    raise ValueError(f"All samples in a batch must provide {key}; use -100 for missing frame labels")
                values = [torch.as_tensor(feature[key], dtype=torch.float32) for feature in features]
                batch[key] = (
                    torch.stack(values)
                    if key == "preference_targets"
                    else pad_sequence(values, batch_first=True, padding_value=-100)
                )
        return batch
