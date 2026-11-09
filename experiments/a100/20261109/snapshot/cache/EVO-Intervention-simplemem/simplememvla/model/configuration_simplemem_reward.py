"""SimpleMem native-video backbone with Robometer prediction heads."""

from transformers import PretrainedConfig


class SimpleMemRewardConfig(PretrainedConfig):
    model_type = "simplemem_reward"

    def __init__(
        self,
        backbone_model_name_or_path="Qwen/Qwen3.5-4B",
        backbone_config=None,
        progress_loss_type="discrete",
        progress_discrete_bins=10,
        head_dropout=0.1,
        preference_token_id=None,
        freeze_backbone=False,
        history_video_sec=60.0,
        history_video_fps=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if progress_loss_type not in {"discrete", "l1", "l2"}:
            raise ValueError("progress_loss_type must be discrete, l1 or l2")
        if progress_discrete_bins < 2:
            raise ValueError("progress_discrete_bins must be at least 2")
        self.backbone_model_name_or_path = backbone_model_name_or_path
        self.backbone_config = backbone_config
        self.progress_loss_type = progress_loss_type
        self.progress_discrete_bins = progress_discrete_bins
        self.head_dropout = head_dropout
        self.preference_token_id = preference_token_id
        self.freeze_backbone = freeze_backbone
        self.history_video_sec = history_video_sec
        self.history_video_fps = history_video_fps
