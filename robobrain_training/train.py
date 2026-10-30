"""单卡 LoRA SFT 入口；只有显式运行本模块才会训练。"""

import argparse
import json
from pathlib import Path

from transformers import Trainer, TrainingArguments, set_seed

from .callbacks import RunMonitor
from .data import RewardCollator, RewardDataset
from .model import load_model


def make_trainer(config, model, processor, *, output_dir=None):
    arguments = TrainingArguments(
        output_dir=output_dir or config["output_dir"],
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        logging_nan_inf_filter=False,
        prediction_loss_only=True, remove_unused_columns=False, label_names=["labels"],
        report_to="none", dataloader_num_workers=0, seed=config["seed"],
        **config["training"],
    )
    model.config.use_cache = False
    return Trainer(
        model=model, args=arguments,
        train_dataset=RewardDataset(config["train_data"]),
        eval_dataset=RewardDataset(config["validation_data"]),
        data_collator=RewardCollator(processor, config["max_length"]), processing_class=processor,
        callbacks=[RunMonitor()],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("Output directory is nonempty; choose a new directory or explicitly resume")
    set_seed(config["seed"])
    model, processor = load_model(config, trainable=True)
    trainer = make_trainer(config, model, processor)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()


if __name__ == "__main__":
    main()
