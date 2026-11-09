"""Render training records and full-validation losses at every saved checkpoint."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent


def complete_records(path):
    raw = path.read_text()
    return [json.loads(s) for s in raw[:raw.rfind('\n')+1].splitlines() if s.strip()]


def main():
    config = json.loads((RUN/'config.json').read_text())
    output = Path(config['output_dir'])
    train = complete_records(output/'train.jsonl')
    validation = complete_records(output/'validation.jsonl')
    steps_per_epoch = json.loads((RUN/'settings.json').read_text())['steps_per_epoch']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot([r['step']/steps_per_epoch for r in train], [r['loss'] for r in train],
            color='#6b9dc8', lw=1, alpha=.75, label='Training: logged update loss')
    ax.plot([r['epoch'] for r in validation], [r['train_loss_since_previous_validation'] for r in validation],
            color='#1665b5', marker='o', ms=4, lw=1.7, label='Training: interval sample-weighted mean')
    ax.plot([r['epoch'] for r in validation], [r['loss'] for r in validation],
            color='#df690c', marker='o', ms=4, lw=1.7, label='Validation: complete held-out split')
    for epoch in (1, 2):
        ax.axvline(epoch, color='#9aa4b0', ls='--', lw=.8)
    ax.set(xlabel='Epoch', ylabel='Soft-label cross-entropy loss', xlim=(0, 3), ylim=(0, None))
    ax.set_title('Training and validation loss — checkpoint every 0.2 epoch', loc='left', fontsize=15)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.16)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ('png','svg'):
        temporary = output/f'loss-curve.tmp.{ext}'
        fig.savefig(temporary, dpi=160)
        temporary.replace(output/f'loss-curve.{ext}')
    plt.close(fig)


if __name__=='__main__':
    main()
