"""Load a cached native model input and original-PT supervision separately."""
from pathlib import Path


def load_input(row):
    import torch
    from safetensors.torch import load_file
    text=load_file(row['prompt_tensor_file'])
    batch={k:v.unsqueeze(0) for k,v in text.items()}
    pixels=[load_file(p) for p in row['pixel_caches']]
    for key in ['pixel_values','image_grid_thw']:
        if key in pixels[0]:batch[key]=torch.cat([p[key] for p in pixels])
    supervision={'pt':torch.tensor([row['pt']],dtype=torch.float32)}
    for key in ['before_pt','after_pt']:
        if key in row:supervision[key]=torch.tensor([row[key]],dtype=torch.float32)
    # Keep the model's input clean; downstream objectives must explicitly consume PT.
    return batch,supervision
