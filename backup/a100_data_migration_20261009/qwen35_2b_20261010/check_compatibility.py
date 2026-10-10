"""Configuration/structure audit only: no downloaded 2B weights, data or training loop."""
from pathlib import Path
from unittest.mock import patch
import argparse,ast,hashlib,json
import torch,transformers,peft
from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration
from qwen_training.model import QwenProgressModel
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

b=Path('/data/evo-inter/verification/qwen35_2b_20261010')
raw=json.loads((b/'config-2b.json').read_text())
config=Qwen3_5Config.from_dict(raw)
config._attn_implementation='sdpa'
# Instantiate the real installed architecture on meta tensors, without loading any checkpoint.
with torch.device('meta'):
 base=Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
 with patch('qwen_training.model.AutoTokenizer.from_pretrained',return_value=object()), \
      patch('qwen_training.model.AutoImageProcessor.from_pretrained',return_value=object()), \
      patch('qwen_training.model.Qwen3_5ForConditionalGeneration.from_pretrained',return_value=base):
  model=QwenProgressModel('/configuration-only-audit',dtype=torch.bfloat16,freeze_backbone=True,lora_rank=16,lora_alpha=32,lora_dropout=0.05)
modules={
 'language_layers':len(model.qwen.model.language_model.layers),
 'language_hidden_dim':model.qwen.config.text_config.hidden_size,
 'vision_layers':len(model.qwen.model.visual.blocks),
 'vision_hidden_dim':model.qwen.config.vision_config.hidden_size,
 'vision_output_dim':model.qwen.config.vision_config.out_hidden_size,
 'visual_adapter_input_dim':model.visual_adapter[1].in_features,
 'visual_adapter_output_dim':model.visual_adapter[1].out_features,
 'progress_head_input_dim':model.progress_head.mlp[0].in_features,
 'lora_linear_modules':sum(hasattr(m,'lora_A') for m in model.modules()),
 'lora_parameters':sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'lora_' in n),
 'all_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
 'vision_frozen':not any(p.requires_grad for p in model.qwen.model.visual.parameters()),
}
assert modules['language_layers']==24 and modules['language_hidden_dim']==2048
assert modules['vision_layers']==24 and modules['vision_output_dim']==2048
assert modules['visual_adapter_input_dim']==modules['visual_adapter_output_dim']==modules['progress_head_input_dim']==2048
assert modules['vision_frozen'] and modules['lora_linear_modules']>0
# Exercise the existing trainable head implementation at its new dimension on CPU.
# This is synthetic component verification, not pretrained-model inference or an optimizer step.
model.visual_adapter.to_empty(device='cpu');model.progress_head.to_empty(device='cpu')
model.visual_adapter.float();model.progress_head.float()
for root in [model.visual_adapter,model.progress_head]:
 for module in root.modules():
  if hasattr(module,'reset_parameters'):module.reset_parameters()
x=torch.randn(2,9,2048,requires_grad=True)
y=model.visual_adapter(x)
mask=torch.ones((2,9),dtype=torch.bool);mask[0,:2]=False
progress=model.progress_head(y,mask)
assert progress.shape==(2,) and torch.isfinite(progress).all()
progress.square().mean().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
# Reuse only the actual parser declarations; do not invoke main or distributed initialization.
source=Path('/data/evo-inter/code/qwen_training/train_distributed.py').read_text()
main=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
parser_nodes=[]
for node in main.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets):break
 parser_nodes.append(node)
namespace={'argparse':argparse,'__doc__':'Configuration-only verification'}
exec(compile(ast.Module(body=parser_nodes,type_ignores=[]),'<parser-only>','exec'),namespace)
args=namespace['parser'].parse_args(['--model-path','/data/evo-inter/models/Qwen3.5-4B','--index-files','unused-index','--output-dir','unused-output','--epochs','12','--save-epochs','2','4','6','8','10','12','--global-batch-size','32','--model-path','/data/evo-inter/models/Qwen3.5-2B'])
assert args.model_path=='/data/evo-inter/models/Qwen3.5-2B' and args.resume is None
files=['model.py','vision_encoder.py','mem_attention.py','train_distributed.py','run_a100_8gpu.sh']
report={
 'status':'PASS: configuration and module dimensions support 2B without source edits',
 'production_code_modified':False,'formal_training_started':False,'download_progress_checked':False,
 'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},
 'fast_kernels_available':bool(is_fast_path_available),'structure':modules,
 'component_check':'Existing visual adapter and progress head ran forward/backward with synthetic CPU tensors at width 2048; no optimizer step.',
 'cli_override':{'model_path':args.model_path,'epochs':args.epochs,'save_epochs':args.save_epochs,'global_batch_size':args.global_batch_size,'resume':args.resume},
 'source_sha256':{f:hashlib.sha256((Path('/data/evo-inter/code/qwen_training')/f).read_bytes()).hexdigest() for f in files},
 'findings':[
  'Qwen3.5-2B uses the same Qwen3_5ForConditionalGeneration class supported by the installed Transformers.',
  'LoRA targets are discovered from language-layer Linear modules; layer count and widths are not hardcoded.',
  'Visual adapter dimensions come from vision/text config; ProgressHead input comes from text hidden size.',
  'MEM uses actual attention head dimensions, block count, merger and output tensor shape. References to 2560 in these files are documentation, not shape constraints.',
  'The launcher appends user arguments after defaults; argparse accepts the final --model-path override.',
  'Dataset, 30/70 prompt sampling, labels, loss weighting and validation split do not depend on the language hidden dimension.'
 ],
 'requirements_before_real_training':[
  'Wait for the independently running downloader to complete; this audit did not inspect download state.',
  'Use a fresh output directory and omit --resume from any 4B run: resume restores the checkpoint configuration, including model_path.',
  'Retain the existing batch settings initially. A larger 2B batch is optional tuning, not required for dimension compatibility.',
  'Actual pretrained loading, full multimodal forward/backward, eight-GPU validation and throughput are not verified by this configuration-only check.'
 ],
 'command_example_not_executed':'bash /data/evo-inter/code/qwen_training/run_a100_8gpu.sh /data/evo-inter/outputs/qwen35-2b-8gpu --model-path /data/evo-inter/models/Qwen3.5-2B'
}
(b/'compatibility_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
