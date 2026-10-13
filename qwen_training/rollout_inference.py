"""Greedy previous-response draft verification for causal video rollouts.

Every token, including EOS, must be the current model's greedy choice. Any
mismatch or low-margin token causes a full ordinary autoregressive generation.
The only draft source is the preceding prediction, never an annotated answer.
Progress is pooled from prompt positions only, excluding every draft token.
"""

import torch


@torch.inference_mode()
def generate_verified(model, messages, visual_features, visual_mask, drafts,
                      max_new_tokens=192, minimum_margin=1.0):
    if len(drafts) != len(messages):
        raise ValueError("Drafts must match the batch")
    count=len(messages)
    responses=[None]*count
    # Keep PT completely independent of draft length/content and fallback batch
    # composition, using the original prompt-only regression path once for all.
    model.qwen.model.rope_deltas=None
    progress=model(messages,visual_features,visual_mask).progress
    accepted=[False]*count
    margins=[None]*count
    draft_tokens=[model.tokenizer.encode(text,add_special_tokens=False)+[model.tokenizer.eos_token_id]
                  if text else [] for text in drafts]
    eligible=[i for i,ids in enumerate(draft_tokens) if 0<len(ids)<=max_new_tokens]
    if eligible:
        features=visual_features[eligible]
        masks=visual_mask[eligible]
        prompt=model.build_inputs([messages[i] for i in eligible],features,masks)
        length=prompt['inputs_embeds'].shape[1]
        width=max(len(draft_tokens[i]) for i in eligible)
        device=features.device
        ids=torch.full((len(eligible),width),model.tokenizer.pad_token_id,device=device,dtype=torch.long)
        valid=torch.zeros_like(ids,dtype=torch.bool)
        for j,i in enumerate(eligible):
            ids[j,:len(draft_tokens[i])]=torch.tensor(draft_tokens[i],device=device)
            valid[j,:len(draft_tokens[i])]=True
        suffix=model.qwen.get_input_embeddings()(ids[:,:-1])
        embedded=torch.cat((prompt['inputs_embeds'],suffix),dim=1)
        attention=torch.cat((prompt['attention_mask'],valid[:,:-1].long()),dim=1)
        # Each video time step is a fresh sequence. A prior generation may have
        # left rotary-position deltas for a different batch size on this module.
        model.qwen.model.rope_deltas=None
        # This is a causal verification pass. It cannot see future draft tokens
        # when predicting an earlier token; EOS itself is verified, not injected.
        hidden=model.qwen.model(inputs_embeds=embedded,attention_mask=attention,
                                use_cache=False,return_dict=True,output_hidden_states=False).last_hidden_state
        logits=model.qwen.lm_head(hidden[:,length-1:length+width-1])
        top=logits.float().topk(2,dim=-1)
        matches=(top.indices[...,0]==ids)|~valid
        delta=(top.values[...,0]-top.values[...,1]).masked_fill(~valid,float('inf'))
        minimum=delta.min(dim=-1).values
        use=(matches.all(dim=-1)&(minimum>=minimum_margin)).cpu().tolist()
        local_margins=minimum.cpu().tolist()
        for j,i in enumerate(eligible):
            margins[i]=local_margins[j]
            if use[j]:
                responses[i]=model.tokenizer.decode(draft_tokens[i],skip_special_tokens=True)
                accepted[i]=True
        del hidden,logits,top,embedded,suffix,prompt
    fallback=[i for i in range(count) if not accepted[i]]
    if fallback:
        model.qwen.model.rope_deltas=None
        normal=model.generate([messages[i] for i in fallback],visual_features[fallback],visual_mask[fallback],max_new_tokens=max_new_tokens)
        for j,i in enumerate(fallback):
            responses[i]=normal['responses'][j]
    return dict(responses=responses,progress=progress,draft_accepted=accepted,
                draft_minimum_margin=margins,minimum_margin=minimum_margin)
