"""MEM 时序注意力核心：复用 Qwen3.5 视觉层权重，不新增训练参数。

下文维度符号：
B = 样本数，V = 视角数（当前为 3），K = 每个视角的帧数（num_frames）。
M = B*V*K，表示图像总数；P = 所有图像在 Merger 合并前的 patch 总数。
n = 当前图像的 patch 数，同一视角的各帧相同，不同样本/视角可以不同。
D = 视觉特征维度；A = 注意力头数；d = 每个头的维度，D = A*d。
当前 Qwen3.5-4B：D=1024、A=16、d=64；Merger 输出维度为 2560。
"""

import torch
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision


def spatial_attention(query, key, value, frame_patch_counts, dropout_p=0.0, scale=None):
    """相同网格的图像放到 batch 维，一次调用 SDPA；图像之间仍完全隔离。"""
    heads, head_dim = value.shape[1:]
    if len(set(frame_patch_counts)) == 1:
        patches = frame_patch_counts[0]
        q, k, v = [x.reshape(-1, patches, heads, head_dim).transpose(1, 2)
                   for x in (query, key, value)]
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)
        return output.transpose(1, 2).reshape(-1, heads * head_dim)
    outputs = []
    for q, k, v in zip(query.split(frame_patch_counts), key.split(frame_patch_counts),
                       value.split(frame_patch_counts)):
        q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)
        outputs.append(output.squeeze(0).transpose(0, 1).reshape(-1, heads * head_dim))
    return torch.cat(outputs)


def spatial_block_forward(block, hidden_states, position_embeddings, frame_patch_counts):
    """与原生视觉块相同的计算，合并逐帧 SDPA 调用以减少 CPU/GPU 调度开销。"""
    attn = block.attn
    query, key, value = (
        attn.qkv(block.norm1(hidden_states))
        .reshape(-1, 3, attn.num_heads, attn.head_dim).permute(1, 0, 2, 3).unbind(0)
    )
    query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
    output = spatial_attention(
        query, key, value, frame_patch_counts,
        dropout_p=attn.attention_dropout if attn.training else 0.0, scale=attn.scaling,
    )
    hidden_states = hidden_states + attn.proj(output)
    return hidden_states + block.mlp(block.norm2(hidden_states))


def temporal_attention(query, key, value, frame_patch_counts, num_frames, frame_mask=None):
    """Q/K/V 为 [所有 patch, heads, head_dim]，每个视角的帧按从旧到新连续排列。"""
    heads, head_dim = value.shape[1:]  # value: [P,A,d]；取出两个整数 A=16、d=64。
    # 每个样本、每个视角是一条独立序列，可以有不同的 patch 数量。
    stream_sizes = [sum(frame_patch_counts[i:i + num_frames])
                    for i in range(0, len(frame_patch_counts), num_frames)]
    # frame_patch_counts 是长度 M 的列表；stream_sizes 是长度 B*V 的列表，每项为 K*n。
    causal_mask = torch.ones(num_frames, num_frames, device=value.device, dtype=torch.bool).tril()
    if len(set(frame_patch_counts)) == 1:
        patches = frame_patch_counts[0]
        stream_count = len(frame_patch_counts) // num_frames
        q, k, v = [x.reshape(stream_count, num_frames, patches, heads, head_dim)
                   .permute(0, 2, 3, 1, 4).reshape(-1, heads, num_frames, head_dim)
                   for x in (query, key, value)]
        if frame_mask is None:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            allowed = causal_mask & frame_mask[:, None, :] & frame_mask[:, :, None]
            allowed = allowed.repeat_interleave(patches, dim=0)[:, None]
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            valid = frame_mask.repeat_interleave(patches, dim=0)
            output = output.masked_fill(~valid[:, None, :, None], 0)
        return (output.reshape(stream_count, patches, heads, num_frames, head_dim)
                .permute(0, 3, 1, 2, 4).reshape(-1, heads, head_dim))
    outputs = []  # 收集 B*V 个张量，每个形状为 [K*n,A,d]。
    streams = zip(query.split(stream_sizes), key.split(stream_sizes), value.split(stream_sizes))
    for stream_index, (q, k, v) in enumerate(streams):
        # 这里一次会只拿到一个视角的跨域K时序的图像帧
        # split 沿第 0 维拆分；本次 q/k/v 都是一个视角的完整历史：[K*n,A,d]。
        patches = q.shape[0] // num_frames  # 整数 n = (K*n)/K，即该视角每帧的 patch 数。
        # reshape: [K*n,A,d] → [K,n,A,d]；permute: → [n,A,K,d]。
        q, k, v = [x.reshape(num_frames, patches, heads, head_dim).permute(1, 2, 0, 3)
                   for x in (q, k, v)]
        if frame_mask is None:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            valid = frame_mask[stream_index]  # [K]，True 才是真实帧。
            allowed = causal_mask & valid[None, :] & valid[:, None]  # [K,K]：因果 + 有效 Q/K。
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            # 全部被遮挡的 padding query 输出为 0；显式清零，无效帧不参与后续融合。
            output = output.masked_fill(~valid[None, None, :, None], 0)
        # 输出形状仍为 [n,A,K,d]。
        # 做了不同时间的帧之间的注意力
        # permute: [n,A,K,d] → [K,n,A,d]；reshape: → [K*n,A,d]，恢复帧优先的排列。
        outputs.append(output.permute(2, 0, 1, 3).reshape(-1, heads, head_dim))
    return torch.cat(outputs)  # 沿第 0 维拼回所有样本/视角：[P,A,d]。


def mem_block_forward(block, hidden_states, position_embeddings, frame_patch_counts, num_frames,
                      frame_mask=None, time_embeddings=None):
    """同一组 Q/K/V：先沿时间混合 V，再用帧内空间注意力读取混合后的 V。"""
    attn = block.attn  # 当前层的注意力模块，不是张量；输入 hidden_states 为 [P,D]。
    # 时间编码只加到 Q/K/V 投影前的输入；残差仍使用原 hidden_states。当前帧编码为 0。
    attention_input = hidden_states if time_embeddings is None else hidden_states + time_embeddings
    query, key, value = (
        attn.qkv(block.norm1(attention_input))  # norm1: [P,D] → [P,D]；qkv: → [P,3*D]。
        .reshape(-1, 3, attn.num_heads, attn.head_dim)  # [P,3*D] → [P,3,A,d]，3 对应 Q/K/V。
        .permute(1, 0, 2, 3)  # [P,3,A,d] → [3,P,A,d]。
        .unbind(0)  # 拆开第 0 维，得到 Q、K、V 三个 [P,A,d] 张量。
    )
    # cos/sin 各为 [P,d]，内部扩展为 [P,1,d]，广播到 A 个头；同 patch 的空间位置跨帧相同。
    query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
    # 旋转后的 query/key 仍各为 [P,A,d]。
    # position_embeddings [P 64]的二元组
    value = temporal_attention(query, key, value, frame_patch_counts, num_frames, frame_mask)
    # [P,A,d] → [P,A,d]。
    # num_frames 表示每个视角参与时序注意力的帧数 K
    output = spatial_attention(query, key, value, frame_patch_counts)
    hidden_states = hidden_states + attn.proj(output)
    # norm2 保持 [P,D]；当前 MLP 为 [P,1024] → [P,4096] → [P,1024]；残差后仍为 [P,D]。
    return hidden_states + block.mlp(block.norm2(hidden_states))


def temporal_position_encoding(frame_positions, hidden_size):
    """相对帧位置 → 固定时间编码 [...,D]；sin(tω)、cos(tω)-1 保证 e(0)=0。"""
    frequencies = 10000.0 ** (-torch.arange(0, hidden_size, 2, device=frame_positions.device).float()
                             / hidden_size)  # [ceil(D/2)]，无可学习参数。
    angles = frame_positions.float()[..., None] * frequencies  # [...,ceil(D/2)]。
    return torch.stack((angles.sin(), angles.cos() - 1), dim=-1).flatten(-2)[..., :hidden_size]


def mem_vision_forward(visual, pixel_values, image_grid_thw, num_frames, frame_mask=None):
    """对已逐帧预处理的 patch 执行 MEM 编码，返回所有帧的 Merger 输出 token。

    输入排列为 [样本, 视角, 时间]，时间从旧到新；每帧都用 images 接口单独预处理，
    image_grid_thw 为 [B*V*K,3]，这个 3 表示每张图像用三个数字描述网格：[T, H, W]
    K表示每个视角参与时序注意力的帧数
    每行的 T 必须为 1。BVK 分别表示[样本, 视角, 时间]
    同一视角在各时刻的 H/W 网格必须一致。
    MEM 固定在第 4、8、12……层执行；帧位置为 [-(K-1),...,0]，不表示实际秒数。
    frame_mask 为 [B*V,K]，默认全部有效；False 的帧不参与融合，输出 token 清零。
    此底层函数保留所有帧 token，由 encode_history 在融合完成后筛选各视角的当前帧。

    当前模型输入：pixel_values [P,1536]、image_grid_thw [M,3]。
    1536 = RGB通道3 * 原生时间patch大小2 * 空间patch大小16 * 16，是像素展开长度。
    单张图像被复制以满足时间patch大小2，两份合成一个时间块，因此网格 T 仍为 1。
    输出：[P/4,2560]，包含所有帧；4 来自 Merger 的 2*2 空间合并。
    例：B=1、V=3、K=3，每图256*256像素，则 M=9、每图256个patch、P=2304。
    """
    if num_frames < 1 or len(image_grid_thw) % num_frames:  # 整数检查：K>=1，且 M 能被 K 整除。
        raise ValueError("图像数量必须是 num_frames 的整数倍，且 num_frames >= 1")
    grid = image_grid_thw.to(device=visual.device)  # [M,3] = [B*V*K,3]；仅换设备，形状不变。
    streams = grid.reshape(-1, num_frames, 3)  # [M,3] → [B*V,K,3]，每组是一条视角历史。
    # grid[:,0]: [M]；streams[:,:1]: [B*V,1,3]，广播到 [B*V,K,3] 后比较；any() 返回标量。
    if (grid[:, 0] != 1).any() or (streams != streams[:, :1]).any():
        raise ValueError("MEM 需要逐帧图像输入，且同一视角的历史帧网格必须一致")
    temporal_attention_layer_indices = set(range(3, len(visual.blocks), 4))
    # Python 整数集合，当前模型为 {3,7,11,15,19,23}；不是张量。
    if frame_mask is None:
        frame_mask = torch.ones(streams.shape[:2], device=visual.device, dtype=torch.bool)
    else:
        frame_mask = torch.as_tensor(frame_mask, device=visual.device, dtype=torch.bool)
    if frame_mask.shape != streams.shape[:2]:
        raise ValueError("frame_mask 必须为 [B*V,K]")
    # [K] → [B*V,K]；按帧顺序生成相对位置，当前帧为 0，无效位置清零。
    frame_positions = torch.arange(1 - num_frames, 1, device=visual.device).expand(streams.shape[:2])
    frame_positions = frame_positions.masked_fill(~frame_mask, 0)
    counts = grid.prod(dim=-1)  # [M]，每张图像合并前的 patch 数。
    token_mask = frame_mask.flatten().repeat_interleave(counts // visual.spatial_merge_size**2)
    # token_mask: [P/4]，每个 Merger 输出 token 是否来自有效帧。
    pixels = pixel_values.to(device=visual.device, dtype=visual.dtype)  # [P,1536]，仅换设备/类型。
    if num_frames == 1 or not temporal_attention_layer_indices:  # K=1 或视觉编码器不足 4 层。
        tokens = visual(pixels, grid_thw=grid, return_dict=True).pooler_output
        return tokens.masked_fill(~token_mask[:, None], 0)  # [P/4,2560]。

    # patch_embed: [P,1536] → [P,1024]；空间位置编码: [M,3] → [P,1024]；相加后 [P,1024]。
    hidden_states = visual.patch_embed(pixels) + visual.fast_pos_embed_interpolate(grid)
    time_embeddings = temporal_position_encoding(frame_positions, hidden_states.shape[-1])
    # [B*V,K,D] → [M,D] → [P,D]，同一帧的所有 patch 共用该帧的时间编码。
    time_embeddings = time_embeddings.reshape(-1, hidden_states.shape[-1]).repeat_interleave(counts, dim=0)
    time_embeddings = time_embeddings.to(hidden_states.dtype)
    rotary = visual.rot_pos_emb(grid)  # [M,3] → [P,d/2] = [P,32]，每个 patch 的空间旋转角度。
    rotary = torch.cat((rotary, rotary), dim=-1)  # [P,32] + [P,32] → [P,64]，匹配每头维度 d。
    position_embeddings = (rotary.cos(), rotary.sin())  # 二元组，cos/sin 各为 [P,64]。
    counts = grid.prod(dim=-1)  # [M,3] → [M]，每项 T*H*W 是一张图像合并前的 patch 数。
    frame_patch_counts = counts.tolist()  # [M] 张量 → 长度 M 的 Python 整数列表，各项之和为 P。
    # cumsum: [M]；左侧补一个 0 后为 [M+1]，记录每张图像在 [P,D] 中的起止边界。
    cu_seqlens = F.pad(counts.cumsum(dim=0, dtype=torch.int32), (1, 0))

    for index, block in enumerate(visual.blocks):  # index 是整数；每层输入/输出均为 [P,1024]。
        if index in temporal_attention_layer_indices:  # 选定层：时序注意力 + 空间注意力。
            hidden_states = mem_block_forward(
                block, hidden_states, position_embeddings, frame_patch_counts, num_frames,
                frame_mask=frame_mask, time_embeddings=time_embeddings,
            )  # [P,1024] → [P,1024]，时序融合不减少 patch 数量。
        elif len(set(frame_patch_counts)) == 1:
            hidden_states = spatial_block_forward(
                block, hidden_states, position_embeddings, frame_patch_counts,
            )
        else:  # 可变网格沿用原生路径。
            hidden_states = block(
                hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings,
            )  # [P,1024] → [P,1024]。
    # 2*2 patch 拼接并投影：[P,1024] → [P/4,4096] → [P/4,2560]；仍包含全部历史帧。
    tokens = visual.merger(hidden_states)  # 示例：[2304,1024] → [576,2560]。
    return tokens.masked_fill(~token_mask[:, None], 0)
