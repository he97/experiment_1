import math

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from collections import namedtuple
from einops import rearrange, repeat
from entmax import entmax15


# constants

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates'
])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def not_equals(val):
    def inner(x):
        return x != val
    return inner

def equals(val):
    def inner(x):
        return x == val
    return inner

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std = 0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device = x.device)
        return self.emb(n)[None, :, :]

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim = 1, offset = 0):
        t = torch.arange(x.shape[seq_dim], device = x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

class RelativePositionBias(nn.Module):
    def __init__(self, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + bias

# classes

class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.value, *rest)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        n = torch.norm(x, dim = -1, keepdim = True).clamp(min = self.eps)
        return x / n * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_class = nn.LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm_class(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out = None, mult = 4, glu = False, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(0.2),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head =512,
        heads =2,
        causal = False,
        mask = None,
        talking_heads = False,
        sparse_topk = None,
        use_entmax15 = False,
        num_mem_kv = 0,
        dropout = 0.,
        on_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.dropout = nn.Dropout(dropout)

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # entmax
        self.attn_fn = entmax15 if use_entmax15 else F.softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim * 2), nn.GLU()) if on_attn else nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        rel_pos = None,
        sinusoidal_emb = None,
        prev_attn = None,
        mem = None
    ):
        b, n, _, h, talking_heads, device = *x.shape, self.heads, self.talking_heads, x.device
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim = -2)
            v_input = torch.cat((mem, v_input), dim = -2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset = offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device = device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim = -2)
            v = torch.cat((mem_v, v), dim = -2)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value = True)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots

        if talking_heads:
            dots = einsum('b h i j, h k -> b k i j', dots, self.pre_softmax_proj).contiguous()

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones((i, j), device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim = -1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, dim = -1)
        post_softmax_attn = attn

        attn = self.dropout(attn)

        if talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj).contiguous()

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        intermediates = Intermediates(
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        return self.to_out(out), intermediates

class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        heads = 2,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rezero = False,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        position_infused_attn = False,
        custom_layers = None,
        sandwich_coef = None,
        par_ratio = None,
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        pre_norm = True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        self.has_pos_emb = position_infused_attn or rel_pos_bias
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None
        self.rel_pos = RelativePositionBias(causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance) if rel_pos_bias else None

        self.pre_norm = pre_norm and not residual_attn

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_fn = partial(norm_class, dim)

        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn  = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        for layer_type in self.layer_types:
            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)

            self.layers.append(nn.ModuleList([
                norm_fn(),
                layer
            ]))

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        mems = None,
        return_hiddens = False
    ):
        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        for ind, (layer_type, (norm, block)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == 'a':
                hiddens.append(x)
                layer_mem = mems.pop(0)

            if self.pre_norm:
                x = norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask = mask, sinusoidal_emb = self.pia_pos_emb, rel_pos = self.rel_pos, prev_attn = prev_attn, mem = layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            x = x + out

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if not self.pre_norm and not is_last:
                x = norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens = hiddens,
                attn_intermediates = intermediates
            )

            return x, intermediates

        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=102, embed_dim=512, patch_size=9):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv1d(in_channels=patch_size*patch_size, out_channels=embed_dim, kernel_size=1)


    def forward(self, x):
        # B, C, H, W = x.shape
        # x = x.view(B, C, -1)
        # x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        return x

class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal = False, **kwargs)


class DTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        image_size,
        patch_size,
        attn_layers,
        num_classes = None,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        assert isinstance(attn_layers, Encoder), 'attention layers must be an Encoder'
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

        dim = attn_layers.dim

        patch_dim =512

        self.num_class = num_classes

        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(num_patches, patch_dim, patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = FeedForward(dim, dim_out = num_classes, dropout = dropout) if exists(num_classes) else None

        self.conv1 = nn.Conv1d(in_channels=1, out_channels= 64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels= 128, kernel_size=1)
        self.BN = nn.BatchNorm1d(220, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.Linear1_1 = nn.Linear(in_features=1536, out_features=200)
        self.Linear1_2 = nn.Linear(in_features=1152, out_features=200)
        self.Linear1_3 = nn.Linear(in_features=1024, out_features=200)
        self.Linear1_4 = nn.Linear(in_features=640, out_features=200)
        self.Linear1_5 = nn.Linear(in_features=512, out_features=200)
        self.Linear1_6 = nn.Linear(in_features=512, out_features=200)
        self.Linear1_7 = nn.Linear(in_features=22784, out_features=200)
        self.Linear2 = nn.Linear(in_features=200, out_features=num_classes)
    def forward_one(self, x, sign):#(128,220,25)
        B, C, N = x.shape
        x = x.permute(0, 2, 1)
        fea = torch.zeros(B, N, self.num_class).cuda()
        for i in range(N):
            x1 = x[:, i, :].unsqueeze(1)
            x1 = self.conv1(x1)
            x1 = self.conv2(x1)
            # x1 = self.BN(x1)
            # x1 = self.relu(x1)
            x1 = x1.view(B, 1, -1)
            if sign == 1:
                x1 = self.Linear1_1(x1)
            elif sign == 2:
                x1 = self.Linear1_2(x1)
            elif sign == 3:
                x1 = self.Linear1_3(x1)
            elif sign == 4:
                x1 = self.Linear1_4(x1)
            elif sign == 5:
                x1 = self.Linear1_5(x1)
            elif sign == 6:
                x1 = self.Linear1_6(x1)
            elif sign == 7:
                x1 = self.Linear1_7(x1)
            x1 = self.Linear2(x1)
            x1 = x1.squeeze(1)
            fea[:, i, :] = x1
        return fea

    def forward_two(self, x):#(128,25,112)
        B, N, C = x.shape
        cnt = 0
        fea = torch.empty(B, N, 6216)
        for i in range(C-1):
            for j in range(i+1, C):
                x1 = torch.div(x[:, :, i] - x[:, :, j], x[:, :, i] + x[:, :, j])
                fea[:, :, cnt] = x1
                cnt = cnt + 1
        return fea

    def forward(self, img):
        # Update

        B, C, _, _ = img.shape
        V = torch.zeros(B, 4, 25).cuda()

        img = img.view(B, C, -1)
        V[:, 0, :] = img[:, 39, :]
        V[:, 1, :] = img[:, 18, :]
        V[:, 2, :] = img[:, 17, :]
        V[:, 3, :] = img[:, 14, :]
        img = self.BN(img)
        img1 = img[:, :12, :]
        img2 = img[:, 12:21, :]
        img3 = img[:, 21:29, :]
        img4 = img[:, 29:34, :]
        img5 = img[:, 34:38, :]
        img6 = img[:, 38:42, :]
        img7 = img[:, 42:, :]

        feature1 = self.forward_one(img1, 1)
        feature2 = self.forward_one(img2, 2)
        feature3 = self.forward_one(img3, 3)
        feature4 = self.forward_one(img4, 4)
        feature5 = self.forward_one(img5, 5)
        feature6 = self.forward_one(img6, 6)
        feature7 = self.forward_one(img7, 7)

        #Skip connect
        feature_1 = torch.cat((feature1, feature2), 2)
        feature_1 = torch.cat((feature_1, feature3), 2)
        feature_1 = torch.cat((feature_1, feature4), 2)
        feature_1 = torch.cat((feature_1, feature5), 2)
        feature_1 = torch.cat((feature_1, feature6), 2)
        feature_1 = torch.cat((feature_1, feature7), 2)

        #Pixel binary model
        # feature_2 = self.forward_two(feature_1)

        # feature = torch.cat((feature_1, feature_2), 2)
        # feature = feature.cuda()

        img = self.patch_embed(feature_1)
        # img = feature.permute(0, 2, 1)
        b, n, _ = img.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, img), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)
        if not exists(self.mlp_head):
            return x

        return V, feature_1, x[:,0], self.mlp_head(x[:, 0])

