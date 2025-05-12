import math

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from timm.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange, repeat

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from vssd import Mamba2

from hilbert import decode, encode
from pyzorder import ZOrderIndexer

from mamba_util import PatchMerging, SimplePatchMerging, Stem, SimpleStem, Mlp


class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])


class StandardAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = inner_dim


    def forward(self, x, H, W):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SCANS(nn.Module):
    def __init__(self, size=16, dim=2, scan_type='scan', ):
        super().__init__()
        size = int(size)
        max_num = size ** dim
        indexes = np.arange(max_num)
        if 'sweep' == scan_type:  # ['sweep', 'scan', 'zorder', 'zigzag', 'hilbert']
            locs_flat = indexes
        elif 'scan' == scan_type:
            indexes = indexes.reshape(size, size)
            for i in np.arange(1, size, step=2):
                indexes[i, :] = indexes[i, :][::-1]
            locs_flat = indexes.reshape(-1)
        elif 'zorder' == scan_type:
            zi = ZOrderIndexer((0, size - 1), (0, size - 1))
            locs_flat = []
            for z in indexes:
                r, c = zi.rc(int(z))
                locs_flat.append(c * size + r)
            locs_flat = np.array(locs_flat)
        elif 'zigzag' == scan_type:
            indexes = indexes.reshape(size, size)
            locs_flat = []
            for i in range(2 * size - 1):
                if i % 2 == 0:
                    start_col = max(0, i - size + 1)
                    end_col = min(i, size - 1)
                    for j in range(start_col, end_col + 1):
                        locs_flat.append(indexes[i - j, j])
                else:
                    start_row = max(0, i - size + 1)
                    end_row = min(i, size - 1)
                    for j in range(start_row, end_row + 1):
                        locs_flat.append(indexes[j, i - j])
            locs_flat = np.array(locs_flat)
        elif 'hilbert' == scan_type:
            bit = int(math.log2(size))
            locs = decode(indexes, dim, bit)
            locs_flat = self.flat_locs_hilbert(locs, dim, bit)
        else:
            raise Exception('invalid encoder mode')
        locs_flat_inv = np.argsort(locs_flat)
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)

    def flat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = []
        l = 2 ** num_bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(num_dim):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)
        return np.array(ret).astype(np.uint64)

    def __call__(self, img):
        img_encode = self.encode(img)
        return img_encode

    def encode(self, img):
        img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat_inv.expand(img.shape), img)
        return img_encode

    def decode(self, img):
        img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat.expand(img.shape), img)
        return img_decode


class Mamba2AD(nn.Module):
    def __init__(self,
        d_model,
        d_conv=3, #default to 3 for 2D
        conv_init=None,
        expand=2,
        headdim=64, #default to 64
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="silu", #default to silu
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False, #default to False, for custom implementation
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        linear_attn_duality=False,
        d_state = 64,
        size: int = 8,
        scan_type: str = 'scan',
        **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim  # equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads_g = self.d_inner // self.headdim
        self.nheads = ngroups * self.nheads_g
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        # convert chunk_size to triton.language.int32
        self.chunk_size = chunk_size  # torch.tensor(chunk_size,dtype=torch.int32)
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.ssd_positve_dA = kwargs.get('ssd_positve_dA', True)  # default to False, ablation for linear attn duality
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads_g
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs)  #

        conv_dim = self.d_inner + 2 * self.d_state

        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads_g, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads_g, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        A_log = repeat(A_log, "n -> r n", r=ngroups).flatten()
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # modified from RMSNormGated to layer norm
        # assert RMSNormGated is not None
        # self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # linear attention duality
        self.linear_attn_duality = linear_attn_duality
        self.kwargs = kwargs
        self.scans = SCANS(size, scan_type=scan_type)
        kwargs['bidirection'] = False

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        '''
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        '''

        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3)  # (B, H, L, D)
        dt = dt.permute(0, 2, 1)  # (B, H, L)
        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
        if self.ssd_positve_dA: dA = -dA

        V_scaled = V * dA
        K = B.view(batch, 1, seqlen, dstate)  # (B, 1, L, D)
        if getattr(self, "__DEBUG__", False):
            A_mat = dA.cpu().detach().numpy()
            A_mat = A_mat.reshape(batch, -1, H, W)
            setattr(self, "__data__", dict(
                dA=A_mat, H=H, W=W, V=V, ))

        if self.ngroups == 1:
            ## get kv via transpose K and V
            KV = K.transpose(-2, -1) @ V_scaled  # (B, H, dstate, D)
            Q = C.view(batch, 1, seqlen, dstate)  # .repeat(1, head, 1, 1)
            x = Q @ KV  # (B, H, L, D)
            x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        else:
            assert head % self.ngroups == 0
            dstate = dstate // self.ngroups
            K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4)  # (B, 1, g, L, dstate)
            V_scaled = V_scaled.view(batch, head // self.ngroups, self.ngroups, seqlen, dim)  # (B, H//g, g, L, D)
            Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4)  # (B, 1, g, L, dstate)

            KV = K.transpose(-2, -1) @ V_scaled  # (B,  H//g, g, dstate, D)
            x = Q @ KV  # (B, H//g, g, L, D)
            V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head // self.ngroups,
                                                                                self.ngroups, seqlen,
                                                                                dim)  # (B, H//g, g, L, D)
            x = x + V_skip  # (B, H//g, g, L, D)
            x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim)  # (B, L, H, D)
            x = x.contiguous()

        return x


    def forward(self, u, H, W, seq_idx=None):
        """
        u: (B,C,H,W)
        Returns: same shape as u
        """
        batch, dim, H, W = u.shape
        L = H * W
        u = u.permute(0, 2, 3, 1)

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)


        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads_g], dim=-1
        )

        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]

        #2D Convolution
        xBC = xBC.permute(0, 3, 1, 2).contiguous()
        #xBC = xBC.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).contiguous()
        #xBC = xBC.permute(0, 2, 3, 1).view(B, H*W, -1).contiguous()

        xBCdt = torch.cat([xBC, dt], dim=-1)

        K = self.ngroups
        xBCdts = []
        assert K in [1, 2, 4, 8]
        xBCdt = xBCdt.permute(0, 3, 1, 2)
        if K >= 2:
            xBCdts.append(self.scans.encode(xBCdt.view(batch, -1, L)))
        if K >= 4:
            xBCdts.append(self.scans.encode(torch.transpose(xBCdt, dim0=2, dim1=3).contiguous().view(batch, -1, L)))
        if K >= 8:
            xBCdts.append(self.scans.encode(torch.rot90(xBCdt, k=1, dims=(2, 3)).contiguous().view(batch, -1, L)))
            xBCdts.append(self.scans.encode(
                torch.transpose(torch.rot90(xBCdt, k=1, dims=(2, 3)), dim0=2, dim1=3).contiguous().view(batch, -1, L)))

        xBCdts = torch.stack(xBCdts, dim=1).view(batch, K // 2, -1, L)
        xBCdts = torch.cat([xBCdts, torch.flip(xBCdts, dims=[-1])], dim=1)
        xBCdts = xBCdts.to(u.dtype)

        xBC, dt = torch.split(xBCdts, [self.d_inner + 2 * self.d_state, self.nheads_g], dim=-2)
        dt = dt.permute(0, 3, 1, 2).flatten(2).contiguous()

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC.permute(0, 3, 1 ,2).flatten(2), [self.ngroups * self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x, dt, A, B, C = to_ttensor(x, dt, A, B, C)
        if self.linear_attn_duality:
            y = self.non_casual_linear_attn(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt, A, B, C, self.D, H, W
            )
        else:
            if self.kwargs.get('bidirection', False):
                #assert self.ngroups == 2 #only support bidirectional with 2 groups
                x = to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)).chunk(2, dim=-2)
                B = to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                C = to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                dt = dt.chunk(2, dim=-1) # (B, L, nheads) -> (B, L, nheads//2)*2
                A, D = A.chunk(2, dim=-1), self.D.chunk(2,dim=-1) # (nheads) -> (nheads//2)*2
                y_forward = mamba_chunk_scan_combined(
                    x[0], dt[0], A[0], B[0], C[0], chunk_size=self.chunk_size, D=D[0], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y_backward = mamba_chunk_scan_combined(
                    x[1].flip(1), dt[1].flip(1), A[1], B[1].flip(1), C[1].flip(1), chunk_size=self.chunk_size, D=D[1], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y = torch.cat([y_forward, y_backward.flip(1)], dim=-2)
            else:
                y = mamba_chunk_scan_combined(
                    to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)),
                    to_ttensor(dt),
                    to_ttensor(A),
                    to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)),
                    to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)),
                    chunk_size=self.chunk_size,
                    D=to_ttensor(self.D),
                    z=None,
                    seq_idx=seq_idx,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
        out_y = y.flatten(2).view(batch, L, K, -1).permute(0, 2, 3, 1)
        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(batch, K // 2, -1, L)
        ys = []
        if K >= 2:
            ys.append(self.scans.decode(out_y[:, 0]))
            ys.append(self.scans.decode(inv_y[:, 0]))
        if K >= 4:
            ys.append(
                torch.transpose(self.scans.decode(out_y[:, 1]).view(batch, -1, W, H), dim0=2, dim1=3).contiguous().view(batch,
                                                                                                                    -1,
                                                                                                                    L))
            ys.append(
                torch.transpose(self.scans.decode(inv_y[:, 1]).view(batch, -1, W, H), dim0=2, dim1=3).contiguous().view(batch,
                                                                                                                    -1,
                                                                                                                    L))
        if K >= 8:
            ys.append(
                torch.rot90(self.scans.decode(out_y[:, 2]).view(batch, -1, W, H), k=3, dims=(2, 3)).contiguous().view(batch, -1,
                                                                                                                  L))
            ys.append(
                torch.rot90(self.scans.decode(inv_y[:, 2]).view(batch, -1, W, H), k=3, dims=(2, 3)).contiguous().view(batch, -1,
                                                                                                                  L))
            ys.append(
                torch.rot90(torch.transpose(self.scans.decode(out_y[:, 3]).view(batch, -1, W, H), dim0=2, dim1=3), k=3,
                            dims=(2, 3)).contiguous().view(batch, -1, L))
            ys.append(
                torch.rot90(torch.transpose(self.scans.decode(inv_y[:, 3]).view(batch, -1, W, H), dim0=2, dim1=3), k=3,
                            dims=(2, 3)).contiguous().view(batch, -1, L))
        y = sum(ys)
        y = y.view(batch, -1, H, W).permute(0, 2, 3, 1)

        # # Multiply "gate" branch and apply extra normalization layer
        # y = self.norm(y, z)
        y = self.norm(y)
        y = y*z
        out = self.out_proj(y).permute(0, 3, 1, 2).contiguous()
        return out


class VMAMBA2Block(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ssd_expansion=2, ssd_chunk_size=256,
                 linear_attn_duality=False, d_state = 64, size: int = 8, scan_type: str = 'scan', num_direction: int = 8,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        if kwargs.get('attn_type', 'mamba2') == 'standard':
            self.attn = StandardAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop)
        elif kwargs.get('attn_type', 'mamba2') == 'mamba2':
            self.attn = Mamba2AD(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=num_direction, chunk_size=ssd_chunk_size,
                                linear_attn_duality=linear_attn_duality, d_state=d_state, size=size, scan_type=scan_type, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x + self.cpe1(x)
        shortcut = x

        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # SSD or Standard Attention
        x = self.attn(x, H, W)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        return x
