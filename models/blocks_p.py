import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_, to_3tuple

from models.modules import ModuleParallel


class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w d -> b h w d c')
        x = self.norm(x)
        
        return rearrange(x, 'b h w d c -> b c h w d')


class DownBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_type='reflect', relufactor=0.2):
        super().__init__()

        self.conv1 = ModuleParallel(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type), 'conv1')
        self.norm1 = ModuleParallel(nn.InstanceNorm3d(out_channels), 'norm1')
        
        self.conv2 = ModuleParallel(nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=pad_type), 'conv2')
        self.norm2 = ModuleParallel(nn.InstanceNorm3d(out_channels), 'norm2')

        self.conv3 = ModuleParallel(nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=pad_type), 'conv3')
        self.norm3 = ModuleParallel(nn.InstanceNorm3d(out_channels), 'norm3')

        self.act = ModuleParallel(nn.LeakyReLU(negative_slope=relufactor, inplace=True), 'act')

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        residual = [x_.clone() for x_ in x]

        x = self.act(self.norm2(self.conv2(x)))
        x = self.act(self.norm3(self.conv3(x)))

        x = [x_ + re_ for (x_, re_) in zip(x, residual)]

        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query, key):
        B, N_q, C = query.shape
        B, N_k, _ = key.shape
        
        q = self.q(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.einsum("nhqd,nhkd->nhqk", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("nhqk,nhkd->nhqd", attn, v)
        out = out.transpose(1, 2).reshape(B, N_q, C)

        out = self.proj(out)

        return out


class AgentAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, agent_tokens):
        B, N, C = x.shape
        _, N_a, _ = agent_tokens.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(B, N_a, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        agent_attn = torch.einsum("nhad,nhkd->nhak", agent_tokens, k) * self.scale
        agent_attn = F.softmax(agent_attn, dim=-1)
        agent_v = torch.einsum("nhak,nhkd->nhad", agent_attn, v)

        q_attn = torch.einsum("nhqd,nhad->nhqa", q, agent_tokens) * self.scale
        q_attn = F.softmax(q_attn, dim=-1)
        out = torch.einsum("nhqa,nhad->nhqd", q_attn, agent_v)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out


class Projection(nn.Module):
    
    def __init__(self, dim_in, dim_out, hidden_dim=None):
        super().__init__()
        
        hidden_dim = hidden_dim or dim_in

        self.layers = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_out)
        )

    def forward(self, x):
        
        return self.layers(x)


class DynamicProjection(nn.Module):
    
    def __init__(self, dim, num_modalities):
        super().__init__()
        
        self.dim = dim
        self.num_modalities = num_modalities

        self.mlps = nn.ModuleDict()
        for i in range(1, 2**num_modalities-1):
            modality_comb = bin(i)[2:].zfill(num_modalities)
            dim_in = dim * sum(int(b) for b in modality_comb)
            self.mlps[modality_comb] = Projection(dim_in, dim)

    def forward(self, x, mask):
        modality_comb = ''.join(str(int(m)) for m in mask.tolist())
        selected_mlp = self.mlps[modality_comb]
        x = torch.cat(x, dim=-1)
        x = selected_mlp(x)
        
        return x


class MMTCBlock(nn.Module):
    
    def __init__(self, dim, num_modalities=4, num_heads=8, num_tokens=8, loss_type="smooth_l1"):
        super().__init__()

        self.dim = dim
        self.num_modalities = num_modalities
        self.loss_type = loss_type

        self.agent_tokens = nn.ParameterList(
            [nn.Parameter(torch.randn(1, num_tokens, dim)) for _ in range(num_modalities)]
        )
        for agent_tokens in self.agent_tokens:
            trunc_normal_(agent_tokens, std=0.02)

        self.update_cross_attention = nn.ModuleList([CrossAttention(dim, num_heads) for _ in range(num_modalities)])
        self.agent_cross_attention = CrossAttention(dim, num_heads)
        self.agent_attention = AgentAttention(dim, num_heads)

        self.masked_proj = DynamicProjection(dim, num_modalities)
        
        loss_options = {
            "smooth_l1": nn.SmoothL1Loss(),
            "mse": nn.MSELoss(),
            "kl": nn.KLDivLoss(reduction="batchmean"),
            "mae": nn.L1Loss(),
            "huber": nn.SmoothL1Loss(),
            "log_cosh": self._log_cosh_loss
        }
        self.alignment_loss = loss_options.get(loss_type, None)
    
    def _log_cosh_loss(self, preds, targets):
        return torch.mean(torch.log(torch.cosh(preds - targets)))

    def forward(self, real_tokens, real_mask):
        B, C, hh, ww, dd = real_tokens[0].shape
        real_tokens = [rt_.permute(0, 2, 3, 4, 1).flatten(1, 3) for rt_ in real_tokens]
        tokens = [rt_.clone() for rt_ in real_tokens]
        mask = torch.zeros_like(real_mask, dtype=torch.bool, device=real_mask.device)
        batch_alignment_losses = []

        for b in range(real_mask.shape[0]):
            real_available_modalities = [i for i in range(self.num_modalities) if real_mask[b, i]]
            masked_modalities = [i for i in range(self.num_modalities) if not mask[b, i]]
            alignment_losses = []

            if self.training:
                updated_agent_tokens = []
                for idx, agent_token in enumerate(self.agent_tokens):
                    if idx in real_available_modalities:
                        updated_token = self.update_cross_attention[idx](agent_token, real_tokens[idx][b].unsqueeze(0))
                        updated_agent_tokens.append(updated_token)
                    else:
                        updated_agent_tokens.append(agent_token)
                agent_tokens = updated_agent_tokens
            else:
                agent_tokens = [at_.detach().clone() for at_ in self.agent_tokens]
                for idx in real_available_modalities:
                    agent_tokens[idx] = self.update_cross_attention[idx](agent_tokens[idx], real_tokens[idx][b].unsqueeze(0))

            for i in masked_modalities:
                attended_tokens = []
                proj_mask = real_mask.clone()
                for j in real_available_modalities:
                    if j != i:
                        agent_cross_tokens = self.agent_cross_attention(agent_tokens[i], agent_tokens[j])
                        attended_j = self.agent_attention(real_tokens[j][b].unsqueeze(0), agent_cross_tokens)
                        attended_tokens.append(attended_j)
                    else:
                        proj_mask[b, j] = 0
                
                if attended_tokens:
                    updated_tokens = self.masked_proj(attended_tokens, proj_mask[b])

                    update_condition = (real_mask[b, i] == 0) and (mask[b, i] == 0)
                    if update_condition:
                        tokens[i][b] = updated_tokens

                    if self.training:
                        loss_condition = (real_mask[b, i] == 1) and (mask[b, i] == 0)
                        if loss_condition and self.alignment_loss is not None:
                            if self.loss_type == "kl":
                                tokens_softmax = torch.log_softmax(updated_tokens, dim=-1)
                                real_tokens_softmax = torch.softmax(real_tokens[i][b], dim=-1)
                                alignment_loss = self.alignment_loss(tokens_softmax, real_tokens_softmax.detach())
                            else:
                                alignment_loss = self.alignment_loss(updated_tokens, real_tokens[i][b].detach())
                            alignment_losses.append(alignment_loss)
            
            if self.training:
                if masked_modalities:
                    avg_loss = sum(alignment_losses) / len(alignment_losses) if alignment_losses else torch.tensor(0.0, device=real_mask.device)
                    batch_alignment_losses.append(avg_loss)

        total_alignment_loss = sum(batch_alignment_losses) / len(batch_alignment_losses) if batch_alignment_losses else torch.tensor(0.0, device=real_mask.device)
        
        tokens = [t_.view(B, hh, ww, dd, C).permute(0, 4, 1, 2, 3) for t_ in tokens]

        return tokens, total_alignment_loss


class Attention_p(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = ModuleParallel(nn.Conv3d(dim, dim * 3, kernel_size=1, stride=1, padding=0, bias=qkv_bias), 'qkv')
        self.attn_drop = ModuleParallel(nn.Dropout(attn_drop), 'attn_drop')
        self.proj = ModuleParallel(nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0), 'proj')
        self.proj_drop = ModuleParallel(nn.Dropout(proj_drop), 'proj_drop')
    
    def forward(self, x):
        B, C, H, W, D = x[0].size()
        N = H * W * D
        qkv = self.qkv(x)

        qkv = [torch.chunk(qkv_, 3, dim=1) for qkv_ in qkv]
        q = [qkv[0][0], qkv[1][0], qkv[2][0], qkv[3][0]]
        k = [qkv[0][1], qkv[1][1], qkv[2][1], qkv[3][1]]
        v = [qkv[0][2], qkv[1][2], qkv[2][2], qkv[3][2]]

        q = [q_.reshape(B * self.num_heads, self.head_dim, N).mul(self.scale) for q_ in q]
        k = [k_.reshape(B * self.num_heads, self.head_dim, N) for k_ in k]
        v = [v_.reshape(B * self.num_heads, self.head_dim, N) for v_ in v]
        attn = [torch.einsum('b c m, b c n -> b m n', q_, k_) for (q_, k_) in zip(q, k)]
        attn = [F.softmax(attn_, dim=2) for attn_ in attn]
        attn = self.attn_drop(attn)

        x = [torch.einsum('b m n, b c n -> b c m', attn_, v_) for (attn_, v_) in zip(attn, v)]
        x = [x_.reshape(B, C, H, W, D) for x_ in x]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

class Unfold(nn.Module):
    
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size ** 3)
        weights = weights.reshape(kernel_size ** 3, 1, kernel_size, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        B, C, H, W, D = x[0].shape
        x = [F.conv3d(x_.reshape(B * C, 1, H, W, D), self.weights, stride=1, padding=self.kernel_size // 2) for x_ in x]  
        
        return [x_.reshape(B, C * (self.kernel_size ** 3), H * W * D) for x_ in x]


class Fold(nn.Module):
    
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size ** 3)
        weights = weights.reshape(kernel_size ** 3, 1, kernel_size, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        x = [F.conv_transpose3d(x_, self.weights, stride=1, padding=self.kernel_size // 2) for x_ in x] 
        
        return x


class SuperAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                 super_token_size=1, n_iter=1, refine=True, refine_attention=True):
        super().__init__()
        
        self.n_iter = n_iter
        self.super_token_size = to_3tuple(super_token_size)
        self.refine = refine
        self.refine_attention = refine_attention  
        
        self.scale = dim ** - 0.5
        
        self.unfold = Unfold(kernel_size=3)
        self.fold = Fold(kernel_size=3)
        
        if refine:
            if refine_attention:
                self.super_refine = Attention_p(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.super_refine = nn.Sequential(
                    ModuleParallel(nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0),'conv1'),
                    ModuleParallel(nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim), 'conv2'),
                    ModuleParallel(nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0), 'conv3'),
                )

    def super_forward(self, x):
        B, C, H0, W0, D0 = x[0].shape
        h, w, d = self.super_token_size

        pad_l = pad_t = pad_f = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        pad_d = (d - D0 % d) % d

        if pad_r > 0 or pad_b > 0 or pad_d > 0:
            x = [F.pad(x_, (pad_l, pad_r, pad_t, pad_b, pad_f, pad_d)) for x_ in x]

        _, _, H, W, D = x[0].shape
        hh, ww, dd = H // h, W // w, D // d

        super_tokens = [F.adaptive_avg_pool3d(x_, (hh, ww, dd)) for x_ in x]  # (B, C, hh, ww, dd)
        
        pixel_tokens = [
            x_.reshape(B, C, hh, h, ww, w, dd, d)
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(B, hh * ww * dd, h * w * d, C) for x_ in x
        ]

        with torch.no_grad():
            for idx in range(self.n_iter):
                super_tokens = self.unfold(super_tokens)  # (B, C*27, hh*ww*dd)
                super_tokens = [s_.transpose(1, 2).reshape(B, hh * ww * dd, C, 27) for s_ in super_tokens]

                affinity_matrix = [p @ s * self.scale for (p, s) in zip(pixel_tokens, super_tokens)]
                affinity_matrix = [m_.softmax(-1) for m_ in affinity_matrix]  # (B, hh*ww*dd, h*w*d, 27)

                affinity_matrix_sum = [m_.sum(2).transpose(1, 2).reshape(B, 27, hh, ww, dd) for m_ in affinity_matrix]
                affinity_matrix_sum = self.fold(affinity_matrix_sum)

                if idx < self.n_iter - 1:
                    super_tokens = [
                        p_.transpose(-1, -2) @ m_ for (p_, m_) in zip(pixel_tokens, affinity_matrix)
                    ]  # (B, hh*ww*dd, C, 27)
                    super_tokens = [
                        s_.permute(0, 2, 3, 1).reshape(B * C, 27, hh, ww, dd) for s_ in super_tokens
                    ]
                    super_tokens = self.fold(super_tokens)
                    super_tokens = [s_.reshape(B, C, hh, ww, dd) for s_ in super_tokens]
                    super_tokens = [s_ / (ms_ + 1e-12) for (s_, ms_) in zip(super_tokens, affinity_matrix_sum)]  # (B, C, hh, ww, dd)

        super_tokens = [p_.transpose(-1, -2) @ m_ for (p_, m_) in zip(pixel_tokens, affinity_matrix)]
        super_tokens = [
            s_.permute(0, 2, 3, 1).reshape(B * C, 27, hh, ww, dd) for s_ in super_tokens
        ]
        super_tokens = self.fold(super_tokens)
        super_tokens = [s_.reshape(B, C, hh, ww, dd) for s_ in super_tokens]
        super_tokens = [s_ / (ms_ + 1e-12) for (s_, ms_) in zip(super_tokens, affinity_matrix_sum)]  # (B, C, hh, ww, dd)

        if self.refine:
            super_tokens = self.super_refine(super_tokens)

        super_tokens = self.unfold(super_tokens)  # (B, C*27, hh*ww*dd)
        super_tokens = [s_.transpose(1, 2).reshape(B, hh * ww * dd, C, 27) for s_ in super_tokens]  # (B, hh*ww*dd, C, 27)
        pixel_tokens = [s_ @ m_.transpose(-1, -2) for (s_, m_) in zip(super_tokens, affinity_matrix)]  # (B, hh*ww*dd, C, h*w*d)
        pixel_tokens = [
            p_.reshape(B, hh, ww, dd, C, h, w, d)
            .permute(0, 4, 1, 5, 2, 6, 3, 7)
            .reshape(B, C, H, W, D) for p_ in pixel_tokens
        ]

        if pad_r > 0 or pad_b > 0 or pad_d > 0:
            pixel_tokens = [p_[:, :, :H0, :W0, :D0] for p_ in pixel_tokens]

        return pixel_tokens
    
    def direct_forward(self, x):
        B, C, H, W, D = x[0].shape
        pixel_tokens = x
        if self.refine:
            pixel_tokens = self.super_refine(pixel_tokens)
        
        return pixel_tokens
        
    def forward(self, x):
        if self.super_token_size[0] > 1 or self.super_token_size[1] > 1 or self.super_token_size[2] > 1:
            return self.super_forward(x)
        else:
            return self.direct_forward(x)


class MaskedSuperAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., super_token_size=1, 
                 num_agent_tokens=32, n_iter=1, alignment_loss_type="smooth_l1", missing_completion=False):
        super().__init__()
        
        self.n_iter = n_iter
        self.super_token_size = to_3tuple(super_token_size)
        
        self.scale = dim ** - 0.5

        self.missing_completion = missing_completion
        
        self.unfold = Unfold(kernel_size=3)
        self.fold = Fold(kernel_size=3)

        ##########
        self.mmtc = MMTCBlock(dim, num_tokens=num_agent_tokens, loss_type=alignment_loss_type) if missing_completion else ModuleParallel(nn.Identity(), 'mmtc')
        ##########


    def super_forward(self, x, mask):
        B, C, H0, W0, D0 = x[0].shape
        h, w, d = self.super_token_size

        pad_l = pad_t = pad_f = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        pad_d = (d - D0 % d) % d

        if pad_r > 0 or pad_b > 0 or pad_d > 0:
            x = [F.pad(x_, (pad_l, pad_r, pad_t, pad_b, pad_f, pad_d)) for x_ in x]

        _, _, H, W, D = x[0].shape
        hh, ww, dd = H // h, W // w, D // d

        super_tokens = [F.adaptive_avg_pool3d(x_, (hh, ww, dd)) for x_ in x]  # (B, C, hh, ww, dd)
        
        pixel_tokens = [
            x_.reshape(B, C, hh, h, ww, w, dd, d)
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(B, hh * ww * dd, h * w * d, C) for x_ in x
        ]

        with torch.no_grad():
            for idx in range(self.n_iter):
                super_tokens = self.unfold(super_tokens)  # (B, C*27, hh*ww*dd)
                super_tokens = [s_.transpose(1, 2).reshape(B, hh * ww * dd, C, 27) for s_ in super_tokens]

                affinity_matrix = [p @ s * self.scale for (p, s) in zip(pixel_tokens, super_tokens)]
                affinity_matrix = [m_.softmax(-1) for m_ in affinity_matrix]  # (B, hh*ww*dd, h*w*d, 27)

                affinity_matrix_sum = [m_.sum(2).transpose(1, 2).reshape(B, 27, hh, ww, dd) for m_ in affinity_matrix]
                affinity_matrix_sum = self.fold(affinity_matrix_sum)

                if idx < self.n_iter - 1:
                    super_tokens = [
                        p_.transpose(-1, -2) @ m_ for (p_, m_) in zip(pixel_tokens, affinity_matrix)
                    ]  # (B, hh*ww*dd, C, 27)
                    super_tokens = [
                        s_.permute(0, 2, 3, 1).reshape(B * C, 27, hh, ww, dd) for s_ in super_tokens
                    ]
                    super_tokens = self.fold(super_tokens)
                    super_tokens = [s_.reshape(B, C, hh, ww, dd) for s_ in super_tokens]
                    super_tokens = [s_ / (ms_ + 1e-12) for (s_, ms_) in zip(super_tokens, affinity_matrix_sum)]  # (B, C, hh, ww, dd)

        super_tokens = [p_.transpose(-1, -2) @ m_ for (p_, m_) in zip(pixel_tokens, affinity_matrix)]
        super_tokens = [
            s_.permute(0, 2, 3, 1).reshape(B * C, 27, hh, ww, dd) for s_ in super_tokens
        ]
        super_tokens = self.fold(super_tokens)
        super_tokens = [s_.reshape(B, C, hh, ww, dd) for s_ in super_tokens]
        super_tokens = [s_ / (ms_ + 1e-12) for (s_, ms_) in zip(super_tokens, affinity_matrix_sum)]  # (B, C, hh, ww, dd)

        ################################################################################
        if self.missing_completion:
            # compute affinity_matrix for missing modalities
            updated_affinity_matrix = [mat.clone() for mat in affinity_matrix]
            for b in range(B):
                available_matrices = [affinity_matrix[j][b] for j in range(len(affinity_matrix)) if mask[b, j]]
                if available_matrices:
                    avg_matrix = sum(available_matrices) / len(available_matrices)
                    for j in range(len(affinity_matrix)):
                        if not mask[b, j]:
                            updated_affinity_matrix[j][b] = avg_matrix.detach()
            affinity_matrix = updated_affinity_matrix

            super_tokens, total_alignment_loss = self.mmtc(super_tokens, mask)
        else:
            total_alignment_loss = torch.tensor(0.0, device=mask.device)
        ################################################################################

        super_tokens = self.unfold(super_tokens)  # (B, C*27, hh*ww*dd)
        super_tokens = [s_.transpose(1, 2).reshape(B, hh * ww * dd, C, 27) for s_ in super_tokens]  # (B, hh*ww*dd, C, 27)
        pixel_tokens = [s_ @ m_.transpose(-1, -2) for (s_, m_) in zip(super_tokens, affinity_matrix)]  # (B, hh*ww*dd, C, h*w*d)
        pixel_tokens = [
            p_.reshape(B, hh, ww, dd, C, h, w, d)
            .permute(0, 4, 1, 5, 2, 6, 3, 7)
            .reshape(B, C, H, W, D) for p_ in pixel_tokens
        ]

        if pad_r > 0 or pad_b > 0 or pad_d > 0:
            pixel_tokens = [p_[:, :, :H0, :W0, :D0] for p_ in pixel_tokens]

        return pixel_tokens, total_alignment_loss
    
    def direct_forward(self, x, mask):
        B, C, H, W, D = x[0].shape
        pixel_tokens = x
        
        ################################################################################
        if self.missing_completion:
            pixel_tokens, total_alignment_loss = self.mmtc(pixel_tokens, mask)
        else:
            total_alignment_loss = torch.tensor(0.0, device=mask.device)
        ################################################################################
        
        return pixel_tokens, total_alignment_loss
        
    def forward(self, x, mask):
        if self.super_token_size[0] > 1 or self.super_token_size[1] > 1 or self.super_token_size[2] > 1:
            return self.super_forward(x, mask)
        else:
            return self.direct_forward(x, mask)


class MLPWithConv_p(nn.Module):
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ModuleParallel(nn.Conv3d(in_features, hidden_features, kernel_size=1, stride=1, padding=0), 'fc1')
        self.dwconv = ModuleParallel(nn.Conv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features), 'dwconv')
        self.act = ModuleParallel(act_layer(), 'act')
        self.drop1 = ModuleParallel(nn.Dropout(drop, inplace=True), 'drop1')
        self.fc2 = ModuleParallel(nn.Conv3d(hidden_features, out_features, kernel_size=1, stride=1, padding=0), 'fc2')
        self.drop2 = ModuleParallel(nn.Dropout(drop, inplace=True), 'drop2')

    def forward(self, x):
        x = self.fc1(x)
        residual = [x_.clone() for x_ in x]
        x = self.dwconv(x)
        x = [x_ + r_ for (x_, r_) in zip(x, residual)]
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        
        return x


class MissingCompletionBlock(nn.Module):
    
    def __init__(self, input_size, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNormProxy,  missing_completion=True, super_token_size=1, 
                 num_agent_tokens=32, n_iter=1, alignment_loss_type="smooth_l1", refine=True, refine_attention=True, 
                 use_lpu=True, pos_embed=False, init_value=0.):
        super().__init__()

        self.use_lpu = use_lpu
        self.missing_completion = missing_completion

        self.norm1 = ModuleParallel(norm_layer(dim), 'norm1')
        self.attn = SuperAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
                                   super_token_size=super_token_size, n_iter=n_iter, refine=refine, refine_attention=refine_attention)
        self.drop_path = ModuleParallel(DropPath(drop_path), 'drop_path') if drop_path > 0. else ModuleParallel(nn.Identity(), 'drop_path')
        self.norm2 = ModuleParallel(norm_layer(dim), 'norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPWithConv_p(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.lpu = ModuleParallel(nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim), 'lpu') if use_lpu else ModuleParallel(nn.Identity(), 'lpu')

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, dim, input_size, input_size, input_size)) for _ in range(4)])
        
        # init layer scale
        self.layer_scale_1 = nn.ParameterList([nn.Parameter(init_value * torch.ones(1, dim, 1, 1, 1), requires_grad=True) for _ in range(4)])
        self.layer_scale_2 = nn.ParameterList([nn.Parameter(init_value * torch.ones(1, dim, 1, 1, 1), requires_grad=True) for _ in range(4)])

        self.maskedsuper = MaskedSuperAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, super_token_size=super_token_size, 
                 num_agent_tokens=num_agent_tokens, n_iter=n_iter, alignment_loss_type=alignment_loss_type, missing_completion=missing_completion) if missing_completion else ModuleParallel(nn.Identity(), 'maskedsuper')

    def forward(self, x, mask):
        if self.use_lpu:
            pos = self.lpu([x_.contiguous() for x_ in x])
            x = [x_ + pos_ for (x_, pos_) in zip (x, pos)]
        elif self.pos_embed is not None:
            x = [x_ + pos_ for (x_, pos_) in zip (x, self.pos_embed)]

        f = self.attn(self.norm1(x))
        f = [scale_ * f_ for (scale_, f_) in zip (self.layer_scale_1, f)]
        f = self.drop_path(f)
        x = [x_ + f_ for (x_, f_) in zip(x, f)]

        f = self.mlp(self.norm2(x))
        f = [scale_ * f_ for (scale_, f_) in zip (self.layer_scale_2, f)]
        f = self.drop_path(f)
        x = [x_ + f_ for (x_, f_) in zip(x, f)]
            
        if self.missing_completion:
            s, alignment_loss = self.maskedsuper(x, mask)
            updated_x = []
            batch_updated_x = []
            for b in range(mask.size(0)):
                available_modalities = [x[k][b] for k in range(len(x)) if mask[b, k]]

                if available_modalities:
                    available_x_mean = torch.stack(available_modalities, dim=0).mean(dim=0)  # (C, H, W, D)
                else:
                    available_x_mean = torch.zeros_like(x[0][b])

                sample_x = []
                for i, (x_, s_) in enumerate(zip(x, s)):
                    if mask[b, i]:
                        sample_x.append(x_[b])
                    else:
                        replacement = s_[b] + available_x_mean
                        sample_x.append(replacement)
                batch_updated_x.append(torch.stack(sample_x, dim=0))  # (K, C, H, W, D)

            updated_x = torch.stack(batch_updated_x, dim=0)  # (B, K, C, H, W, D)
            x = [updated_x[:, i] for i in range(updated_x.size(1))]  # (B, C, H, W, D)
        else:
            alignment_loss = torch.tensor(0.0, device=mask.device)

        return x, alignment_loss




def _grid2seq(x, region_size, num_heads):
    B, C, H, W, D = x[0].shape
    region_h, region_w, region_d = H // region_size[0], W // region_size[1], D // region_size[2]
    x = [x_.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1], region_d, region_size[2]) for x_ in x]
    x = [torch.einsum('bmchpwqdr->bmhwdpqrc', x_).flatten(2, 4).flatten(-4, -2) for x_ in x] # (B, num_heads, nregion, reg_size, head_dim)
    
    return x, region_h, region_w, region_d


def _seq2grid(x, region_h, region_w, region_d, region_size):
    bs, nhead, nregion, reg_size_cube, head_dim = x[0].shape
    x = [x_.view(bs, nhead, region_h, region_w, region_d, region_size[0], region_size[1], region_size[2], head_dim) for x_ in x]
    x = [torch.einsum('bmhwdpqrc->bmchpwqdr', x_).reshape(bs, nhead * head_dim,
        region_h * region_size[0], region_w * region_size[1], region_d * region_size[2]) for x_ in x]
    
    return x


class MultimodalRegionAwareAttention(nn.Module):
    
    def __init__(self, dim, apply_attn_mask=False):
        super().__init__()

        self.apply_attn_mask = apply_attn_mask

    def _apply_attn_mask(self, attn, mask, num_modalities=4):
        B, H, N_r, M, N = attn[0].shape

        mask = mask.view(B, 1, 1, 1, num_modalities).expand(-1, H, N_r, M, -1)
        mask = mask.repeat_interleave(N//num_modalities, dim=-1)

        attn = [attn_.masked_fill(mask == 0, float('-inf')) for attn_ in attn]

        return attn

    def forward(self, mask, query, key, value, scale, region_graph, region_size, kv_region_size=None, auto_pad=True):
        kv_region_size = kv_region_size or region_size
        bs, nhead, q_nregion, topk = region_graph[0].shape
        
        q_pad_b, q_pad_r, q_pad_d = 0, 0, 0
        kv_pad_b, kv_pad_r, kv_pad_d = 0, 0, 0
        if auto_pad:
            _, _, Hq, Wq, Dq = query[0].shape
            q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
            q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
            q_pad_d = (region_size[2] - Dq % region_size[2]) % region_size[2]
            if (q_pad_b > 0 or q_pad_r > 0 or q_pad_d > 0):
                query = [F.pad(q_, (0, q_pad_d, 0, q_pad_r, 0, q_pad_b)) for q_ in query]

            _, _, Hk, Wk, Dk = key[0].shape
            kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
            kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
            kv_pad_d = (kv_region_size[2] - Dk % kv_region_size[2]) % kv_region_size[2]
            if (kv_pad_r > 0 or kv_pad_b > 0 or kv_pad_d > 0):
                key = [F.pad(k_, (0, kv_pad_d, 0, kv_pad_r, 0, kv_pad_b)) for k_ in key]
                value = [F.pad(v_, (0, kv_pad_d, 0, kv_pad_r, 0, kv_pad_b)) for v_ in value]

        query, q_region_h, q_region_w, q_region_d = _grid2seq(query, region_size=region_size, num_heads=nhead)
        key, _, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
        value, _, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

        bs, nhead, kv_nregion, kv_region_size, head_dim = key[0].shape
        broadcasted_region_graph = [rg_.view(bs, nhead, q_nregion, topk, 1, 1).\
            expand(-1, -1, -1, -1, kv_region_size, head_dim) for rg_ in region_graph]
        key_g = [torch.gather(k_.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).\
            expand(-1, -1, query[0].shape[2], -1, -1, -1), dim=3,
            index=index) for (k_, index) in zip(key, broadcasted_region_graph)]  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
        value_g = [torch.gather(v_.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).\
            expand(-1, -1, query[0].shape[2], -1, -1, -1), dim=3,
            index=index) for (v_, index) in zip(value, broadcasted_region_graph)]  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)        
        
        key_g = [kg_.flatten(-3, -2) for kg_ in key_g]
        value_g = [vg_.flatten(-3, -2) for vg_ in value_g]

        key_g = torch.cat(key_g, dim=-2)
        value_g = torch.cat(value_g, dim=-2)

        attn = [(q_ * scale) @ key_g.transpose(-1, -2) for q_ in query]

        if self.apply_attn_mask:
            attn = self._apply_attn_mask(attn, mask)

        attn = [torch.softmax(attn_, dim=-1) for attn_ in attn]
        output = [attn_ @ value_g for attn_ in attn]

        output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_d=q_region_d, region_size=region_size)

        if auto_pad and (q_pad_b > 0 or q_pad_r > 0 or q_pad_d > 0):
            output = [output_[:, :, :Hq, :Wq, :Dq] for output_ in output]

        return output, attn


class RegionAwareAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, n_win=1, topk=1, apply_attn_mask=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5
        
        self.topk = topk
        self.n_win = n_win

        self.lepe = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.qkv = nn.Conv3d(dim, 3 * dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.attn_fn = MultimodalRegionAwareAttention(dim, apply_attn_mask=apply_attn_mask)

    def forward(self, x, mask, ret_attn_mask=False):
        B, C, H, W, D = x[0].shape
        region_size = (H // self.n_win, W // self.n_win, D // self.n_win)

        q_m, k_m, v_m, idx_r_m = [], [], [], []
        for x_ in x:
            qkv = self.qkv(x_)  # ncHWD
            q, k, v = qkv.chunk(3, dim=1)  # ncHWD

            q_r = F.avg_pool3d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
            k_r = F.avg_pool3d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)  # nchwd
            q_r = q_r.permute(0, 2, 3, 4, 1).flatten(1, 3)  # n(hwd)c
            k_r = k_r.flatten(2, 4)  # nc(hwd)
            a_r = q_r @ k_r  # n(hwd)(hwd), adj matrix of regional graph
            _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hwd)k long tensor
            idx_r = idx_r.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            q_m.append(q)
            k_m.append(k)
            v_m.append(v)
            idx_r_m.append(idx_r)

        x, attn_mat = self.attn_fn(mask=mask, query=q_m, key=k_m, value=v_m, scale=self.scale,
                                   region_graph=idx_r_m, region_size=region_size)

        out = []
        for (x_, v_) in zip(x, v_m):
            x_ = x_ + self.lepe(v_)  # ncHWD
            x_ = self.proj(x_)  # ncHWD
            out.append(x_)

        if ret_attn_mask:
            return out, attn_mat

        return out


class MLPWithConv(nn.Module):
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.drop2 = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        residual = x.clone()
        x = self.dwconv(x)
        x = x + residual
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        
        return x


class HighLevelFusionBlock(nn.Module):
    
    def __init__(self, input_size, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=LayerNormProxy, 
                 n_win=1, topk=1, apply_attn_mask=True, use_lpu=True, pos_embed=False, init_value=0.):
        super().__init__()

        self.use_lpu = use_lpu

        self.norm1 = norm_layer(dim)
        self.attn = RegionAwareAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, n_win=n_win, topk=topk, apply_attn_mask=apply_attn_mask)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPWithConv(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.lpu = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) if use_lpu else nn.Identity()

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, dim, input_size, input_size, input_size)) for _ in range(4)])
        
        # init layer scale
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(1, dim, 1, 1, 1), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(1, dim, 1, 1, 1), requires_grad=True)


    def forward(self, x, mask):
        if self.use_lpu:
            x = [x_ + self.lpu(x_.contiguous()) for x_ in x]
        elif self.pos_embed is not None:
            x = [x_ + pos_ for (x_, pos_) in zip (x, self.pos_embed)]

        f = [self.norm1(x_) for x_ in x]
        f = self.attn(f, mask)
        x = [x_ + self.drop_path(self.layer_scale_1 * f_) for (x_, f_) in zip(x, f)]
        x = [x_ + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x_))) for x_ in x]

        return x


class ChannelAggregation(nn.Module):
    
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, norm_layer=LayerNormProxy, drop=0., expand_ratio=4, init_value=0.):
        super().__init__()
        
        hidden_dim = hidden_dim or dim * expand_ratio

        self.dim_conv = dim // 4
        self.dim_untouched = dim - self.dim_conv 
        self.partial_conv = nn.Conv3d(self.dim_conv, self.dim_conv, kernel_size=3, stride=1, padding=1, bias=False, groups=self.dim_conv)

        self.norm = norm_layer(dim)
        self.linear1 = nn.Conv3d(dim, hidden_dim*2, kernel_size=1, stride=1, padding=0)
        self.dwconv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.linear2 = nn.Conv3d(hidden_dim, dim, kernel_size=1, stride=1, padding=0)
        self.drop2 = nn.Dropout(drop, inplace=True)

        # init layer scale
        self.layer_scale = nn.Parameter(init_value * torch.ones(1, dim, 1, 1, 1), requires_grad=True)

    def forward(self, x):
        x = torch.cat(x, dim=1)

        residual = x.clone()

        x = self.norm(x)

        x1, x2,= torch.split(x, [self.dim_conv,self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), 1)

        x = self.linear1(x)
        x, value = x.chunk(2,dim=1)

        x = self.act(self.dwconv(x)) * value
        x = self.drop1(x)
        
        x = self.linear2(x)
        x = self.drop2(x)

        x = self.layer_scale * x + residual

        return torch.chunk(x, 4, dim=1)


class SpatialAggregation(nn.Module):

    def __init__(self, dim, act_layer=nn.SiLU, dw_dilation=[1, 2, 2], channel_split=[1, 3, 4], init_value=0.):
        super().__init__()

        self.dim = dim
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.dim_1 = int(self.split_ratio[1] * dim)
        self.dim_2 = int(self.split_ratio[2] * dim)
        self.dim_0 = dim - self.dim_1 - self.dim_2
        
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert dim % sum(channel_split) == 0

        self.norm = nn.GroupNorm(num_groups=self.dim//8, num_channels=self.dim)
        self.act = act_layer()

        self.linear1 = nn.Conv3d(self.dim, self.dim*2, kernel_size=1, stride=1, padding=0)
        self.linear2 = nn.Conv3d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.linear3 = nn.Conv3d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

        self.dwconv0 = nn.Conv3d(self.dim, self.dim, kernel_size=7, padding=3, stride=1, groups=self.dim, dilation=dw_dilation[0])
        self.dwconv1 = nn.Conv3d(self.dim_1, self.dim_1, kernel_size=3, padding=2, stride=1, groups=self.dim_1, dilation=dw_dilation[1])
        self.dwconv2 = nn.Conv3d(self.dim_2, self.dim_2, kernel_size=5, padding=4, stride=1, groups=self.dim_2, dilation=dw_dilation[2])

        # init layer scale
        self.layer_scale = nn.Parameter(init_value * torch.ones(1, dim, 1, 1, 1), requires_grad=True)

    def forward(self, x):
        x = torch.cat(x,dim=1)
        residual = x.clone()
        
        x = self.norm(x)

        x = self.linear1(x)
        x, value = x.chunk(2,dim=1)

        x_0 = self.dwconv0(x)
        x_1 = self.dwconv1(
            x_0[:, self.dim_0: self.dim_0+self.dim_1, ...])
        x_2 = self.dwconv2(
            x_0[:, self.dim-self.dim_2:, ...])
        x = torch.cat([
            x_0[:, :self.dim_0, ...], x_1, x_2], dim=1)
        x = self.linear2(x)

        x = self.act(x) * self.act(value)

        x = self.linear3(x)

        x = self.layer_scale * x + residual

        return torch.chunk(x, 4, dim=1)


class LowLevelFusionBlock(nn.Module):
    
    def __init__(self, dim, num_modalities=4, apply_ca=False, apply_sa=False, dw_dilation=[1, 2, 2], channel_split=[1, 3, 4], init_value=0.):
        super().__init__()
        
        self.apply_ca = apply_ca
        self.apply_sa = apply_sa

        self.channel_aggregation = ChannelAggregation(dim*num_modalities, init_value=init_value) if apply_ca else ModuleParallel(nn.Identity(), 'channel_aggregation')
        self.spatial_aggregation = SpatialAggregation(dim*num_modalities, dw_dilation=dw_dilation, channel_split=channel_split, init_value=init_value) if apply_sa else ModuleParallel(nn.Identity(), 'spatial_aggregation')

    def forward(self, x):
        if self.apply_ca:
            x = self.channel_aggregation(x)
        
        if self.apply_sa:
            x = self.spatial_aggregation(x)

        return x


class MultimodalFusionBlock(nn.Module):
    
    def __init__(self, input_size, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=LayerNormProxy, 
                 apply_fusion=True, n_win=1, topk=1, apply_attn_mask=True, use_lpu=True, pos_embed=False, init_value=0.,
                 apply_aggregation=False, apply_ca=False, apply_sa=False, dw_dilation=[1, 2, 2], channel_split=[1, 3, 4]):
        super().__init__()

        self.apply_fusion = apply_fusion
        self.apply_aggregation = apply_aggregation

        self.fusion = HighLevelFusionBlock(input_size=input_size, dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
                 attn_drop=attn_drop,drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, n_win=n_win, topk=topk, 
                 apply_attn_mask=apply_attn_mask, use_lpu=use_lpu, pos_embed=pos_embed, init_value=init_value) if apply_fusion else ModuleParallel(nn.Identity(), 'fusion')

        self.aggregation = LowLevelFusionBlock(dim=dim, apply_ca=apply_ca, apply_sa=apply_sa, 
                                               dw_dilation=dw_dilation, channel_split=channel_split, init_value=init_value) if apply_aggregation else ModuleParallel(nn.Identity(), 'aggregation')

    def forward(self, x, mask):
        if self.apply_fusion:
            x = self.fusion(x, mask)

        if self.apply_aggregation:
            x = self.aggregation(x)
        
        return x