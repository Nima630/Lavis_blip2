import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        B, N, D = x.shape
        H = self.heads

        q = self.to_q(x).view(B, N, H, D // H).transpose(1, 2)  # [B, H, N, D/H]
        k = self.to_k(context).view(B, -1, H, D // H).transpose(1, 2)  # [B, H, S, D/H]
        v = self.to_v(context).view(B, -1, H, D // H).transpose(1, 2)  # [B, H, S, D/H]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, S]
        attn = attn.softmax(dim=-1)

        out = attn @ v  # [B, H, N, D/H]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, heads, dropout)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.cross_attn(self.ln2(x), context)
        x = x + self.mlp(self.ln3(x))
        return x

class MiniQFormer(nn.Module):
    def __init__(self, num_query_tokens=8, encoder_width=768, depth=4, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, encoder_width))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(encoder_width, heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.ln_final = nn.LayerNorm(encoder_width)

    def forward(self, encoder_feats):
        B = encoder_feats.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_query_tokens, D]
        for blk in self.transformer_blocks:
            queries = blk(queries, encoder_feats)
        return self.ln_final(queries)

# # Export class so the user can copy/paste or save
# from inspect import getsource
# import textwrap

# mini_qformer_code = textwrap.dedent(getsource(MiniQFormer))
# mini_qformer_code = mini_qformer_code + "\n\n" + textwrap.dedent(getsource(TransformerBlock))
# mini_qformer_code = mini_qformer_code + "\n\n" + textwrap.dedent(getsource(CrossAttention))
