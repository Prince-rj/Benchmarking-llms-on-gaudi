import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class SoftmaxAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.heads = heads

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads
        q = self.q_proj(x).reshape(B, T, H, D // H)
        k = self.k_proj(x).reshape(B, T, H, D // H)
        v = self.v_proj(x).reshape(B, T, H, D // H)

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        k_sum = k.sum(dim=1, keepdim=True)
        D_inv = 1.0 / torch.einsum('bthd,bThd->bth', q, k_sum + 1e-6).unsqueeze(-1)
        kv = torch.einsum('bThd,bThv->bhdv', k, v)
        out = torch.einsum('bthd,bhdv->bthv', q, kv)
        return out * D_inv





class TransformerBlock(nn.Module):
    def __init__(self, dim, attn_type="softmax", heads=4):
        super().__init__()
        if attn_type == "softmax":
            self.attn = SoftmaxAttention(dim, heads)
        elif attn_type == "linear":
            self.attn = LinearAttention(dim, heads)
        elif attn_type == "flash":
            self.attn = FlashAttentionWrapper(dim, heads)
        else:
            raise ValueError("Unknown attention type")

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


def run(attn_type, device="cuda"):
    model = TransformerBlock(128, attn_type=attn_type).to(device)
    x = torch.randn(8, 256, 128).to(device)


    for _ in range(3):
        _ = model(x)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(10):
            y = model(x)

    torch.cuda.synchronize()
    end = time.time()

    print(f"{attn_type.upper()} Attention - Time for 10 runs: {end - start:.4f} sec")



if __name__ == "__main__":
    torch.manual_seed(42)
    print("Running Transformer Variants...\n")

    run("softmax")
    run("linear")

