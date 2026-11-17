""" the ViSpec draft (Kang et al., "ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative
Decoding", arXiv 2509.15235; official code refs/ViSpec) for the image-to-SVG blocks of table 1, ported from
refs/ViSpec/vispec/model/cnets_ours.py (class Model) to plain torch so that training (baselines/train_vispec.py) and
the strict tree verifier (baselines/vispec_spec.py) run the same arithmetic.

Architecture (= cnets_ours.Model with the official Qwen2.5-VL-7B draft config, refs/ViSpec/vispec/train/
qwen2.5_vl_7B_config.json: one decoder layer, hidden 3584, 28 heads, 28 KV heads, MLP 18944, qkv bias, rope_theta 1e6):
  - EAGLE-1/2 input: x_t = fc(cat(emb(x_{t+1}), h_t)) with h_t the target's last hidden state (after the final norm,
    HF hidden_states[-1]) at t; the decoder layer has no input norm (index 0); its output is read by the TARGET's
    lm_head (no draft head, no final norm) and is also the hidden state fed to deeper draft steps.
  - vision adaptor (ViSpec): the image tokens' input embeddings (the vision features the target merged into its input)
    are compressed by ImgAdaptor (num_q = 2 learned queries, cross-attention); the first num_q - 1 compressed vectors
    replace the image span in the draft's sequence, at the span's last positions; the last one is the global image
    vector g, and every text position after the image uses img_fc(cat(h_t, g)) instead of h_t (g = 0 before the image).
    img_fc is initialised to [I | 0] (identity on h_t), so a stage-1 checkpoint is reproduced exactly at the start of
    stage 2.
  - stage 1 of the official recipe (refs/ViSpec/vispec/train/main.py, plain EAGLE on text data) has no img_fc / adaptor:
    use_img = False skips them (x_t = fc(cat(emb, h_t))).
  - positions: 1-D RoPE at the ORIGINAL sequence positions (the compressed image span keeps its last positions);
    cos / sin computed in fp32 and cast to the query dtype.
The token embedding and lm_head are the target's (frozen; not stored in the head).

optional per-channel feature scale (cfg "feat_norm": true; OmniSVG only): the head divides the incoming target
features by a fixed vector sigma (buffer feat_scale, estimated once from training rows) and multiplies its output by
sigma, so it reads and writes raw-scale features (inference code unchanged) while its own arithmetic and the stage-1
regression run on h / sigma. Reason: OmniSVG-8B's last hidden states carry a channel ~7x larger than InternSVG-8B's on
every token (per-token max |h| median 87 vs 12, runs/2026-10-31 notes), and ViSpec's stage-1 regression diverged on it
three times. Heads without "feat_norm" have no buffer and behave exactly as before.

optional output norm (cfg "out_norm": true; OmniSVG only, the table's "ViSpec†"): an RMSNorm on the decoder
layer's output, initialised with the target's final-norm weights and trained, so the frozen lm_head (and the
regression onto the target's post-norm h, and the next draft step) read a normalised vector, as in the target itself and
in EAGLE-3's draft. Reason: on OmniSVG-8B the official head (residual stream read directly by the frozen lm_head)
diverged in every setting tried - lr 6e-5 / 3e-5 / 1e-5, norm clipping, per-channel feature scaling (feat_norm), and
no regression term at all (runs/2026-10-31/stress_vispec2.sh) - while EAGLE-3, which has an output norm, trains on the
same target. Heads without "out_norm" have no such module and behave exactly as before.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_config(target_cfg: dict) -> dict:
    """ViSpec's Qwen2.5-VL-7B draft config, with the target LM's sizes (all 7B Qwen2 LMs here have the same shapes)."""
    return {"architectures": ["ViSpecDraft"], "hidden_size": target_cfg["hidden_size"],
            "num_attention_heads": target_cfg["num_attention_heads"],
            "num_key_value_heads": target_cfg["num_attention_heads"],          # official draft: full MHA (28 / 28)
            "intermediate_size": target_cfg["intermediate_size"], "rms_norm_eps": target_cfg.get("rms_norm_eps", 1e-6),
            "rope_theta": target_cfg["rope_theta"], "qkv_bias": True, "num_q": 2, "vocab_size": target_cfg["vocab_size"],
            "use_img": False}


class RMSNorm(nn.Module):
    def __init__(self, n, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class ImgAdaptor(nn.Module):
    """= cnets_ours.ImgAdaptor: num_q learned queries attend (non-causally) over the image embeddings."""

    def __init__(self, H, nh, num_q, bias):
        super().__init__()
        self.H, self.nh, self.hd, self.num_q = H, nh, H // nh, num_q
        self.q = nn.Parameter(torch.empty(num_q, nh, self.hd))
        nn.init.normal_(self.q, mean=0, std=self.hd ** -0.5)
        self.k_proj = nn.Linear(H, H, bias=bias)
        self.v_proj = nn.Linear(H, H, bias=bias)
        self.o_proj = nn.Linear(H, H, bias=False)

    def forward(self, x):                                  # (n_img, H) -> (num_q, H)
        n = x.shape[0]
        q = self.q.to(x.dtype).transpose(0, 1)[None]       # (1, nh, num_q, hd)
        k = self.k_proj(x).view(1, n, self.nh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(1, n, self.nh, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.o_proj(o.transpose(1, 2).reshape(self.num_q, self.H))


class ViSpecDraft(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        H, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.H, self.nh, self.nkv, self.hd = H, nh, nkv, H // nh
        bias = bool(cfg.get("qkv_bias", True))
        self.fc = nn.Linear(2 * H, H, bias=True)
        self.img_fc = nn.Linear(2 * H, H, bias=True)
        self.imadpt = ImgAdaptor(H, nh, int(cfg.get("num_q", 2)), bias)
        self.q_proj = nn.Linear(H, nh * self.hd, bias=bias)
        self.k_proj = nn.Linear(H, nkv * self.hd, bias=bias)
        self.v_proj = nn.Linear(H, nkv * self.hd, bias=bias)
        self.o_proj = nn.Linear(nh * self.hd, H, bias=False)
        self.post_attention_layernorm = RMSNorm(H, cfg.get("rms_norm_eps", 1e-6))
        I = cfg["intermediate_size"]
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.reset_img_fc()
        self.use_img = bool(cfg.get("use_img", False))
        self.register_buffer("inv_freq", self._inv_freq(), persistent=False)
        self.feat_norm = bool(cfg.get("feat_norm", False))
        if self.feat_norm:
            self.register_buffer("feat_scale", torch.ones(H))           # persistent: saved with the head
        self.has_out_norm = bool(cfg.get("out_norm", False))
        assert not (self.feat_norm and self.has_out_norm), "feat_norm and out_norm are alternatives"
        if self.has_out_norm:
            self.out_norm = RMSNorm(H, cfg.get("rms_norm_eps", 1e-6))

    def _inv_freq(self):
        return 1.0 / (float(self.cfg["rope_theta"]) ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))

    @torch.no_grad()
    def reset_img_fc(self):
        H = self.H
        self.img_fc.weight.zero_()
        self.img_fc.weight[:, :H].copy_(torch.eye(H))
        self.img_fc.bias.zero_()

    # ------------------------------------------------------------------ pieces
    def input_proj(self, emb, hid, g=None):
        """emb, hid: (..., H); g: (H,) global image vector or None (= 0). -> fc input of the decoder layer."""
        if self.feat_norm:
            hid = (hid.float() / self.feat_scale.float()).to(hid.dtype)
        if self.use_img:
            gg = torch.zeros_like(hid) if g is None else g.to(hid.dtype).expand_as(hid)
            hid = self.img_fc(torch.cat((hid, gg), dim=-1))
        return self.fc(torch.cat((emb.to(hid.dtype), hid), dim=-1))

    def rope(self, x, pos):
        """x: (1, heads, q, hd); pos: (q,) long."""
        f = pos.to(torch.float32)[:, None] * self.inv_freq.to(x.device)[None, :]
        emb = torch.cat((f, f), dim=-1)
        cos, sin = emb.cos().to(x.dtype)[None, None], emb.sin().to(x.dtype)[None, None]
        return x * cos + rotate_half(x) * sin

    def layer(self, x, pos, mask, past=None, is_causal=False):
        """x: (1, q, H) layer input (input_proj output); pos: (q,); mask: additive (1, 1, q, S) over [past | x] or None
        with is_causal (no past). Returns (out (1, q, H), k (1, nkv, q, hd), v)."""
        q_len = x.shape[1]
        q = self.q_proj(x).view(1, q_len, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(1, q_len, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(1, q_len, self.nkv, self.hd).transpose(1, 2)
        q, k = self.rope(q, pos), self.rope(k, pos)
        K, V = k, v
        if past is not None:
            K, V = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            K = K[:, :, None].expand(1, self.nkv, rep, *K.shape[2:]).reshape(1, self.nh, *K.shape[2:])
            V = V[:, :, None].expand(1, self.nkv, rep, *V.shape[2:]).reshape(1, self.nh, *V.shape[2:])
        o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, is_causal=is_causal and mask is None)
        h = x + self.o_proj(o.transpose(1, 2).reshape(1, q_len, self.nh * self.hd))
        y = self.post_attention_layernorm(h)
        h = h + self.down_proj(F.silu(self.gate_proj(y)) * self.up_proj(y))
        if self.has_out_norm:
            h = self.out_norm(h)
        if self.feat_norm:
            h = (h.float() * self.feat_scale.float()).to(h.dtype)
        return h, k, v

    # ------------------------------------------------------------------ the draft sequence of a context
    def build_inputs(self, emb, hid, span, g0=None):
        """The draft sequence of EAGLE pairs t = 0.. n-1 (emb[t] = input embedding of x_{t+1}, hid[t] = h_t), with the
        image span [a, b) (pairs whose x_{t+1} is an image token; None = no image) compressed as ViSpec does.
        Returns (X (m, H) layer inputs, pos (m,) original positions, idx (m,) full index of each element, g (H,) or g0)."""
        n = emb.shape[0]
        dev = emb.device
        if span is None or not self.use_img:
            X = self.input_proj(emb, hid, g0)
            ar = torch.arange(n, device=dev)
            return X, ar, ar, g0
        a, b = span
        c = self.imadpt(emb[a:b].to(self.fc.weight.dtype))                # (num_q, H)
        nq1 = c.shape[0] - 1
        g = c[-1]
        after = self.input_proj(emb[b:], hid[b:], g)
        parts = [self.input_proj(emb[:a], hid[:a], g0), c[:-1], after]
        pos = torch.cat((torch.arange(0, a, device=dev), torch.arange(b - nq1, b, device=dev),
                         torch.arange(b, n, device=dev)))
        return torch.cat([p.to(after.dtype) for p in parts], dim=0), pos, pos, g

    def forward_seq(self, emb, hid, span):
        """Training / teacher-forced forward over a whole context (causal over the draft sequence).
        Returns (out (n, H) scattered back to the full positions, zeros where the sequence has no element)."""
        X, pos, idx, _ = self.build_inputs(emb, hid, span)
        h, _, _ = self.layer(X[None], pos, None, is_causal=True)
        out = torch.zeros(emb.shape[0], self.H, device=h.device, dtype=h.dtype)
        return out.index_copy(0, idx, h[0])


def save_head(draft: ViSpecDraft, out: Path):
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in draft.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(draft.cfg, indent=2))


def load_head(path, dtype=torch.bfloat16, device="cuda", use_img=None) -> ViSpecDraft:
    from safetensors.torch import load_file
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    if use_img is not None:
        cfg["use_img"] = bool(use_img)
    m = ViSpecDraft(cfg)
    m.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    m = m.to(device=device, dtype=dtype).eval()
    m.inv_freq = m._inv_freq().to(device)            # non-persistent: rebuilt in fp32 (memory hf-from-pretrained-...)
    return m


def is_vispec_head(path) -> bool:
    try:
        cfg = json.loads((Path(path) / "config.json").read_text())
    except Exception:
        return False
    return "ViSpecDraft" in (cfg.get("architectures") or [])
