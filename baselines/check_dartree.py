""" checks of the DARTree port (baselines/dartree_spec.py) on a Domino head of train_domino.py.
 (a) our GRU + correction MLP vs SpecForge's DominoDraftModel.compute_correction_logits on the same inputs;
 (b) the tree code with W = 1, E = full vocabulary, B = depth, no depth bonus vs the reference Domino chain loop
     (refs/DARTree/utils/draft_model.spec_generate) on the same backbone states of a real prompt.
  PYTHONPATH=refs/SpecForge python baselines/check_dartree.py <domino head>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import torch
from sightline.paths import REFS

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "h2", ROOT / "e3", ROOT / "src", ROOT / "t7", REFS / "SpecForge"):
    sys.path.insert(0, str(p))
from baselines.dartree import DominoTreeDrafter, load_domino, reference_chain  # noqa: E402


@torch.inference_mode()
def main(head_path):
    from transformers import AutoModelForCausalLM
    from eval_b3 import MODEL, load_rows
    target = AutoModelForCausalLM.from_pretrained(MODEL / "OmniSVG1.1_8B-qwen2-text", torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa").cuda().eval()
    head = load_domino(head_path)
    dev, emb = "cuda", target.get_input_embeddings().weight
    # (a) correction logits
    from specforge.modeling.draft.domino import DominoDraftModel
    ref = DominoDraftModel.from_pretrained(head_path, torch_dtype=torch.bfloat16).cuda().eval()
    torch.manual_seed(0)
    bs, H = int(head.block_size), head.config.hidden_size
    hid = torch.randn(1, 1, bs, H, device=dev, dtype=torch.bfloat16)
    toks = torch.randint(0, 150000, (bs,), device=dev)
    c_ref = ref.compute_correction_logits(prev_token_embeddings=emb[toks][None, None], hidden_states=hid).float()[0, 0]
    dr = DominoTreeDrafter(head, target, bs + 1)
    g = dr._gru(toks[:1], torch.zeros((1, dr.gru.hidden_size), device=dev, dtype=torch.bfloat16))
    ours = []
    for j in range(1, bs):
        g = dr._gru(toks[j:j + 1], g)
        ours.append(head.domino.proj(torch.cat([hid[0, 0, j:j + 1], g], dim=-1)).float()[0])
    ours = torch.stack(ours)
    print(f"(a) correction logits: shape {tuple(c_ref.shape)} vs {tuple(ours.shape)}, max |diff| "
          f"{float((c_ref - ours).abs().max()):.4g} (scale {float(c_ref.abs().max()):.3g}), argmax agree "
          f"{float((c_ref.argmax(-1) == ours.argmax(-1)).float().mean()):.3f}")
    # (b) chain equivalence on a real prompt
    for r in load_rows(["bench"])[:3]:
        prompt = r["prompt_input_ids"]
        out = target(torch.tensor([prompt], device=dev), output_hidden_states=True)
        first = int(out.logits[0, -1].argmax())
        chain_dr = DominoTreeDrafter(head, target, bs + 1, width=1, cand=int(emb.shape[0]), budget=bs, depth_bonus=0.0)
        chain_dr.prefill_done(out.hidden_states)
        committed = list(prompt) + [first]
        h, base = chain_dr.block_states(committed, len(committed) - 1, bs)
        want = reference_chain(chain_dr, committed, h, base)
        nodes = chain_dr.tree_from_states(committed, h, base)
        got = [n.token for n in sorted(nodes, key=lambda n: n.depth)]
        tree = DominoTreeDrafter(head, target, bs + 1).tree_from_states(committed, h, base)
        print(f"(b) gidx {r['gidx']}: chain identical {got == want} ({len(got)} tokens); default tree "
              f"{len(tree)} nodes, depth max {max(n.depth for n in tree)}, parents-first "
              f"{all(n.parent < i for i, n in enumerate(tree))}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
