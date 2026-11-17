"""CPU checks for imports, token offsets, rendering and the A1 runner's protocol."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
from run import ROOT, REF, decode, generation_config, load_official
from sightline.paths import RUNS


def main():
    official = load_official()
    official.svg_tokenizer = official.SVGTokenizer(str(REF / "config.yaml"), model_size="8B")
    tok = official.svg_tokenizer
    assert torch.tensor([1]).numpy().tolist() == [1]  # Catch Torch/NumPy ABI failures.
    assert tok.CMD_TOKEN_START == 152067 and tok.COORD_TOKEN_START == 152072
    start = tok.COORD_TOKEN_START + 20 + 200 * 20
    ids = [152067, start, start,
           152068, tok.COORD_TOKEN_START + 180 + 200 * 20,
           152068, tok.COORD_TOKEN_START + 180 + 200 * 180,
           152068, tok.COORD_TOKEN_START + 20 + 200 * 180,
           152071, start, 192077, official.EOS_TOKEN_ID]
    wrapped = torch.tensor([[official.BOS_TOKEN_ID] + ids + [official.EOS_TOKEN_ID]])
    xy = tok.process_generated_tokens(wrapped)
    tensors, _ = tok.raster_svg(xy)
    assert tensors[0][0][0, 12:14].tolist() == [20, 20]
    svg, image, valid, reason = decode(official, ids, "icon")
    assert valid and reason == "ok" and image.size == (1024, 1024)
    assert np.array(image)[512, 512].tolist() == [0, 0, 0]
    assert np.array(image)[0, 0].tolist() == [255, 255, 255]

    # Compare our decoder with the actual official candidate routine on fake outputs.
    class FakeTarget(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.last_config = None

        def generate(self, input_ids, attention_mask, **kwargs):
            self.last_config = kwargs
            return torch.cat([input_ids, torch.tensor([ids])], dim=1).repeat(kwargs["num_return_sequences"], 1)

    target = FakeTarget()
    official.sketch_decoder = SimpleNamespace(transformer=target)
    inputs = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}
    candidates = official.generate_candidates(inputs, "text-to-svg", "icon", 0.5, 0.88, 50, 1.05, 1536, 4)
    assert len(candidates) == 4 and target.last_config["num_return_sequences"] == 8
    assert all(c["svg"] == svg for c in candidates)
    sample_cfg = generation_config(official, "icon", "sample", 8, 1536)
    for k, v in sample_cfg.items():
        assert target.last_config[k] == v, (k, target.last_config.get(k), v)
    assert generation_config(official, "illustration", "greedy-penalty", 1, 1536)["repetition_penalty"] == 1.03
    assert generation_config(official, "icon", "greedy-pure", 1, 1536)["repetition_penalty"] == 1.0

    # Check every downloaded benchmark input image without loading model weights.
    for p in (RUNS / "2026-10-07-omnisvg-inputs").glob("image2svg-*/images/*.png"):
        with Image.open(p) as im:
            im.verify()
    # Quantify the upstream heuristic's disagreement with actual benchmark labels.
    mismatches = []
    for p in (RUNS / "2026-10-07-omnisvg-inputs").glob("text2svg-*/manifest.jsonl"):
        for line in p.read_text().splitlines():
            row = json.loads(line)
            predicted = official.detect_text_subtype(row["text"])
            if predicted != row["subtype"]:
                mismatches.append({"id": row["id"], "benchmark": row["subtype"], "heuristic": predicted})
    out = RUNS / "2026-10-07-omnisvg-preflight"
    image.save(out / "synthetic-tokenizer-check.png")
    (out / "subtype-mismatches.json").write_text(json.dumps(mismatches, indent=2))
    print(json.dumps({"status": "passed", "torch": torch.__version__,
                      "coordinate_zero_id": tok.COORD_TOKEN_START,
                      "subtype_mismatches_out_of_300": len(mismatches)}))


if __name__ == "__main__":
    main()
