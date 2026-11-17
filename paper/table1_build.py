""" table A v11 (= paper table 1) = runs/2026-10-31/tableA_v10.py plus ViSpec in the OmniSVG-8B image-to-SVG block.

The out-norm recipe (n31-vs-omnid: an output RMSNorm initialised from the target's last norm, official 3e-5) trained
without divergence; watch_1031 picked VS-k8-d4-N29 on 60 held-out rows (tokens per round 3.31, ~2.07x; d12-N127 had tokens per round 4.27 but ~1.78x)
and ran 6 + 8 shards (1,800 own + 1,800 SArena rows) and the fp32 audit (10/10 identical). The row is marked ViSpec+ in the
paper (architecture change, appendix). This is the only change from v10 (VS_ROOTS is read at call time).
  python3 runs/2026-11-02/tableA_v11.py [--paper | --compact] > runs/2026-11-02/tableA_v11{,_paper,_compact}.md
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("v10", HERE.parents[1] / "2026-10-31" / "tableA_v10.py")
v10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v10)
v10.VS_ROOTS["OmniSVG-8B, image-to-SVG"] = ["vispec/omnisvg/own", "vispec/omnisvg/x"]

if __name__ == "__main__":
    v10.v6.main()
