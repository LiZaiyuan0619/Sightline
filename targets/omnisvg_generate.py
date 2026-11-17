"""A1 runner: reuse official inputs/decoding, retain every raw stream and GPU timing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import argparse
import importlib
import json
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
from sightline.paths import REFS

ROOT = Path(__file__).resolve().parents[1]
REF = REFS / "OmniSVG"
MODELS = MODELS


def load_official():
    import torch
    sys.path.insert(0, str(REF))
    previous = Path.cwd()
    try:
        os.chdir(REF)  # Upstream inference.py opens./config.yaml at import time.
        # Torch 2.3's bf16 probe initializes CUDA even on a CPU-only host.
        probe = nullcontext() if torch.cuda.is_available() else patch("torch.cuda.is_bf16_supported", return_value=False)
        with probe:
            official = importlib.import_module("inference")
    finally:
        os.chdir(previous)
    official.CONFIG_PATH = str(REF / "config.yaml")
    return official


def generation_config(official, subtype, mode, batch_size, max_tokens):
    task = "image-to-svg" if subtype == "image" else f"text-to-svg-{subtype}"
    defaults = official.TASK_CONFIGS[task]
    cfg = dict(do_sample=mode == "sample", max_new_tokens=max_tokens,
               num_return_sequences=batch_size, use_cache=True,
               repetition_penalty=1.0 if mode == "greedy-pure" else defaults["default_repetition_penalty"],
               eos_token_id=official.EOS_TOKEN_ID, pad_token_id=official.PAD_TOKEN_ID,
               bos_token_id=official.BOS_TOKEN_ID, no_repeat_ngram_size=0)
    if mode == "sample":
        cfg.update(temperature=defaults["default_temperature"],
                   top_p=defaults["default_top_p"], top_k=defaults["default_top_k"])
    return cfg


RENDER_TIMEOUT = 20        # see render_guard


class render_guard:
    """Bound one render: cairosvg 2.7.1's path parser loops forever on some generated path data (numbers after a
    closepath with no current subpath), which hung an a batch GPU job for 47 h on. The loop is pure Python, so SIGALRM
    breaks it and the caller's `except Exception` turns it into a failed render. Nesting-aware (an outer guard, e.g.
    sightline/score_server.py's, keeps its own shorter deadline) and a no-op outside the main thread."""

    def __enter__(self):
        import signal
        self.prev = None
        try:
            self.prev = signal.alarm(0)                     # 0 = no outer deadline; >0 = seconds left on one
            if self.prev:
                signal.alarm(self.prev)                     # an outer guard is already running; leave it alone
            else:
                signal.signal(signal.SIGALRM, self._fire)
                signal.alarm(RENDER_TIMEOUT)
        except ValueError:                                  # not the main thread
            self.prev = None
        return self

    def __exit__(self, *exc):
        import signal
        if self.prev == 0:
            signal.alarm(0)
        return False

    @staticmethod
    def _fire(signum, frame):
        raise TimeoutError(f"render exceeded {RENDER_TIMEOUT} s")


def decode(official, raw_ids, subtype):
    import torch
    # Match upstream: an artificial BOS/EOS wrapper is used only for decoding.
    wrapper = torch.tensor([[official.BOS_TOKEN_ID] + raw_ids + [official.EOS_TOKEN_ID]])
    xy = official.svg_tokenizer.process_generated_tokens(wrapper)
    if len(xy) == 0:
        return None, None, False, "no_decoded_tokens"
    tensors, colors = official.svg_tokenizer.raster_svg(xy)
    if not tensors or not tensors[0]:
        return None, None, False, "no_paths"
    while len(colors) < len(tensors[0]):
        colors.append(official.BLACK_COLOR_TOKEN)
    svg = official.svg_tokenizer.apply_colors_to_svg(tensors[0], colors).to_str()
    if "width=" not in svg:
        svg = svg.replace("<svg", f'<svg width="{official.TARGET_IMAGE_SIZE}" height="{official.TARGET_IMAGE_SIZE}"', 1)
    with render_guard():
        image = official.render_svg_to_image(svg, size=official.RENDER_SIZE)
        valid, reason = official.is_valid_candidate(svg, image, subtype)
    return svg, image, valid, reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-size", choices=["8B", "4B"], default="8B",
                        help="4B (10-14): base config/tokenizer from Qwen2.5-VL-3B-Instruct-configs (no base weights; "
                             "the model is built from the config and every weight comes from OmniSVG1.1_4B)")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Existing shared Qwen2.5-VL-7B-Instruct weights; never downloads them")
    parser.add_argument("--weight-path", type=Path, default=None)
    parser.add_argument("--mode", choices=["sample", "greedy-penalty", "greedy-pure"], default="sample")
    parser.add_argument("--batch-size", choices=[1, 8], type=int, default=8,
                        help="Actual streams, without the upstream +4 buffer")
    parser.add_argument("--subtype-policy", choices=["benchmark", "official-auto"], default="benchmark")
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = MODELS / ("Qwen2.5-VL-7B-Instruct" if args.model_size == "8B" else "Qwen2.5-VL-3B-Instruct-configs")
    if args.weight_path is None:
        args.weight_path = MODELS / f"OmniSVG1.1_{args.model_size}"
    config_only = args.model_size == "4B" and not list(args.model_path.glob("*.safetensors"))
    if args.mode != "sample" and args.batch_size != 1:
        parser.error("Greedy modes require --batch-size 1")
    if args.max_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--max-tokens and --limit must be positive")
    if not args.model_path.is_dir() or (not config_only and not list(args.model_path.glob("*.safetensors"))):
        parser.error("--model-path must contain existing Qwen2.5-VL-7B-Instruct weights (metadata alone is insufficient)")
    index_path = args.model_path / "model.safetensors.index.json"
    if index_path.exists() and not config_only:
        shards = set(json.loads(index_path.read_text())["weight_map"].values())
        missing = sorted(name for name in shards if not (args.model_path / name).is_file())
        if missing:
            parser.error(f"Qwen checkpoint is incomplete: {missing}")
    if not (args.weight_path / "pytorch_model.bin").is_file():
        parser.error("OmniSVG checkpoint download has not completed")
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("GPU required: run this command on a GPU node")
    official = load_official()
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.jsonl").exists():
        raise SystemExit("Output already contains results; use a fresh directory")
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(source_commit=subprocess.check_output(["git", "-C", str(REF), "rev-parse", "HEAD"], text=True).strip(),
                    render_size=official.RENDER_SIZE, model_path=str(args.model_path.resolve()),
                    weight_path=str(args.weight_path.resolve()), torch=str(torch.__version__),
                    selection="raw candidate 0 for single-stream; first valid for batch 8",
                    cuda_device=torch.cuda.get_device_name(0))
    (args.output / "config.yaml").write_text(official.yaml.safe_dump(protocol, allow_unicode=True))
    # Set before model initialization too (resize_token_embeddings consumes RNG).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if config_only:
        # The official SketchDecoder calls Qwen2_5_VLForConditionalGeneration.from_pretrained(base) and then loads the
        # full OmniSVG state dict (strict) over it, so the base weights are never used: build from the config instead.
        import decoder as official_decoder
        cls = official_decoder.Qwen2_5_VLForConditionalGeneration
        def from_config_only(path, torch_dtype=None, attn_implementation=None, device_map=None, **kw):
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(path)
            m = cls._from_config(cfg, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
            return m.to("cuda")
        official_decoder.Qwen2_5_VLForConditionalGeneration = type(
            "Qwen2_5_VLFromConfig", (cls,), {"from_pretrained": staticmethod(from_config_only)})
    official.load_models(args.model_size, protocol["weight_path"], protocol["model_path"])
    # Tiny warm-up, excluded from timings. Each measured sample reseeds afterward.
    warm = official.prepare_inputs("text-to-svg", "A circle icon")
    warm = {k: v.to(official.get_model_input_device()) for k, v in warm.items()}
    with torch.no_grad():
        official.sketch_decoder.transformer.generate(**warm, **generation_config(official, "icon", args.mode, args.batch_size, 1))
    torch.cuda.synchronize()
    with (args.output / "results.jsonl").open("w") as log:
        for index, record in enumerate(records):
            task = "text-to-svg" if record["task"] == "text2svg" else "image-to-svg"
            subtype = record["subtype"] if task == "text-to-svg" else "image"
            if task == "text-to-svg" and args.subtype_policy == "official-auto":
                subtype = official.detect_text_subtype(record["text"])
            seed = args.seed + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            cfg = generation_config(official, subtype, args.mode, args.batch_size, args.max_tokens)
            end_to_end_start = time.perf_counter()
            if task == "text-to-svg":
                inputs = official.prepare_inputs(task, record["text"])
            else:
                image, _ = official.preprocess_image_for_svg(record["image_path"], replace_background=True)
                image_path = args.output / "processed-input.png"
                image.save(image_path)
                inputs = official.prepare_inputs(task, str(image_path))
            device = official.get_model_input_device()
            inputs = {k: v.to(device, dtype=official.DTYPE) if k == "pixel_values" else v.to(device)
                      for k, v in inputs.items()}
            prompt_ids = inputs["input_ids"][0].tolist()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                result = official.sketch_decoder.transformer.generate(**inputs, **cfg)
            torch.cuda.synchronize()
            generate_seconds = time.perf_counter() - start
            raw_batch = result[:, len(prompt_ids):].cpu().tolist()
            candidates = []
            selected = None
            for i, raw_ids in enumerate(raw_batch):
                ids = raw_ids[:raw_ids.index(official.EOS_TOKEN_ID) + 1] if official.EOS_TOKEN_ID in raw_ids else raw_ids
                c = {"index": i, "raw_token_ids": raw_ids, "token_ids": ids,
                     "length": len(ids), "eos_found": official.EOS_TOKEN_ID in ids}
                try:
                    # Decode raw padded IDs exactly as upstream does.
                    svg, image, valid, reason = decode(official, raw_ids, subtype)
                    c.update(valid=valid, reason=reason)
                    stem = args.output / f"{record['id']}-candidate-{i}"
                    if svg:
                        stem.with_suffix(".svg").write_text(svg)
                    if image is not None:
                        image.save(stem.with_suffix(".png"))
                    if valid and selected is None:
                        selected = i
                except Exception as e:
                    c.update(valid=False, reason="decode_error", error=str(e))
                candidates.append(c)
            entry = {**record, "mode": args.mode, "seed": seed, "generation_config": cfg,
                     "prompt_input_ids": prompt_ids, "candidates": candidates,
                     "selected_index": selected, "generate_seconds": generate_seconds,
                     "end_to_end_seconds": time.perf_counter() - end_to_end_start}
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.flush()
            print(record["id"], generate_seconds, "valid", sum(c["valid"] for c in candidates), flush=True)


if __name__ == "__main__":
    main()
