"""Export the trained model to ONNX + int8 in the layout Transformers.js expects,
into models/grambot-gec at the repository root."""

import argparse
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.exporters.onnx import main_export

HERE = Path(__file__).parent
DEST = HERE.parent / "models" / "grambot-gec"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "out" / "grambot-gec"))
    args = ap.parse_args()
    if not (Path(args.model) / "config.json").exists():
        raise SystemExit(f"No trained model at {args.model}. Train first (README steps 4 and 5), then export.")

    tmp = HERE / "out" / "onnx"
    shutil.rmtree(tmp, ignore_errors=True)
    main_export(args.model, output=tmp, task="text2text-generation-with-past")

    shutil.rmtree(DEST, ignore_errors=True)
    (DEST / "onnx").mkdir(parents=True)
    for name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        if (tmp / name).exists():
            shutil.copy(tmp / name, DEST / name)
    for part in ["encoder_model", "decoder_model_merged"]:
        quantize_dynamic(tmp / f"{part}.onnx", DEST / "onnx" / f"{part}_quantized.onnx", weight_type=QuantType.QUInt8,
                        extra_options={"EnableSubgraph": True})  # merged decoder keeps weights in If-branches
    shutil.copy(Path(args.model) / "grambot.json", DEST / "grambot.json")
    size = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    print(f"exported -> {DEST} ({size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
