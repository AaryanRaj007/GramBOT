"""Build the Grambot grammar-correction training set.

Sources (all allow redistribution of a derived model):
  * agentlans/grammar-correction: a 125k-pair filtered subset of Google's C4_200M
    synthetic GEC corpus (CC BY 4.0, text from C4 / ODC-BY). Pairs whose "fix" rewrites
    the sentence are dropped, so the model learns to correct, not paraphrase.
  * Synthetic errors from noise.py on the clean side of that corpus: 15 error types
    (agreement, tense, articles, homophones, apostrophes, run-ons, punctuation, ...).
  * Reported speech that should be in quotation marks.
  * Identity pairs (correct in, correct out) so the model leaves good text alone.

Outputs in data/:
  train.jsonl   everything, for stage 1
  polish.jsonl  chat-style errors + identity, for stage 2 (short, low learning rate)
  valid.jsonl   held out, never trained on
"""

import argparse
import io
import json
import random
from pathlib import Path

from noise import corrupt, reported, run_on, speech

OUT = Path(__file__).parent / "data"

# How much synthetic data per clean sentence and how big each part is.
PROFILES = {
    "fast":   {"variants": 1, "run_ons": 15_000, "speech": 4_000, "reported": 3_000, "identity": 15_000, "polish": 30_000},
    "strong": {"variants": 1, "run_ons": 30_000, "speech": 8_000, "reported": 6_000, "identity": 30_000, "polish": 50_000},
    "max":    {"variants": 2, "run_ons": 50_000, "speech": 12_000, "reported": 10_000, "identity": 50_000, "polish": 70_000},
    # max plus ~1.8 M extra real pairs from the full C4_200M (3 shards downloaded, ~1.1 GB)
    "mega":   {"variants": 2, "run_ons": 60_000, "speech": 15_000, "reported": 20_000, "identity": 80_000, "polish": 90_000,
               "big_shards": 3, "big": 1_800_000},
}


def read_zst_jsonl(path):
    import zstandard
    with open(path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in text:
            if line.strip():
                row = json.loads(line)
                yield row["input"].strip(), row["output"].strip()


def close(a, b, min_dice=0.75):
    """Keep corrections, drop C4 pairs where the "fix" rewrites the sentence (it teaches meaning drift)."""
    wa, wb = a.lower().split(), b.lower().split()
    common = len(set(wa) & set(wb))
    return common * 2 / (len(set(wa)) + len(set(wb)) or 1) >= min_dice and abs(len(wa) - len(wb)) <= 4


def tidy(s):
    return s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').strip()


def good_target(b, vocab):
    """Raw C4_200M targets are sometimes worse than the input (typos, web junk). Keep only clean sentences
    whose every word is common in the corpus, so the model never learns to "fix" text into a typo."""
    words = b.split()
    if not (4 <= len(words) <= 40 and b.isascii() and b[:1].isupper() and b[-1:] in ".?!"):
        return False
    if " n't" in b or " 's " in b or "  " in b or "http" in b:
        return False
    return all(vocab.get(w, 0) >= 30 for w in (t.strip(".,!?;:\"'()").lower() for t in words) if w.isalpha())


def big_c4(shards, cap, skip):
    """Real pairs from the full C4_200M (liweili/c4_200m parquet mirror, 1.83 M pairs per shard)."""
    import pyarrow.parquet as pq
    from collections import Counter
    from huggingface_hub import hf_hub_download
    rows = []
    for i in range(shards):
        path = hf_hub_download("liweili/c4_200m", f"default/partial-train/{i:04d}.parquet", repo_type="dataset",
                               revision="refs/convert/parquet")
        t = pq.read_table(path, columns=["input", "output"])
        rows += zip(map(tidy, t.column("input").to_pylist()), map(tidy, t.column("output").to_pylist()))
        print(f"shard {i}: {len(rows):,} raw pairs", flush=True)
    vocab = Counter(w.strip(".,!?;:\"'()").lower() for _, b in rows for w in b.split())
    kept = [(a, b) for a, b in rows if b not in skip and close(a, b, 0.8) and good_target(b, vocab)]
    print(f"big C4_200M: kept {len(kept):,} of {len(rows):,} ({len(kept) / len(rows):.0%})", flush=True)
    return kept[:cap]


def usable(s):
    return 3 <= len(s.split()) <= 40 and s.isascii() and s[-1:] in ".?!" and '"' not in s and s[:1].isupper()


def synthetic(clean, rng, p):
    rows = []
    for s in clean:
        for _ in range(p["variants"]):
            noisy = corrupt(s, rng)
            if noisy:
                rows.append((noisy, s))
    short = [s for s in clean if len(s.split()) <= 14]
    for _ in range(p["run_ons"]):
        rows.append(run_on(rng.sample(short, rng.choice([2, 2, 3])), rng))
    rows += [speech(rng) for _ in range(p["speech"])]
    rows += [reported(rng) for _ in range(p["reported"])]
    rows += [(s, s) for s in rng.sample(clean, min(p["identity"], len(clean)))]
    return rows


def write(name, rows):
    with open(OUT / f"{name}.jsonl", "w") as fh:
        for a, b in rows:
            fh.write(json.dumps({"input": a, "output": b}) + "\n")
    print(f"{name:7} {len(rows):>9,} pairs -> {OUT / (name + '.jsonl')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, default="strong")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    p = PROFILES[args.profile]
    rng = random.Random(args.seed)
    OUT.mkdir(exist_ok=True)

    from huggingface_hub import hf_hub_download
    repo = "agentlans/grammar-correction"
    train_src = list(read_zst_jsonl(hf_hub_download(repo, "train.jsonl.zst", repo_type="dataset")))
    valid_src = list(read_zst_jsonl(hf_hub_download(repo, "validation.jsonl.zst", repo_type="dataset")))
    print(f"downloaded {len(train_src):,} + {len(valid_src):,} C4_200M pairs")

    # Validation: real C4 pairs plus a little of every synthetic kind, all from held-out sentences.
    valid_c4 = [(a, b) for a, b in valid_src[:3_000] if close(a, b)]
    valid_clean = [b for _, b in valid_src[3_000:6_000] if usable(b)]
    small = {"variants": 1, "run_ons": 300, "speech": 150, "reported": 150, "identity": 300}
    valid = valid_c4[:1_500] + synthetic(valid_clean, rng, small)
    rng.shuffle(valid)

    # Clean sentences for synthetic training data come from train and the unused part of validation.
    c4 = [(a, b) for a, b in train_src if close(a, b)]
    if p.get("big"):
        c4 += big_c4(p["big_shards"], p["big"], skip={b for _, b in valid_src})
    # sorted: a set's order changes between runs, and resumed training needs identical data
    clean = sorted({b for _, b in train_src if usable(b)} | {b for _, b in valid_src[6_000:] if usable(b)})
    rng.shuffle(clean)
    synth = synthetic(clean, rng, p)
    train = c4 + synth
    rng.shuffle(train)

    # Stage 2: what people type in chat boxes, plus identity, so the polish pass does not forget "leave it alone".
    chatty = [r for r in synth if len(r[0].split()) <= 30]
    polish = rng.sample(chatty, min(p["polish"], len(chatty))) + rng.sample(c4, min(p["polish"] // 4, len(c4)))
    rng.shuffle(polish)

    write("train", train)
    write("polish", polish)
    write("valid", valid)
    print("example:", json.dumps(train[0]))


if __name__ == "__main__":
    main()
