"""Score a model the way grammar checkers are scored: per edit, not per sentence.

  precision  of the changes the model made, how many were right (false alarms hurt this)
  recall     of the changes needed, how many the model made
  F0.5       the standard GEC score; weighs precision twice as much as recall,
             because a wrong suggestion is worse than a missed one

Reports on the hand-written cases (eval_cases.json) and on held-out validation data.
Exit code 1 if the gate fails. --compare scores a second model side by side.
"""

import argparse
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).parent
PREFIX = "grammar: "
GATE = {"cases_f05": 0.60, "valid_f05": 0.45, "kept": 0.95}


def norm(s):
    return " ".join(s.replace("“", '"').replace("”", '"').replace("’", "'").split())


def tokens(s):
    import re
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", norm(s))


def edits(src, tgt):
    a, b = tokens(src), tokens(tgt)
    return {(i1, i2, " ".join(b[j1:j2])) for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            if op != "equal"}


class Score:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, src, hyp, refs):
        h = edits(src, hyp)
        best = None
        for r in refs:  # credit the closest acceptable answer
            g = edits(src, r)
            tp = len(h & g)
            cand = (tp, len(h) - tp, len(g) - tp)
            if best is None or cand[0] - cand[1] - cand[2] > best[0] - best[1] - best[2]:
                best = cand
        self.tp += best[0]
        self.fp += best[1]
        self.fn += best[2]

    def prf(self):
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0
        f = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
        return p, r, f


def load(path):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(path).eval().to(dev)

    def fix(texts):
        ids = tok([PREFIX + t for t in texts], return_tensors="pt", padding=True, max_length=128, truncation=True).to(dev)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=128, num_beams=1)
        return tok.batch_decode(out, skip_special_tokens=True)
    return fix


def run(path, n_valid, verbose):
    fix = load(path)
    cases = json.load(open(HERE / "eval_cases.json"))
    cs, kept, kept_n = Score(), 0, 0
    t = time.perf_counter()
    outs = [fix([c[0]])[0] for c in cases]
    per_case_ms = (time.perf_counter() - t) / len(cases) * 1000
    for (inp, *refs), got in zip(cases, outs):
        correct_input = refs == [inp]
        hit = norm(got) in {norm(r) for r in refs}
        if correct_input:
            kept_n += 1
            kept += hit
        else:
            cs.add(inp, got, refs)
        if verbose:
            print(("OK " if hit else "XX ") + json.dumps(inp[:70]) + "\n     => " + json.dumps(got))

    vs = Score()
    if n_valid:
        rows = [json.loads(l) for l in open(HERE / "data/valid.jsonl")][:n_valid]
        for i in range(0, len(rows), 16):
            chunk = rows[i:i + 16]
            for r, got in zip(chunk, fix([r["input"] for r in chunk])):
                vs.add(r["input"], got, [r["output"]])
    return {"cases": cs.prf(), "valid": vs.prf() if n_valid else None, "kept": kept / kept_n, "kept_n": (kept, kept_n),
            "ms": per_case_ms}


def show(name, r):
    p, rc, f = r["cases"]
    line = f"{name:28} cases P {p:.0%} R {rc:.0%} F0.5 {f:.2f}"
    if r["valid"]:
        vp, vr, vf = r["valid"]
        line += f" | valid P {vp:.0%} R {vr:.0%} F0.5 {vf:.2f}"
    line += f" | correct kept {r['kept_n'][0]}/{r['kept_n'][1]} | {r['ms']:.0f} ms/sentence (PyTorch)"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "out" / "grambot-gec"))
    ap.add_argument("--compare", help="a second model to score side by side, e.g. out/main")
    ap.add_argument("--valid", type=int, default=800, help="validation pairs to score (0 = skip)")
    args = ap.parse_args()
    for path in filter(None, [args.model, args.compare]):
        if not (Path(path) / "config.json").exists():
            raise SystemExit(f"No trained model at {path}. Train first (README steps 4 and 5), then evaluate.")

    r = run(args.model, args.valid, verbose=True)
    print()
    show(Path(args.model).name, r)
    if args.compare:
        show(Path(args.compare).name, run(args.compare, args.valid, verbose=False))

    ok = r["cases"][2] >= GATE["cases_f05"] and r["kept"] >= GATE["kept"] and (not r["valid"] or r["valid"][2] >= GATE["valid_f05"])
    print("GATE", "PASS" if ok else "FAIL", GATE)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
