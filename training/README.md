# Training guide

Everything runs on your Mac. Nothing you type in the browser is used for training.

## What makes v2 stronger
| | v1 | v2 strong | v2 max |
|---|---|---|---|
| Model | FLAN-T5-small (77 M) | **FLAN-T5-base (248 M)**, 3× bigger | FLAN-T5-base |
| Error types practised | 5 | **15** (agreement, tense, articles, homophones, apostrophes, plurals, missing / doubled words, word order, typos, capitals, punctuation, run-ons, "could of", "more taller", quotes) | 15, 2 variants per sentence |
| Training data | ~124 k pairs | ~220 k | ~360 k, 2 passes |
| Training | 1 stage | **2 stages**: main, then a chat-style polish | 2 stages |
| Keeps best checkpoint, stops early if it stops improving | no | yes | yes |
| Scoring | exact match | **edit-level precision / recall / F0.5** (how grammar checkers are judged), 50 hand cases | same |
| Model size in the extension | ~97 MB | ~300 MB | ~300 MB |
| Time on M2, 8 GB (estimate) | ~2 h | **~10–14 h** (overnight) | ~1.5–2 days |

The times are estimates. The progress bar shows the real ETA after a few minutes.

## Free cloud option: Kaggle (laptop can be off)
Kaggle gives every account free GPU time: about 30 h a week, up to 12 h per run. A run keeps going after you close the laptop.
1. Make an account at kaggle.com and verify your phone number (needed for GPU and internet).
2. **Create → New Notebook**, then **File → Import Notebook** and upload `training/kaggle/grambot_train.ipynb`.
3. In the right panel set **Accelerator: GPU T4 x2** and **Internet: on**.
4. Click **Save Version → Save & Run All (Commit) → Save**. You can close the laptop now.
   The notebook defaults to the strongest profile (`max`, ~10–17 h in total, estimated). Kaggle stops a run at 12 h, so:
   - Run 1 trains until about 11 h, then saves its progress (`grambot-state`) as output.
   - For run 2, open the notebook editor and click **Add Input → Your Work →** this notebook (latest version).
     Then run **Save & Run All** again. It continues exactly where run 1 stopped.
   - Repeat until the log ends with `ALL TRAINING DONE`. The `strong` profile usually needs one run.
5. When the version shows **Complete**, open it, go to **Output**, and download `grambot-model.zip`.
6. On your Mac:
   ```bash
   cd ~/Downloads/GramBot
   unzip ~/Downloads/grambot-model.zip
   cat training/eval.txt | tail -3
   ```
   This puts the browser model in `models/grambot-gec` and the trained model in `training/out/grambot-gec`.
   Steps 4–7 below are then already done.

### Mega: continue training with 1.8 M more real pairs (~14 h, 2 runs)
1. New notebook → **File → Import Notebook** → `training/kaggle/grambot_train.ipynb` (profile `mega`).
2. **Add Input → Your Work →** the earlier Grambot notebook (the version whose output has `grambot-model.zip`).
3. GPU T4 x2, Internet on, **Save Version → Save & Run All (Commit)**.
4. Run 1 stops at about 11 h and saves progress. Then **Add Input → Your Work → this notebook** (latest version) and Save & Run All again.
5. `eval.txt` compares the new model with the one it started from.

Changed a training script? Run `python3 make_kaggle_notebook.py` to rebuild the notebook.
Your training data is a public dataset, and nothing you type in the browser goes to Kaggle.

## Before you start (training on this Mac)
- Plug in the charger and close Brave and other heavy apps. Training uses nearly all 8 GB of memory.
- Commands go in Terminal. If you close Terminal, run `cd ~/Downloads/GramBot/training && source .venv/bin/activate` before the next step.

## 1. Setup (once, ~3 min)
```bash
cd ~/Downloads/GramBot/training
rm -rf .venv data out
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python noise.py
```
`noise.py` shows one example of each of the 15 error types and ends with `self-check OK`.

## 2. Build the data (~2 min)
```bash
python prepare_data.py --profile strong
```
This prints three files: `train`, `polish` and `valid`. Open `data/train.jsonl` to look at real examples.

## 3. Quick test (~3 min), so a problem doesn't show up 6 hours in
```bash
python train.py --profile strong --max-train 200 --max-steps 3
rm -rf out
```
It should end with `saved -> …/out/main`. If it says `MPS backend out of memory`, add `--batch 2` to every train command below.

## 4. Stage 1: main training (~9–12 h)
```bash
caffeinate -is python train.py --profile strong
```
- Every 1,000 steps it prints `eval_loss`. Lower is better. It usually falls fast at first, then slowly.
- It stops on its own if `eval_loss` stops improving 4 times in a row.
- If it gets interrupted, run the same command with `--resume` added. It continues from the last saved checkpoint, which is never more than 1,000 steps back.

## 5. Stage 2: polish (~1–2 h)
```bash
caffeinate -is python train.py --profile strong --stage polish
```

## 6. Score it (~10 min)
```bash
python evaluate.py --compare out/main
```
- Each hand case prints `OK` or `XX` with what the model wrote.
- The summary line shows **P** (precision: how many of its changes were right), **R** (recall: how many needed fixes it found) and **F0.5**.
- `--compare` also scores the stage 1 model, so you can see what the polish stage added, and prints the F0.5 difference.
- The last line must say `GATE PASS`: cases F0.5 ≥ 0.60, validation F0.5 ≥ 0.45, and ≥ 95 % of correct sentences left unchanged.

## 7. Export into the extension (~2 min)
```bash
python export.py
```
Expect `exported -> …/models/grambot-gec (~265 MB)`.

Ship only a model whose evaluation ends with `GATE PASS`.

## Want it even better? (`max` profile)
```bash
python prepare_data.py --profile max
caffeinate -is python train.py --profile max
caffeinate -is python train.py --profile max --stage polish
python evaluate.py --compare out/main
```
It's the same pipeline with twice the practice sentences and two passes over them.

## Tests (no model needed, ~1 s)
```bash
python -m unittest discover -s tests -v
```
Covers edit extraction, P/R/F0.5 scoring and the error generators. CI runs them on every push.

## If the gate fails
- **Low P, or correct sentences changed:** the model over-corrects. Raise the `identity` count in `prepare_data.py` so it sees more "leave it alone" examples.
- **Low R:** the model misses errors. Retrain with `max`, or add practice for the failing `XX` cases in `noise.py`.
- **One case type always fails** (for example quotes): add a targeted generator for it in `noise.py`.

## Files
`noise.py` error generator · `prepare_data.py` data · `train.py` training · `evaluate.py` scoring · `export.py` browser format · `eval_cases.json` the 50 hand cases (36 with mistakes, 14 correct)
