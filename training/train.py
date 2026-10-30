"""Fine-tune FLAN-T5 (Apache-2.0) into Grambot's on-device grammar model.

Two stages:
  1. --stage main     all data (C4_200M + synthetic), the long run
  2. --stage polish   short pass on chat-style errors at a low learning rate, starting from stage 1

Runs on Apple Silicon (MPS), CUDA or CPU. Profiles are tuned for an 8 GB M-series Mac.
Interrupted? Run the same command again with --resume.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback)

HERE = Path(__file__).parent
PREFIX = "grammar: "  # the extension adds the same prefix

PROFILES = {
    # base model, micro-batch, epochs, learning rate. Effective batch is always 32.
    "fast":   {"base": "google/flan-t5-small", "batch": 8, "epochs": 1, "lr": 1e-3},
    "strong": {"base": "google/flan-t5-base",  "batch": 4, "epochs": 1, "lr": 5e-4},
    "max":    {"base": "google/flan-t5-base",  "batch": 4, "epochs": 2, "lr": 5e-4},
    # continues from an already trained Grambot model in init/ (Kaggle: the previous notebook's output)
    "mega":   {"base": "google/flan-t5-base",  "batch": 4, "epochs": 1, "lr": 3e-4, "init": "init/grambot-gec"},
}
EFFECTIVE_BATCH = 32
OUT_OF_TIME = 3  # exit code: stopped by --hours, run again with --resume


class Progress(TrainerCallback):
    """One plain line per log step with an ETA (progress bars do not show in cloud logs)."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0, self.s0 = time.time(), state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        done = state.global_step - self.s0
        if done <= 0:
            return
        per = (time.time() - self.t0) / done
        eta = (state.max_steps - state.global_step) * per / 3600
        print(f"step {state.global_step}/{state.max_steps} | {per:.2f} s/step | ETA {eta:.1f} h | "
              + " ".join(f"{k} {v:.4g}" for k, v in (logs or {}).items() if isinstance(v, float)), flush=True)


class TimeLimit(TrainerCallback):
    """Save a checkpoint and stop cleanly before a cloud session (e.g. Kaggle's 12 h) ends."""

    def __init__(self, hours):
        self.deadline = time.time() + hours * 3600
        self.hit = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() > self.deadline:
            self.hit = control.should_save = control.should_training_stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, default="strong")
    ap.add_argument("--stage", choices=["main", "polish"], default="main")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint of this stage")
    ap.add_argument("--max-train", type=int, default=0, help="limit pairs (0 = all); 300 for a smoke test")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=0, help="override the micro-batch (lower it if memory runs out)")
    ap.add_argument("--hours", type=float, default=0, help="stop and checkpoint after this long (0 = no limit)")
    args = ap.parse_args()
    p = dict(PROFILES[args.profile])
    if args.batch:
        p["batch"] = args.batch

    main_out = HERE / "out" / "main"
    final_out = HERE / "out" / "grambot-gec"
    if args.stage == "main":
        start, data_file, lr, epochs, out = p["base"], "train.jsonl", p["lr"], p["epochs"], main_out
        if p.get("init"):
            start = str(HERE / p["init"])
            if not (Path(start) / "config.json").exists():
                raise SystemExit(f"Profile {args.profile} starts from a trained model, but {start} is missing.")
    else:
        if not main_out.exists():
            raise SystemExit("Run stage 1 first: python train.py --profile %s" % args.profile)
        start, data_file, lr, epochs, out = str(main_out), "polish.jsonl", p["lr"] / 3, 1, final_out

    tok = AutoTokenizer.from_pretrained(start)
    model = AutoModelForSeq2SeqLM.from_pretrained(start)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if p["base"].endswith("base") and device != "cuda":
        model.gradient_checkpointing_enable()  # trades ~25% speed for the memory an 8 GB Mac does not have
        model.config.use_cache = False

    data = load_dataset("json", data_files={"train": str(HERE / "data" / data_file), "valid": str(HERE / "data/valid.jsonl")})
    if args.max_train:
        data["train"] = data["train"].select(range(min(args.max_train, len(data["train"]))))
    data["valid"] = data["valid"].shuffle(seed=1).select(range(min(1_000, len(data["valid"]))))

    def encode(batch):
        enc = tok([PREFIX + s for s in batch["input"]], max_length=96, truncation=True)
        enc["labels"] = tok(text_target=batch["output"], max_length=96, truncation=True)["input_ids"]
        return enc

    data = data.map(encode, batched=True, remove_columns=["input", "output"], num_proc=4)
    steps = len(data["train"]) * epochs // EFFECTIVE_BATCH
    print(f"stage {args.stage} | {start} | device {device} | {len(data['train']):,} pairs | ~{steps:,} steps")

    smoke = args.max_steps > 0
    every = max(1, args.max_steps) if smoke else 1_000
    targs = Seq2SeqTrainingArguments(
        output_dir=str(HERE / "out" / f"checkpoints-{args.stage}"),
        per_device_train_batch_size=p["batch"],
        per_device_eval_batch_size=p["batch"] * 2,
        gradient_accumulation_steps=EFFECTIVE_BATCH // p["batch"],
        group_by_length=True,  # similar lengths per batch: less padding, less memory
        optim="adafactor",  # T5's own optimizer: far less memory than AdamW
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,
        label_smoothing_factor=0.1,
        eval_strategy="steps",
        eval_steps=every,
        save_steps=every,
        save_total_limit=3,
        load_best_model_at_end=True,  # keep the checkpoint with the lowest validation loss, not just the last one
        metric_for_best_model="eval_loss",
        logging_steps=1 if smoke else 50,
        bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=device == "cuda",
        report_to=[],
        disable_tqdm=not sys.stdout.isatty(),
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=data["train"], eval_dataset=data["valid"],
        data_collator=DataCollatorForSeq2Seq(tok, model=model, pad_to_multiple_of=32), processing_class=tok,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4), Progress()],
    )
    limit = TimeLimit(args.hours) if args.hours else None
    if limit:
        trainer.add_callback(limit)
    trainer.train(resume_from_checkpoint=args.resume or None)
    if limit and limit.hit:
        print(f"time limit reached; checkpoint saved. Continue with: python train.py --profile {args.profile} "
              f"--stage {args.stage} --resume")
        raise SystemExit(OUT_OF_TIME)
    model.config.use_cache = True
    trainer.save_model(out)
    tok.save_pretrained(out)
    json.dump({"base": p["base"], "profile": args.profile, "stage": args.stage, "prefix": PREFIX},
              open(out / "grambot.json", "w"))
    print(f"saved -> {out}")
    if args.stage == "main":
        print(f"next: python train.py --profile {args.profile} --stage polish")


if __name__ == "__main__":
    main()
