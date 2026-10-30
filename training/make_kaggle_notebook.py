"""Pack the training scripts into one Kaggle notebook: kaggle/grambot_train.ipynb.

Re-run after changing any training script:  python3 make_kaggle_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
FILES = ["requirements.txt", "noise.py", "prepare_data.py", "train.py", "evaluate.py", "export.py", "eval_cases.json"]


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    md("# Grambot grammar model training (mega)\n"
       "**Before the first run:** Add Input → Your Work → the earlier Grambot notebook (the version with `grambot-model.zip`). "
       "Mega continues training that model.\n\n"
       "Settings (right panel): **Accelerator: GPU T4 x2**, **Internet: on**.\n"
       "Then **Save Version → Save & Run All (Commit)**. It runs on Kaggle's servers, so you can close the laptop.\n\n"
       "**Run 1** trains until shortly before Kaggle's 12 h limit, then saves its progress as output.\n"
       "**Run 2+**: click **Add Input → Your Work → this notebook** (the latest version), then Save & Run All again.\n"
       "It picks up where the last run stopped. When everything is done, the version's **Output** tab has `grambot-model.zip`."),
    code('PROFILE = "mega"     # max + 1.8 M extra real pairs, starts from the trained model; ~14 h, 2 runs\n'
         'BATCH = 16           # per GPU; T4 has 16 GB\n'
         'SESSION_HOURS = 11.5 # Kaggle stops a run at 12 h\n'
         'import time; T0 = time.time()'),
    code("!mkdir -p /kaggle/working/training"),
]
for name in FILES:
    cells.append(code(f"%%writefile /kaggle/working/training/{name}\n" + (HERE / name).read_text()))
cells += [
    code('!pip install -q "transformers>=4.46,<5" "datasets>=3.0" sentencepiece "accelerate>=1.0" zstandard "optimum[onnxruntime]>=1.23" onnx\n'
         '# Kaggle preinstalls diffusers, which breaks with the huggingface-hub that transformers 4.x needs,\n'
         '# and optimum\'s ONNX export imports it if present. We never use it.\n'
         '!pip uninstall -y -q diffusers'),
    code("import glob, os, shutil, subprocess, sys\n"
         "os.chdir('/kaggle/working/training')\n"
         "prev = glob.glob('/kaggle/input/**/grambot-state/out', recursive=True)\n"
         "if prev:\n"
         "    shutil.copytree(prev[0], 'out', dirs_exist_ok=True)\n"
         "    print('continuing from', prev[0], os.listdir('out'))\n"
         "else:\n"
         "    print('fresh start')\n"
         "if PROFILE == 'mega' and not os.path.exists('init/grambot-gec/config.json'):\n"
         "    found = glob.glob('/kaggle/input/**/training/out/grambot-gec/config.json', recursive=True)\n"
         "    zips = glob.glob('/kaggle/input/**/grambot-model.zip', recursive=True)\n"
         "    if found:\n"
         "        shutil.copytree(os.path.dirname(found[0]), 'init/grambot-gec')\n"
         "    elif zips:\n"
         "        !unzip -q -o {zips[0]} 'training/out/grambot-gec/*' -d /tmp/prev && mkdir -p init && mv /tmp/prev/training/out/grambot-gec init/\n"
         "    else:\n"
         "        raise SystemExit('Add Input -> Your Work -> the earlier Grambot notebook first (it has grambot-model.zip).')\n"
         "    print('starting from', os.listdir('init/grambot-gec'))\n"
         "!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
         "!python noise.py | tail -1\n"
         "!python prepare_data.py --profile $PROFILE  # same seed every run, so a resumed run sees identical data"),
    code("def sh(cmd, ok=(0,)):\n"
         "    print('$', ' '.join(cmd), flush=True)\n"
         "    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
         "    for line in proc.stdout:\n"
         "        print(line, end='', flush=True)\n"
         "    code = proc.wait()\n"
         "    if code not in ok:\n"
         "        raise RuntimeError(f'failed with exit code {code}: send this log to Claude')\n"
         "    return code\n\n"
         "# Smoke test (fresh start only, ~5 min): runs every step on a tiny slice so a bug fails now, not 10 h in.\n"
         "if not prev:\n"
         "    py = sys.executable\n"
         "    tiny = [py, 'train.py', '--profile', PROFILE, '--batch', '4', '--max-train', '64', '--max-steps', '2']\n"
         "    sh(tiny + ['--hours', '0.0001'], ok=(3,))  # time limit hit after step 1: proves the stop-and-save path\n"
         "    sh(tiny + ['--resume'])                     # and that the next run continues from it\n"
         "    sh([py, 'train.py', '--profile', PROFILE, '--batch', '4', '--max-train', '64', '--max-steps', '2', '--stage', 'polish'])\n"
         "    sh([py, 'evaluate.py', '--valid', '16'], ok=(0, 1))  # 1 = gate fail, expected after 2 steps\n"
         "    sh([py, 'export.py'])\n"
         "    shutil.rmtree('out'); shutil.rmtree('/kaggle/working/public')\n"
         "    print('SMOKE TEST PASSED', flush=True)"),
    code("def hours_left():\n"
         "    return SESSION_HOURS - (time.time() - T0) / 3600 - 0.75  # keep time for scoring, export and saving output\n\n"
         "def stage(name):\n"
         "    done = 'out/main/grambot.json' if name == 'main' else 'out/grambot-gec/grambot.json'\n"
         "    if os.path.exists(done):\n"
         "        print(name, 'already done'); return True\n"
         "    if hours_left() < 0.3:\n"
         "        return False\n"
         "    cmd = [sys.executable, 'train.py', '--profile', PROFILE, '--batch', str(BATCH), '--stage', name,\n"
         "           '--hours', f'{hours_left():.2f}'] + (['--resume'] if glob.glob(f'out/checkpoints-{name}/checkpoint-*') else [])\n"
         "    return sh(cmd, ok=(0, 3)) == 0  # 3 = stopped by the time limit, continue next run\n\n"
         "finished = stage('main') and stage('polish')\n"
         "print('ALL TRAINING DONE' if finished else 'Not finished yet: progress will be saved for the next run.')"),
    code("if finished:\n"
         "    ref = 'init/grambot-gec' if os.path.exists('init/grambot-gec') else 'out/main'  # mega: vs the model it started from\n"
         "    os.system(f'python evaluate.py --compare {ref} 2>&1 | tee eval.txt')\n"
         "    os.system('python export.py')\n"
         "    shutil.rmtree('data', ignore_errors=True)\n"
         "    for d in glob.glob('out/checkpoints-*') + ['out/onnx', 'out/main', 'init']:\n"
         "        shutil.rmtree(d, ignore_errors=True)\n"
         "    os.chdir('/kaggle/working')\n"
         "    os.system('zip -qr grambot-model.zip models/grambot-gec training/out/grambot-gec training/eval.txt')\n"
         "    shutil.rmtree('training', ignore_errors=True); shutil.rmtree('public', ignore_errors=True)\n"
         "    print('Download grambot-model.zip from the Output tab.')\n"
         "else:\n"
         "    shutil.rmtree('data', ignore_errors=True); shutil.rmtree('init', ignore_errors=True)  # both rebuilt next run\n"
         "    os.makedirs('/kaggle/working/grambot-state', exist_ok=True)\n"
         "    shutil.move('out', '/kaggle/working/grambot-state/out')\n"
         "    print('Saved progress. Next: Add Input -> Your Work -> this notebook (latest version), then Save & Run All again.')"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "kaggle" / "grambot_train.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
