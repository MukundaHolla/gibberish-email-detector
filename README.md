# Standalone gibberish email detector

Give the trainer two local text files. One contains full email addresses your reviewers flagged for gibberish spelling. The other contains ordinary full email addresses. It learns one scoring model from the labels. The saved model can score a new bare email address or a whole file. The included demo was trained on invented examples. No real customer lists were supplied.

If machine learning is new to you, start with [How it works](HOW_IT_WORKS.md). It follows one email all the way from characters to a numeric score and includes training and prediction flow diagrams.

## Try it on your Mac in two commands

Extract the ZIP. Open Terminal, type `cd `, drag the extracted `gibberish-email-detector` folder into Terminal, and press Return. Run `python3 --version`. Use Python 3.11 or later. Python 3.12 is the tested version. If you have Homebrew and need Python, `brew install python@3.12` installs it. In that case use `python3.12` wherever these commands say `python3`.

```bash
python3 detector.py explain 'ravi@gmail.com'
python3 detector.py explain 'qzxv8801@gmail.com'
```

The demo scores are about `0.3852` and `0.7488`. Its saved cutoff is about `0.4917`. The first address falls below the cutoff. The second crosses it. For a short JSON result, use `score` instead of `explain`. These commands use Python's standard library and the included weights. Nothing is fitted, downloaded, or sent when you score.

```bash
python3 detector.py score 'ravi@qzxvpk.com'
python3 detector.py explain 'qzxv8801@gmail.com' --all
python3 detector.py score 'qzxv8801@gmail.com' --threshold 0.80
python3 detector.py self-test
```

The threshold experiment leaves the score unchanged. The self-test should print `PASS — 10 reference cases for demo-invented-v1.`

## Train on your own two files

1. Make a folder named `my-data` in the extracted project. Put `flagged.txt` and `ordinary.txt` in it. Use UTF-8 plain text, exactly one complete address per line, with no headings or bullets. Blank lines are ignored. See `examples/flagged.txt` and `examples/ordinary.txt` for the format.
2. Check what the labels mean. For a lexical detector, place addresses in `flagged.txt` when their **spelling** matches your agreed definition of gibberish. An email flagged solely for a chargeback, a disposable provider, a bulk order, or another business reason needs separate review. Keep unresolved examples out of both lists. An unfamiliar name alone is weak evidence.
3. Create a Python environment and install the training software once.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

4. Train a new model, then tell the scorer which model to use.

```bash
python detector.py train \
  --flagged my-data/flagged.txt \
  --ordinary my-data/ordinary.txt \
  --out runs/my-first-model

python detector.py score 'ravi@gmail.com' --model runs/my-first-model
python detector.py explain 'ravi@gmail.com' --model runs/my-first-model
python detector.py self-test --model runs/my-first-model
```

The output directory contains `model.json`, `report.json`, and `golden.json`. A score without `--model` deliberately uses the bundled demo. Training again uses a fresh output directory, for example `runs/second-model`. Updating a text file affects scores only after you train and select the new model.

To practice with the supplied **labeled** data before preparing yours, run the same command with `--flagged examples/flagged.txt --ordinary examples/ordinary.txt --out runs/practice`. In a new Terminal session, navigate back to the project and use `source .venv/bin/activate` again to access the installed training dependency.

## Score a bulk list or test real outcomes

Score new, unlabeled inputs with `batch`. Replace `examples/incoming.txt` with your file.

```bash
python detector.py batch examples/incoming.txt \
  --model runs/my-first-model --out results/predictions.jsonl
```

The output contains one JSON object per nonblank line with the input line number, original email, score, decision, model name, threshold, coverage, and reason. It contains address data and stays on your machine. Use a fresh output filename.

For quality measurement, use emails whose labels you know and which you set aside before development, preferably collected later and from an independent source.

```bash
python detector.py evaluate \
  --flagged holdout/flagged.txt --ordinary holdout/ordinary.txt \
  --model runs/my-first-model
```

Provide the two `holdout/` files yourself. The evaluator counts unique model inputs, rejects contradictory labels, and counts out-of-scope positives as missed flags. Score-only or unsure decisions do not raise an alert. Record prevalence and volumes from naturally sampled data to translate these metrics into expected daily false alerts.

For Python integration from the project directory, load once and call the same function used by the CLI.

```python
from engine import load_model, predict

model = load_model("runs/my-first-model")
answer = predict("ravi@gmail.com", model)
print(answer["score"], answer["decision"])
```

## Project map

| File or directory | What it does |
|---|---|
| `detector.py` | The entry point. Implements `score`, `explain`, `batch`, `train`, `evaluate`, and `self-test`. |
| `engine.py` | Shared spelling preparation, feature counting, coverage checks, scoring arithmetic, and label decision. |
| `training.py` | Reads both labeled lists, separates training/validation/test groups, calls scikit-learn, chooses a cutoff, checks the export, and writes results. |
| `requirements.txt` | Pins the direct training dependency to scikit-learn 1.8.0. |
| `assets/public_suffix_list.dat` | An offline domain-suffix reference used to identify the registrable label, including multi-part suffixes such as `co.in`. |
| `assets/PUBLIC_SUFFIX_LICENSE.md` | Source revision and license notice for that third-party dataset. |
| `examples/ordinary.txt`, `examples/flagged.txt` | All 480 invented, labeled full addresses used to build the bundled demonstration. |
| `examples/incoming.txt` | Four sample inputs for scoring a batch. |
| `demo/model.json` | The learned demo weights and associated scoring settings. |
| `demo/report.json` | Counts and evaluation results from training the demo. This is the file to read when assessing quality. |
| `demo/golden.json` | Ten fixed invented inputs and their expected outputs for repeatability. |
| `test_detector.py` | Regression tests for preprocessing, coverage, cutoff behavior, input conflicts, grouped splits, and batch output. Run `python -m unittest -v test_detector` after installing training dependencies. |
| `README.md` and `HOW_IT_WORKS.md` | Commands and an illustrated explanation. |
| `.venv/` | Created on your computer. Holds the project-specific Python environment and installed dependencies. |
| `runs/` and `results/` | Created by the commands above for your own models and local predictions. |

## Scope and troubleshooting

- `unsure` means the detector declined to assign a score. Inspect `reason` and the per-field coverage. Reasons include unsupported input, short components, a recognized private relay, and too few known fragments. A `null` score expresses missing evidence.
- `score_only` means the model could calculate a score but no cutoff caught a validation positive within the selected false-alert budget. Inspect `report.json` and collect better labeled examples. You can supply an explicit `--threshold` for experiments.
- The text model handles supported ASCII addresses. It uses the username and the registrable domain label. Quoted forms, internationalized addresses, unknown suffixes, and certain recognized relay domains receive an explicit abstention. It performs no mailbox queries. It cannot establish deliverability or ownership.
- The code refuses contradictory labels for the same model input and identifies lines to review. It needs at least five independent eligible username groups per class for its five-fold split, both classes in all three partitions, and a nonempty training vocabulary. These are mechanical minimums. Use many more reviewed examples for any real performance estimate.
- If tiny practice data produces an empty vocabulary, `--min-df 1` retains fragments seen in just one training address. The default is two. For an early learning exercise, the bundled examples give an immediate working reference.
- Training reads labeled examples into memory. Batch inference streams a line at a time. Run real customer data in an approved environment. Keep real lists, result files, and private trained vocabularies out of a public release.
