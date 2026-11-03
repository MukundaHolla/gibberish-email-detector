# Standalone gibberish email detector

Give the trainer two local text files. One contains full email addresses your reviewers flagged for gibberish spelling. The other contains ordinary full email addresses. It learns one scoring model from the labels. The saved model can score a new bare email address or a whole file. The included demo was trained on invented examples. No real customer lists were supplied.

If machine learning is new to you, start with [How it works](HOW_IT_WORKS.md). It follows one email all the way from characters to a numeric score and includes training and prediction flow diagrams.

## Get started with the local interface

You need Python 3.11 or newer. Python 3.12 is the tested version. From a Terminal opened in the project directory, create an environment, install the training dependency, and start the app:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python local_ui.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Keep Terminal open while using the app; press **Ctrl-C** there to stop it. Next time, navigate back to the project directory and run `source .venv/bin/activate` followed by `python local_ui.py`. Use your Python executable in place of `python3.12` if it has a different name.

The server listens on your computer only. Training requires the installed dependency; the demo can be scored with the Python standard library. The demo model uses invented examples and is for learning the workflow, not for judging real-world performance.

## How to use the interface

### 1. Prepare your labeled files

Create two UTF-8 files: **flagged** addresses reviewed for gibberish spelling and **ordinary** addresses. The label describes spelling, not whether a mailbox works or a person is fraudulent. Keep uncertain examples out of both files.

- **TXT:** Put one complete email address on each line, with no header. Blank lines are ignored.
- **CSV:** Include a header row. A column named `email`, `email address`, or `email_address` is detected automatically. Otherwise, enter its exact header in the CSV column field.
- Keep independent holdout data aside before training. Use many reviewed examples; the code's minimum of five eligible username groups per class is only a mechanical requirement.

For a practice run, use the invented files [examples/flagged.txt](examples/flagged.txt) and [examples/ordinary.txt](examples/ordinary.txt).

### 2. Review the data and train

![The local app's training area, with separate file inputs for flagged and ordinary examples.](docs/screenshots/01-train.jpg)

1. Under **Train a model**, choose the flagged and ordinary files. For CSV files, enter the column names if needed.
2. Click **Preview data** to see duplicates, contradictory labels, and unsupported addresses. Select rows and click **Exclude selected & review again** to leave them out of this run; your original files are not changed. Replacing a file or changing a CSV column resets these exclusions.
3. Optionally enter a model name and expand **Training settings**. The defaults are a minimum fragment frequency of 2, minimum component coverage of 20%, and validation false-alert budget of 5%. The budget is a target measured on the validation data, not a future guarantee.
4. Click **Train model**. The status shows the current stage. Only one UI training run can run at a time. Refreshing the page loses the progress view, but training can continue while the server runs; stopping the server interrupts it.
5. Review the saved model's test metrics. The newly trained model is selected for testing, bulk analysis, and evaluation. Use fresh, independent data for a further performance check.

### 3. Test one email

![An invented email being scored, showing its decision, score, threshold, coverage, and fragment contributions.](docs/screenshots/02-test.jpg)

1. Under **Test an email**, select the demo or one of your saved models.
2. Enter a complete address and click **Analyze email**.
3. Read the decision, score, threshold, coverage, and strongest fragment contributions. Positive contributions push toward flagged; negative ones push toward ordinary.

`looks_gibberish` and `looks_ordinary` are model decisions, not proof of fraud or deliverability. `unsure` means the model declined to score; inspect the reason. `score_only` means it could calculate a score but has no saved cutoff.

### 4. Test and review a list

![Bulk analysis of the bundled invented sample list.](docs/screenshots/03-bulk.jpg)

1. Under **Test a list**, select a model and upload an unlabeled TXT or CSV file. For CSV, specify the email column if it isn't detected. Try [examples/incoming.txt](examples/incoming.txt) with the demo.
2. Click **Analyze list**. The summary counts decisions; use **Show** and the page controls to inspect results. The interface accepts up to 10,000 addresses per batch; the CLI's `batch` command is available for larger TXT lists.
3. Click **Download all results · CSV** to save the full list with line numbers, scores, decisions, model, threshold, coverage, and reasons. The download includes all decisions even when a filter is active. Cells that could be interpreted as spreadsheet formulas are prefixed with an apostrophe.

![Sample bulk results with review choices and export controls.](docs/screenshots/04-review.jpg)

To build a reviewed dataset, mark binary decisions **Correct**, **Incorrect**, or **Uncertain** in the last column. For `unsure` or `score_only`, assign a label directly or choose **Uncertain**. Download the reviewed CSV for an audit trail and the separate flagged and ordinary TXT files for a future training run. Uncertain and unreviewed rows are omitted from the TXT files; addresses containing newlines remain in the CSV only. Reviews live in the browser until exported and are cleared when you start a new batch or close the page.

### 5. Evaluate on independent data

![The holdout evaluation form with two example files selected.](docs/screenshots/05-evaluate.jpg)

1. Set aside separate, labeled flagged and ordinary files that were not used to train or select the model, preferably collected later and from an independent source.
2. Under **Evaluate a model**, select the model, choose the holdout files, and click **Evaluate model**.
3. Inspect coverage, recall, precision, false-alert rate, and the input review. Out-of-scope positives and other unscored positives count as missed flags. Group breakdowns use the saved threshold; small groups can give misleading rates, and a registrable domain label is not necessarily a unique provider.
4. Move **Explore a cutoff** to see hypothetical alert and miss counts. The slider does not modify the saved model or the group breakdowns. If you use this dataset to choose a cutoff, measure the chosen cutoff on a different independent holdout before relying on the result.

![Example evaluation results using the invented demo training files, which are not an independent holdout.](docs/screenshots/06-evaluation-results.jpg)

The screenshots use the invented demonstration files to show the workflow. Those files were used to build the demo model and **are not an independent holdout**; the shown results are not a performance estimate. Also, precision depends on the dataset's class balance; naturally sampled data helps estimate real alert volumes.

### 6. Compare saved models

Under **Model history**, select **Use model** to select a saved model across the testing and evaluation sections. Choose two models under **Compare two models** to inspect their reported settings and test metrics. The displayed date is the model file's last-modified time, not a guaranteed training date. Results from different test sets aren't directly comparable; evaluate both models on the same independent holdout for a fairer comparison.

### Where data is saved

- Trained models are in `runs/<model-id>/`: `model.json`, `report.json`, and `golden.json`. The included demo is in `demo/`.
- Your original input files stay where you put them. Temporary copies of training and holdout lists are written to the system temporary directory and removed after the operation.
- Bulk and review results remain in the browser until downloaded; downloads go to your browser's download location. Models and downloaded results can contain sensitive address information or learned fragments. Keep them out of public repositories.
- Do not expose the local server as a public, multi-user service; it is designed for use on your own computer.

## Command-line usage

The CLI supports the same model format and remains useful for scripting. Scoring uses only the standard library; training requires the environment installed above. The CLI's list commands take UTF-8 TXT files with one complete address per line.

### Try the demo from Terminal



Extract the ZIP. Open Terminal, type `cd `, drag the extracted `gibberish-email-detector` folder into Terminal, and press Return. Run `python3 --version`. Use Python 3.11 or later. Python 3.12 is the tested version. If you have Homebrew and need Python, `brew install python@3.12` installs it. In that case use `python3.12` wherever these commands say `python3`.

```bash
python3 detector.py explain 'mukunda@gmail.com'
python3 detector.py explain 'qzxv8801@gmail.com'
```

The demo scores are about `0.4126` and `0.7769`. Its saved cutoff is about `0.4508`. The first address falls below the cutoff. The second crosses it. For a short JSON result, use `score` instead of `explain`. These commands use Python's standard library and the included weights. Nothing is fitted, downloaded, or sent when you score.

```bash
python3 detector.py score 'mukunda@qzxvpk.com'
python3 detector.py explain 'qzxv8801@gmail.com' --all
python3 detector.py score 'qzxv8801@gmail.com' --threshold 0.80
python3 detector.py self-test
```

The threshold experiment leaves the score unchanged. The self-test should print `PASS — 10 reference cases for demo-invented-v1.`

### Train on your own two files

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

python detector.py score 'mukunda@gmail.com' --model runs/my-first-model
python detector.py explain 'mukunda@gmail.com' --model runs/my-first-model
python detector.py self-test --model runs/my-first-model
```

The output directory contains `model.json`, `report.json`, and `golden.json`. A score without `--model` deliberately uses the bundled demo. Training again uses a fresh output directory, for example `runs/second-model`. Updating a text file affects scores only after you train and select the new model.

To practice with the supplied **labeled** data before preparing yours, run the same command with `--flagged examples/flagged.txt --ordinary examples/ordinary.txt --out runs/practice`. In a new Terminal session, navigate back to the project and use `source .venv/bin/activate` again to access the installed training dependency.

### Score a bulk list or test real outcomes

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
answer = predict("mukunda@gmail.com", model)
print(answer["score"], answer["decision"])
```

## Project map

| File or directory | What it does |
|---|---|
| `detector.py` | The entry point. Implements `score`, `explain`, `batch`, `train`, `evaluate`, and `self-test`. |
| `engine.py` | Shared spelling preparation, feature counting, coverage checks, scoring arithmetic, and label decision. |
| `training.py` | Reads both labeled lists, separates training/validation/test groups, calls scikit-learn, chooses a cutoff, checks the export, and writes results. |
| `local_ui.py`, `ui/` | The local web server and interface for training, testing, review, evaluation, and model history. |
| `requirements.txt` | Pins the direct training dependency to scikit-learn 1.8.0. |
| `assets/public_suffix_list.dat` | An offline domain-suffix reference used to identify the registrable label, including multi-part suffixes such as `co.in`. |
| `assets/PUBLIC_SUFFIX_LICENSE.md` | Source revision and license notice for that third-party dataset. |
| `examples/ordinary.txt`, `examples/flagged.txt` | All 480 invented, labeled full addresses used to build the bundled demonstration. |
| `examples/incoming.txt` | Four sample inputs for scoring a batch. |
| `demo/model.json` | The learned demo weights and associated scoring settings. |
| `demo/report.json` | Counts and evaluation results from training the demo. This is the file to read when assessing quality. |
| `demo/golden.json` | Ten fixed invented inputs and their expected outputs for repeatability. |
| `test_detector.py`, `test_local_ui.py`, `test_review.js` | Regression checks for the detector and local interface. Run `python -m unittest -v test_detector test_local_ui` and `node --test test_review.js` after installing the training dependency. |
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
