"""Train or use a standalone whole-address gibberish detector."""

import sys

if sys.version_info < (3, 11):
    raise SystemExit("Use Python 3.11 or later. Python 3.12 is used in the walkthrough.")

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from engine import ROOT, load_model, predict


def show(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def unit_interval(text):
    v = float(text)
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError("choose a number between 0 and 1")
    return v


def explain_one(email, model, threshold, all_terms):
    r = predict(email, model, threshold=threshold, explain=True)
    print("Model:", r["model"])
    if "parts" in r:
        print("1. Prepare the address:", r["parts"]["username"], "(username),", r["parts"]["domain_label"], "(domain label)")
        print("2. Count overlapping strings of 2, 3, and 4 characters.")
        for field, items in r["fragments"].items():
            shown = items if all_terms else items[:14]
            print(f"   {field}: {', '.join(shown)}" + (" …" if len(shown) < len(items) else ""))
        print("3. Check how much of each component the model has seen.")
        for field, stat in r["coverage"].items():
            print(f"   {field}: {stat['known']}/{stat['total']} known ({stat['fraction']:.1%}); "
                  f"minimum {model['min_coverage']:.0%}")
    if r["score"] is None:
        print("Decision: unsure. Reason:", r["reason"], "| score: null")
        return
    m = r["math"]
    print("4. Add learned contributions.")
    print(f"   starting weight {m['intercept']:+.6f}  username {m['user_sum']:+.6f}  "
          f"domain {m['domain_sum']:+.6f}  = margin {m['margin']:+.6f}")
    print(f"   score = 1 / (1 + exp(-margin)) = {r['score']:.6f}")
    terms = r["contributions"] if all_terms else r["contributions"][:5]
    for t in terms:
        print(f"   {t['term']}  input {t['tfidf']:.6f} × learned weight {t['weight']:+.6f}"
              f" = {t['points']:+.6f}")
    if r["threshold"] is None:
        print("5. No cutoff met the validation budget. Returning a score without a label.")
    else:
        print(f"5. Use the cutoff {r['threshold']:.6f}. Decision: {r['decision']}.")


def batch(source, out, model, threshold):
    if out.exists():
        raise ValueError("output exists; use a fresh filename")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    count = 0
    try:
        with source.open(encoding="utf-8-sig") as src, tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=out.parent, delete=False) as f:
            temp = Path(f.name)
            for n, line in enumerate(src, 1):
                address = line.strip()
                if address:
                    data = {"line": n, "email": address, **predict(address, model, threshold=threshold)}
                    f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
        os.link(temp, out)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return {"scored_or_returned_unsure": count, "output": str(out)}


def check_model(directory):
    model = load_model(directory)
    parent = directory if directory.is_dir() else directory.parent
    cases = json.loads((parent / "golden.json").read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("golden.json is empty")
    for i, expected in enumerate(cases, 1):
        got = predict(expected["email"], model)
        score = expected["score"]
        same_score = (got["score"] is None if score is None else
                      got["score"] is not None and math.isclose(got["score"], score, abs_tol=1e-12, rel_tol=0))
        if not same_score or got["decision"] != expected["decision"] or got["reason"] != expected["reason"]:
            raise ValueError(f"reference case {i} failed")
    print(f"PASS — {len(cases)} reference cases for {model['name']}.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="cmd", required=True)
    train = cmds.add_parser("train", help="learn from two text files with one email per line")
    check = cmds.add_parser("evaluate", help="measure a saved model on independent labeled lists")
    for parser in (train, check):
        parser.add_argument("--flagged", type=Path, required=True)
        parser.add_argument("--ordinary", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True, help="new output directory")
    train.add_argument("--name", help="model identifier (default: output directory name)")
    train.add_argument("--min-df", type=int, default=2, help="minimum training documents per feature")
    train.add_argument("--min-coverage", type=unit_interval, default=.20)
    train.add_argument("--max-fpr", type=unit_interval, default=.05, help="empirical validation false-flag budget")
    one = cmds.add_parser("score", help="score one bare email address")
    explain = cmds.add_parser("explain", help="trace one example step by step")
    for parser in (one, explain):
        parser.add_argument("email")
    explain.add_argument("--all", action="store_true", help="show every recognized feature contribution")
    many = cmds.add_parser("batch", help="score every nonblank line in a text file")
    many.add_argument("input", type=Path)
    many.add_argument("--out", type=Path, required=True, help="new local JSONL output file")
    verify = cmds.add_parser("self-test", help="check saved reference cases")
    for parser in (one, explain, many, check, verify):
        parser.add_argument("--model", type=Path, default=ROOT / "demo", help="model directory or model.json")
    for parser in (one, explain, many, check):
        parser.add_argument("--threshold", type=unit_interval, help="override the saved cutoff for this run")
    args = p.parse_args()
    try:
        if args.cmd == "train":
            from training import train as fit
            show(fit(args.flagged, args.ordinary, args.out, args.name, min_df=args.min_df,
                     min_coverage=args.min_coverage, max_fpr=args.max_fpr))
        elif args.cmd == "self-test":
            check_model(args.model)
        else:
            model = load_model(args.model)
            if args.cmd == "score":
                show(predict(args.email, model, threshold=args.threshold))
            elif args.cmd == "explain":
                explain_one(args.email, model, args.threshold, args.all)
            elif args.cmd == "batch":
                show(batch(args.input, args.out, model, args.threshold))
            else:
                from training import evaluate
                show(evaluate(args.flagged, args.ordinary, model, args.threshold))
    except ModuleNotFoundError as e:
        if e.name != "sklearn":
            raise
        p.exit(2, "Training needs scikit-learn. Activate .venv and run: python -m pip install -r requirements.txt\n")
    except (ValueError, OSError) as e:
        p.exit(2, f"detector: {e}\n")


if __name__ == "__main__":
    main()
