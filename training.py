"""Read two labeled lists, fit one classifier, and measure it on separate data."""

from collections import Counter
from itertools import groupby
import json
import math
from pathlib import Path
import shutil
import tempfile

from engine import FORMAT, PREPROCESSOR, decide, feature_terms, predict, preprocess, score_parts


CLASSES = ("ordinary", "flagged")
PROBES = ("anu.rao27@gmail.com", "qzxv8801@gmail.com", "mukunda@gmail.com",
          "mukunda@qzxvpk.com", "MUKUNDA@GMAIL.COM", "rohan+travel@mail.pinepost.co.in",
          "x@gmail.com", "||||||||@gmail.com", "abc@privaterelay.appleid.com", "broken@@example.com")


def read_lists(flagged, ordinary):
    """One label for the whole address. Deduplicate at the representation the model sees."""
    seen = {}
    rows = []
    counts = {}
    for y, path in enumerate((ordinary, flagged)):
        path = Path(path)
        unique, duplicate, nonempty = 0, 0, 0
        excluded = Counter()
        with path.open(encoding="utf-8-sig") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                nonempty += 1
                parts, reason = preprocess(line)
                key = parts if parts is not None else line.strip().lower()
                previous = seen.get(key)
                if previous is not None:
                    py, pname, pn = previous
                    if py != y:
                        raise ValueError(f"contradictory labels for the same model input: "
                                         f"{pname} line {pn} and {path.name} line {n}; review these rows")
                    duplicate += 1
                    continue
                seen[key] = (y, path.name, n)
                if parts is None:
                    excluded[reason] += 1
                else:
                    rows.append((parts, y))
                    unique += 1
        counts[CLASSES[y]] = {"nonblank_lines": nonempty, "eligible_unique": unique,
                              "duplicate_lines": duplicate, "excluded_unique": dict(sorted(excluded.items()))}
    rows.sort(key=lambda r: (r[0].username, r[0].domain, r[1]))
    return rows, counts


def split_rows(rows):
    from sklearn.model_selection import StratifiedGroupKFold
    labels = [y for _, y in rows]
    # Related plus aliases share a split; the original plus tag is still a scoring feature.
    groups = [p.username.partition("+")[0] or p.username for p, _ in rows]
    for y in (0, 1):
        n = len({g for g, label in zip(groups, labels) if label == y})
        if n < 5:
            raise ValueError(f"{CLASSES[y]} has {n} distinct eligible username groups; "
                             "collect at least five per class for this five-fold split")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    folds = [test.tolist() for _, test in splitter.split(groups, labels, groups=groups)]
    splits = {"train": folds[2] + folds[3] + folds[4], "validation": folds[1], "test": folds[0]}
    for name, indices in splits.items():
        if {labels[i] for i in indices} != {0, 1}:
            raise ValueError(f"{name} lacks one class; add more independent groups from both classes")
    return {name: [rows[i] for i in indices] for name, indices in splits.items()}


def fit(rows, min_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    model = make_pipeline(
        TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=min_df,
                        max_features=50_000, norm="l2", smooth_idf=True),
        LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=500, random_state=43),
    )
    docs = [" ".join(feature_terms(p)) for p, _ in rows]
    model.fit(docs, [y for _, y in rows])
    vec, lr = model.steps[0][1], model.steps[1][1]
    if list(lr.classes_) != [0, 1]:
        raise ValueError("training requires both classes")
    terms = {key: [float(idf), float(coef)] for key, idf, coef in
             zip(vec.get_feature_names_out(), vec.idf_, lr.coef_[0], strict=True)}
    return model, terms, float(lr.intercept_[0])


def exported_scores(rows, model, fitted=None):
    scores = [score_parts(p, model)["score"] for p, _ in rows]
    if fitted is not None:
        expected = fitted.predict_proba([" ".join(feature_terms(p)) for p, _ in rows])[:, 1]
        for s, ref in zip(scores, expected, strict=True):
            if s is not None and not math.isclose(s, ref, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError("exported model disagrees with scikit-learn")
    return [(y, s) for (_, y), s in zip(rows, scores, strict=True)]


def choose_threshold(pairs, max_fpr):
    ordinary_n = sum(y == 0 for y, _ in pairs)
    ranked = sorted(((s, y) for y, s in pairs if s is not None), reverse=True)
    best, best_tp = None, 0
    tp = fp = 0
    for cutoff, group in groupby(ranked, key=lambda t: t[0]):
        for _, y in group:
            tp += y
            fp += 1 - y
        if fp / ordinary_n > max_fpr:
            break
        if tp > best_tp:
            best, best_tp = cutoff, tp
    return best


def metrics(pairs, threshold, extra_abstentions=(0, 0)):
    n0 = sum(y == 0 for y, _ in pairs) + extra_abstentions[0]
    n1 = sum(y == 1 for y, _ in pairs) + extra_abstentions[1]
    tp = fp = scored = 0
    for y, value in pairs:
        if value is None:
            continue
        scored += 1
        if decide(value, threshold) == "looks_gibberish":
            tp += y
            fp += 1 - y
    return {"ordinary": n0, "flagged": n1, "scored": scored, "abstained": n0 + n1 - scored,
            "coverage": scored / (n0 + n1) if n0 + n1 else None,
            "true_flags": tp, "false_flags": fp, "missed_flags": n1 - tp, "ordinary_unflagged": n0 - fp,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / n1 if n1 else None,
            "ordinary_false_flag_rate": fp / n0 if n0 else None}


def golden_cases(model, fitted):
    cases = []
    for email in PROBES:
        got = predict(email, model)
        score = got["score"]
        if score is not None:
            parts, _ = preprocess(email)
            score = float(fitted.predict_proba([" ".join(feature_terms(parts))])[0, 1])
        cases.append({"email": email, "score": score, "decision": got["decision"], "reason": got["reason"]})
    return cases


def train(flagged, ordinary, out, name=None, *, min_df=2, min_coverage=.20, max_fpr=.05, progress=None):
    def stage(message):
        if progress is not None:
            progress(message)
    stage('Loading training software')
    import sklearn
    out = Path(out)
    if out.exists():
        raise ValueError("output already exists; choose a new model directory")
    if min_df < 1 or not 0 <= min_coverage <= 1 or not 0 <= max_fpr <= 1:
        raise ValueError("min-df must be positive; coverage and false-flag budget must be in [0,1]")
    stage('Preparing and checking labeled data')
    rows, ingestion = read_lists(flagged, ordinary)
    stage('Splitting independent username groups')
    split = split_rows(rows)
    stage('Fitting vocabulary and model weights')
    fitted, terms, intercept = fit(split["train"], min_df)
    model = {"format": FORMAT, "preprocessor": PREPROCESSOR, "name": name or out.name, "intercept": intercept,
             "min_coverage": min_coverage, "threshold": None, "terms": terms}
    stage('Choosing a cutoff on validation data')
    validation = exported_scores(split["validation"], model, fitted)
    model["threshold"] = choose_threshold(validation, max_fpr)
    stage('Measuring the frozen model on test data')
    test = exported_scores(split["test"], model, fitted)
    report = {"name": model["name"], "scikit_learn_version": sklearn.__version__, "seed": 43,
              "ingestion": ingestion, "split_counts": {n: len(v) for n, v in split.items()},
              "training_class_counts": {label: sum(y == i for _, y in split["train"]) for i, label in enumerate(CLASSES)},
              "feature_count": len(terms), "min_df": min_df, "min_coverage": min_coverage,
              "validation_max_ordinary_false_flag_rate": max_fpr, "chosen_threshold": model["threshold"],
              "validation": metrics(validation, model["threshold"]), "test": metrics(test, model["threshold"])}
    stage('Saving model and reports')
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(dir=out.parent))
    try:
        for filename, data in [("model.json", model), ("report.json", report), ("golden.json", golden_cases(model, fitted))]:
            (temp / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                                                   sort_keys=True) + "\n", encoding="utf-8")
        temp.rename(out)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return {"model": model["name"], "directory": str(out), "split": report["split_counts"],
            "threshold": model["threshold"], "test": report["test"]}


def evaluate(flagged, ordinary, model, threshold=None, *, include_scores=False, include_breakdowns=False):
    rows, counts = read_lists(flagged, ordinary)
    cutoff = model["threshold"] if threshold is None else threshold
    if cutoff is not None and not 0 <= cutoff <= 1:
        raise ValueError("threshold must be in [0, 1]")
    extra = tuple(sum(counts[c]["excluded_unique"].values()) for c in CLASSES)
    if not rows and not sum(extra):
        raise ValueError("both input files are empty")
    pairs = exported_scores(rows, model)
    result = {"model": model["name"], "threshold": cutoff, "ingestion": counts,
              "results": metrics(pairs, cutoff, extra)}
    if include_scores:
        grouped = {}
        for label, score in pairs:
            if score is not None:
                counts_at_score = grouped.setdefault(score, [0, 0])
                counts_at_score[label] += 1
        result["score_groups"] = [[score, *counts] for score, counts in sorted(grouped.items())]
    if include_breakdowns:
        domains = {}
        buckets = {name: [] for name in ("0–<25%", "25–<50%", "50–<75%", "75–100%")}
        for (parts, label), (_, score) in zip(rows, pairs, strict=True):
            domains.setdefault(parts.domain, []).append((label, score))
            fraction = score_parts(parts, model)["coverage"]["user"]["fraction"]
            bucket = min(int(fraction * 4), 3)
            buckets[tuple(buckets)[bucket]].append((label, score))
        result["breakdowns"] = {
            "domain_labels": [{"label": domain, **metrics(values, cutoff)}
                              for domain, values in sorted(domains.items(), key=lambda item: (-len(item[1]), item[0]))],
            "username_coverage": [{"label": name, **metrics(values, cutoff)} for name, values in buckets.items()],
            "excluded_before_grouping": sum(extra),
        }
    return result
