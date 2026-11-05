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


def train(flagged, ordinary, out, name=None, *, min_df=2, min_coverage=.20, max_fpr=.05, progress=None, domain_data=None):
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
    domain_report = None
    if domain_data:
        stage('Training the domain model')
        from domain_ml import fit_domain_model
        model['domain_model'], domain_report = fit_domain_model(domain_data['flagged'], domain_data['ordinary'],
                                                                  min_df=min_df, min_coverage=min_coverage, max_fpr=max_fpr)
    report = {"name": model["name"], "scikit_learn_version": sklearn.__version__, "seed": 43,
              "ingestion": ingestion, "split_counts": {n: len(v) for n, v in split.items()},
              "training_class_counts": {label: sum(y == i for _, y in split["train"]) for i, label in enumerate(CLASSES)},
              "feature_count": len(terms), "min_df": min_df, "min_coverage": min_coverage,
              "validation_max_ordinary_false_flag_rate": max_fpr, "chosen_threshold": model["threshold"],
              "validation": metrics(validation, model["threshold"]), "test": metrics(test, model["threshold"]),
              "domain_model": domain_report, "metrics_scope": "email_model_only"}
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


def evaluate_with_domains(flagged, ordinary, model, threshold=None, *, include_scores=False, include_breakdowns=False):
    """Evaluate the full decision pipeline while retaining full email domains."""
    from domain_rules import email_host
    cutoff = model['threshold'] if threshold is None else threshold
    if cutoff is not None and not 0 <= cutoff <= 1:
        raise ValueError('threshold must be in [0, 1]')
    seen, rows, ingestion = {}, [], {}
    for label, path in enumerate((ordinary, flagged)):
        unique = duplicate = nonblank = 0
        excluded = Counter()
        with Path(path).open(encoding='utf-8-sig') as file:
            for line_number, line in enumerate(file, 1):
                address = line.strip()
                if not address:
                    continue
                nonblank += 1
                parts, reason = preprocess(address)
                host = email_host(address)
                key = (parts, host) if parts is not None else address.lower()
                previous = seen.get(key)
                if previous is not None:
                    if previous != label:
                        raise ValueError('contradictory labels for the same email input; review the holdout files')
                    duplicate += 1
                    continue
                seen[key] = label
                outcome = predict(address, model, threshold=cutoff)
                if parts is None and outcome['decision'] not in ('flagged_domain', 'looks_gibberish'):
                    excluded[reason] += 1
                else:
                    unique += 1
                    rows.append((parts, label, outcome))
        ingestion[CLASSES[label]] = {'nonblank_lines': nonblank, 'eligible_unique': unique,
                                     'duplicate_lines': duplicate, 'excluded_unique': dict(sorted(excluded.items()))}
    extra = tuple(sum(ingestion[c]['excluded_unique'].values()) for c in CLASSES)
    if not rows and not sum(extra):
        raise ValueError('both input files are empty')

    def summarize(items, extras=(0, 0)):
        ordinary_n = sum(label == 0 for _, label, _ in items) + extras[0]
        flagged_n = sum(label == 1 for _, label, _ in items) + extras[1]
        scored = sum(outcome['score'] is not None for _, _, outcome in items)
        tp = sum(label == 1 and outcome['decision'] in ('looks_gibberish', 'flagged_domain') for _, label, outcome in items)
        fp = sum(label == 0 and outcome['decision'] in ('looks_gibberish', 'flagged_domain') for _, label, outcome in items)
        total = ordinary_n + flagged_n
        return {'ordinary': ordinary_n, 'flagged': flagged_n, 'scored': scored, 'abstained': total - scored,
                'coverage': scored / total if total else None, 'true_flags': tp, 'false_flags': fp,
                'missed_flags': flagged_n - tp, 'ordinary_unflagged': ordinary_n - fp,
                'precision': tp / (tp + fp) if tp + fp else None,
                'recall': tp / flagged_n if flagged_n else None,
                'ordinary_false_flag_rate': fp / ordinary_n if ordinary_n else None}

    def domain_alert(outcome):
        return (outcome['decision'] == 'flagged_domain' or
                outcome.get('domain_result', {}).get('decision') == 'looks_gibberish')
    rules = [row for row in rows if domain_alert(row[2])]
    result = {'model': model['name'], 'threshold': cutoff, 'ingestion': ingestion,
              'results': summarize(rows, extra), 'domain_flags': {
                  'ordinary': sum(label == 0 for _, label, _ in rules),
                  'flagged': sum(label == 1 for _, label, _ in rules)},
              'domain_unscored': sum(outcome['score'] is None for _, _, outcome in rules),
              'domain_flag_kind': 'model' if model.get('domain_model') else 'rule',
              'decision_coverage': sum(outcome['decision'] in ('looks_gibberish', 'looks_ordinary', 'flagged_domain') for _, _, outcome in rows) / (len(rows) + sum(extra)) if rows or sum(extra) else None,
              'evaluation_scope': 'model_and_domains'}
    if include_scores:
        grouped = {}
        for _, label, outcome in rows:
            score = outcome['score']
            if score is not None and not domain_alert(outcome):
                counts = grouped.setdefault(score, [0, 0])
                counts[label] += 1
        result['score_groups'] = [[score, *counts] for score, counts in sorted(grouped.items())]
    if include_breakdowns:
        domains = {}
        bucket_names = ('0–<25%', '25–<50%', '50–<75%', '75–100%')
        buckets = {name: [] for name in bucket_names}
        no_parts = 0
        for row in rows:
            parts, _, _ = row
            if parts is None:
                no_parts += 1
                continue
            domains.setdefault(parts.domain, []).append(row)
            fraction = score_parts(parts, model)['coverage']['user']['fraction']
            buckets[bucket_names[min(int(fraction * 4), 3)]].append(row)
        result['breakdowns'] = {
            'domain_labels': [{'label': domain, **summarize(values)} for domain, values in sorted(domains.items(), key=lambda item: (-len(item[1]), item[0]))],
            'username_coverage': [{'label': name, **summarize(values)} for name, values in buckets.items()],
            'excluded_before_grouping': sum(extra) + no_parts}
    return result


def evaluate(flagged, ordinary, model, threshold=None, *, include_scores=False, include_breakdowns=False):
    if model.get('flagged_domains') or model.get('domain_model'):
        return evaluate_with_domains(flagged, ordinary, model, threshold, include_scores=include_scores, include_breakdowns=include_breakdowns)
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
