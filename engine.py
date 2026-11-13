"""Address preparation, feature extraction and inference. Standard library only."""

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cache
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
FORMAT = "whole-email-char-tfidf-logistic-v1"
SIGNAL_FORMAT = "whole-email-char-tfidf-signals-logistic-v2"
PREPROCESSOR = "bare-ascii-psl-823beb1-v1"
LOCAL = re.compile(r"[A-Za-z0-9!#$%&'*+/=?^_\x60{|}~.-]+")
DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
RELAY_HOSTS = {"privaterelay.appleid.com", "private.icloud.com"}


@dataclass(frozen=True)
class Parts:
    username: str
    domain: str
    original_username: str = field(default='', compare=False)
    full_name: str = field(default='', compare=False)


class Model(dict):
    """JSON-compatible model mapping with a cached domain lookup."""


@cache
def suffix_rules():
    # The leading byte in the pinned dataset identifies the ICANN or private section.
    with ZipFile(ROOT / "assets" / "public_suffix_list.dat") as archive:
        return frozenset(row[1:].decode("utf-8") for name in archive.namelist()
                         for row in archive.read(name).splitlines() if row)


def registrable_index(labels):
    rules = suffix_rules()
    longest = 0
    for index in range(len(labels)):
        tail = ".".join(labels[index:])
        if "!" + tail in rules:
            return index
        wildcard = index + 1 < len(labels) and "*." + ".".join(labels[index + 1:]) in rules
        if tail in rules or wildcard:
            longest = max(longest, len(labels) - index)
    if not longest or longest == len(labels):
        return None
    return len(labels) - longest - 1


def domain_label(labels):
    index = registrable_index(labels)
    return labels[index] if index is not None else None


def preprocess(address, full_name=''):
    """Return (feature copy, None) or (None, scope reason); never modifies a mailbox."""
    raw = address.strip()
    if not raw:
        return None, "empty_input"
    if not raw.isascii() or any(c in raw for c in '"<>[]()'):
        return None, "unsupported_address_form"
    if raw.count("@") != 1 or len(raw) > 254:
        return None, "invalid_address_shape"
    local, host = raw.lower().split("@")
    if (not LOCAL.fullmatch(local) or len(local) > 64 or local.startswith(".")
            or local.endswith(".") or ".." in local):
        return None, "invalid_address_shape"
    labels = host.split(".")
    if len(labels) < 2 or len(host) > 253 or not all(DNS_LABEL.fullmatch(s) for s in labels):
        return None, "invalid_address_shape"
    if any(s.startswith("xn--") for s in labels):
        return None, "unsupported_address_form"
    if host in RELAY_HOSTS:
        return None, "recognized_private_relay"
    domain = domain_label(labels)
    if domain is None:
        return None, "no_recognized_registrable_domain"
    if len(local) < 3 or len(domain) < 3:
        return None, "component_too_short"
    return Parts(local, domain, raw.split('@')[0], full_name or ''), None


SIGNALS = ('alphanumeric', 'non_alphanumeric', 'vowels', 'consonants', 'hyphens',
           'colons', 'numbers', 'numbers_in_a_row', 'capital_letters', 'length',
           'vowel_ratio', 'consonant_ratio', 'non_alphanumeric_ratio',
           'name_similarity', 'name_available')


def address_signals(parts):
    """Measurements of the original username; names are optional and never saved."""
    raw = parts.original_username or parts.username
    lower = raw.lower()
    length = len(raw)
    vowels = sum(c in 'aeiou' for c in lower)
    consonants = sum('a' <= c <= 'z' and c not in 'aeiou' for c in lower)
    non_alphanumeric = sum(not c.isalnum() for c in raw)
    numbers_in_a_row = max((len(m.group()) for m in re.finditer(r'[0-9]+', raw)), default=0)
    name = ''.join(c for c in parts.full_name.casefold() if c.isalnum())
    user = ''.join(c for c in lower if c.isalnum())
    similarity = SequenceMatcher(None, name, user, autojunk=False).ratio() if name else None
    return {'alphanumeric': sum(c.isalnum() for c in raw), 'non_alphanumeric': non_alphanumeric,
            'vowels': vowels, 'consonants': consonants, 'hyphens': raw.count('-'),
            'colons': raw.count(':'), 'numbers': sum(c.isdigit() for c in raw),
            'numbers_in_a_row': numbers_in_a_row, 'capital_letters': sum(c.isupper() for c in raw),
            'length': length, 'vowel_ratio': vowels / length, 'consonant_ratio': consonants / length,
            'non_alphanumeric_ratio': non_alphanumeric / length,
            'name_similarity': similarity, 'name_available': int(bool(name))}


def signal_vector(parts):
    values = address_signals(parts)
    # Counts are scaled to the allowed username length; ratios are already in [0, 1].
    return [(values[k] or 0) / (64 if i < 10 else 1) for i, k in enumerate(SIGNALS)]


def grams(text):
    for size in (2, 3, 4):
        for start in range(len(text) - size + 1):
            yield text[start:start + size]


def feature_terms(parts):
    for field, value in (("user", parts.username), ("domain", parts.domain)):
        for gram in grams(value):
            yield field + ":" + gram


def load_model(path=None):
    path = Path(path) if path is not None else ROOT / "demo"
    file = path / "model.json" if path.is_dir() else path
    model = json.loads(file.read_text(encoding="utf-8"))
    if (not isinstance(model, dict) or model.get("format") not in (FORMAT, SIGNAL_FORMAT)
            or model.get("preprocessor") != PREPROCESSOR or not isinstance(model.get("name"), str)
            or not isinstance(model.get("terms"), dict) or not model["terms"]
            or "threshold" not in model):
        raise ValueError("unsupported or incomplete model.json")
    def finite(x):
        return type(x) in (float, int) and math.isfinite(x)
    if not finite(model.get("intercept")) or not finite(model.get("min_coverage")) or not 0 <= model["min_coverage"] <= 1:
        raise ValueError("invalid model intercept or coverage floor")
    if model["threshold"] is not None and (not finite(model["threshold"]) or not 0 <= model["threshold"] <= 1):
        raise ValueError("invalid model threshold")
    for name, pair in model["terms"].items():
        field, _, term = name.partition(":")
        if (field not in {"user", "domain"} or not 2 <= len(term) <= 4 or not isinstance(pair, list)
                or len(pair) != 2 or not all(finite(x) for x in pair) or pair[0] <= 0):
            raise ValueError("invalid feature in model.json")
    signals = model.get('signal_weights')
    if (model['format'] == SIGNAL_FORMAT) != (signals is not None):
        raise ValueError('model format does not match its signal weights')
    if signals is not None and (not isinstance(signals, dict) or set(signals) != set(SIGNALS)
                                or not all(finite(v) for v in signals.values())):
        raise ValueError('invalid signal weights in model.json')
    domain_model = model.get('domain_model')
    if domain_model is not None:
        if model.get('flagged_domains'):
            raise ValueError('model cannot contain both a domain model and a legacy domain list')
        if (not isinstance(domain_model, dict) or domain_model.get('format') != 'registrable-domain-char-tfidf-logistic-v1'
                or not isinstance(domain_model.get('terms'), dict) or not domain_model['terms']
                or not finite(domain_model.get('intercept')) or not finite(domain_model.get('min_coverage'))
                or not 0 <= domain_model['min_coverage'] <= 1 or 'threshold' not in domain_model):
            raise ValueError('invalid domain model in model.json')
        cutoff = domain_model['threshold']
        if cutoff is not None and (not finite(cutoff) or not 0 <= cutoff <= 1):
            raise ValueError('invalid domain model threshold')
        for term, pair in domain_model['terms'].items():
            if (not isinstance(term, str) or not 2 <= len(term) <= 4 or not isinstance(pair, list)
                    or len(pair) != 2 or not all(finite(x) for x in pair) or pair[0] <= 0):
                raise ValueError('invalid domain model feature')
    domains = model.get("flagged_domains", [])
    if not isinstance(domains, list) or not all(isinstance(d, str) for d in domains):
        raise ValueError("invalid flagged domains in model.json")
    if type(model.get("include_subdomains", True)) is not bool:
        raise ValueError("invalid domain matching setting in model.json")
    from domain_rules import normalize_domain
    try:
        if any(normalize_domain(d) != d for d in domains):
            raise ValueError("invalid flagged domains in model.json")
    except ValueError:
        raise ValueError("invalid flagged domains in model.json") from None
    model = Model(model)
    model.domain_lookup = frozenset(domains)
    return model


def logistic(margin):
    if margin >= 0:
        return 1 / (1 + math.exp(-margin))
    e = math.exp(margin)
    return e / (1 + e)


def score_parts(parts, model, *, explain=False):
    """Return the evidence for both fields and the binary classifier's one score."""
    counts = Counter(feature_terms(parts))
    known = Counter()
    total = Counter()
    terms = []
    for name, count in counts.items():
        field = name.split(":", 1)[0]
        total[field] += count
        entry = model["terms"].get(name)
        if entry is not None:
            known[field] += count
            idf, weight = entry
            terms.append((name, field, count, count * idf, weight))
    coverage = {f: {"known": known[f], "total": total[f], "fraction": known[f] / total[f]}
                for f in ("user", "domain")}
    output = {"coverage": coverage, "score": None, "reason": None}
    if explain:
        output["parts"] = {"username": parts.username, "domain_label": parts.domain}
        output["fragments"] = {f: list(grams(t)) for f, t in
                               (("user", parts.username), ("domain", parts.domain))}
        output["known_fragments"] = {f: [g for g in items if f + ':' + g in model["terms"]]
                                     for f, items in output["fragments"].items()}
        output["min_coverage"] = model["min_coverage"]
        output['signals'] = address_signals(parts)
        output['signals_used'] = 'signal_weights' in model
    weak = [f for f, stat in coverage.items()
            if stat["known"] == 0 or stat["fraction"] < model["min_coverage"]]
    if weak:
        output["reason"] = "low_coverage_" + "_and_".join(weak)
        return output

    norm = math.sqrt(math.fsum(value * value for _, _, _, value, _ in terms))
    contributions = [(name, field, count, value / norm, weight, value / norm * weight)
                     for name, field, count, value, weight in terms]
    user_sum = math.fsum(t[-1] for t in contributions if t[1] == "user")
    domain_sum = math.fsum(t[-1] for t in contributions if t[1] == "domain")
    signal_effects = {k: value * model['signal_weights'][k] for k, value in zip(SIGNALS, signal_vector(parts))} if 'signal_weights' in model else {}
    signal_sum = math.fsum(signal_effects.values())
    margin = model["intercept"] + user_sum + domain_sum + signal_sum
    output["score"] = logistic(margin)
    if explain:
        output["math"] = {"intercept": model["intercept"], "user_sum": user_sum, "domain_sum": domain_sum,
                          "signal_sum": signal_sum, "margin": margin, "sigmoid_of_margin": output["score"]}
        output['signal_effects'] = signal_effects
        output["contributions"] = [{"term": name, "count": count, "tfidf": value, "weight": weight,
                                    "points": effect}
                                   for name, _, count, value, weight, effect in
                                   sorted(contributions, key=lambda t: (-abs(t[-1]), t[0]))]
    return output


def decide(score, threshold):
    if score is None:
        return "unsure"
    if threshold is None:
        return "score_only"
    return "looks_gibberish" if score >= threshold else "looks_ordinary"


def predict(address, model, *, threshold=None, explain=False, full_name=''):
    domains = getattr(model, 'domain_lookup', None)
    if domains is None:
        domains = frozenset(model.get('flagged_domains', ()))
    if domains:
        from domain_rules import match_domain
        matched = match_domain(address, domains, model.get('include_subdomains', True))
        if matched:
            return {'model': model['name'], 'decision': 'flagged_domain', 'score': None,
                    'threshold': None, 'coverage': None, 'reason': 'domain_list_match',
                    'matched_domain': matched, 'decision_source': 'domain_rule'}
    cutoff = model["threshold"] if threshold is None else threshold
    if cutoff is not None and not 0 <= cutoff <= 1:
        raise ValueError("threshold must be in [0, 1]")
    parts, reason = preprocess(address, full_name)
    evidence = (score_parts(parts, model, explain=explain) if parts else
                {"score": None, "coverage": None, "reason": reason})
    decision = decide(evidence["score"], cutoff)
    if decision == "score_only":
        evidence["reason"] = "no_usable_validation_threshold"
    result = {"model": model["name"], "decision": decision, "threshold": cutoff, **evidence}
    if model.get('domain_model'):
        from domain_ml import score_domain
        domain = score_domain(address, model['domain_model'])
        email_decision = decision
        domain_decision = domain['decision']
        if 'looks_gibberish' in (email_decision, domain_decision):
            final = 'looks_gibberish'
        elif email_decision == 'looks_ordinary' and domain_decision == 'looks_ordinary':
            final = 'looks_ordinary'
        else:
            final = 'unsure'
        email_reason = result['reason']
        if final == 'looks_gibberish':
            flagged_by = [name for name, value in (('email', email_decision), ('domain', domain_decision)) if value == 'looks_gibberish']
            final_reason = '_and_'.join(flagged_by) + '_model_flagged'
        elif final == 'unsure':
            final_reason = 'one_or_both_models_unsure'
        else:
            final_reason = None
        result.update({'decision': final, 'reason': final_reason, 'email_decision': email_decision,
                       'email_reason': email_reason, 'domain_result': domain, 'decision_source': 'combined_models'})
    return result
