"""A separate character-fragment classifier for registrable domains."""
from collections import Counter
import math
from engine import decide, grams, logistic, registrable_index
from domain_rules import email_host, normalize_domain

DOMAIN_FORMAT = 'registrable-domain-char-tfidf-logistic-v1'


def canonical_domain(value):
    host = normalize_domain(value)
    labels = host.split('.')
    index = registrable_index(labels)
    if index is None:
        raise ValueError('domain has no recognized registrable suffix')
    return '.'.join(labels[index:])


def domain_terms(domain):
    return [g for g in grams(domain)]


def score_domain(address, model):
    host = email_host(address)
    if host is None:
        return {'decision': 'unsure', 'score': None, 'reason': 'unsupported_address', 'coverage': None}
    try:
        domain = canonical_domain(host)
    except ValueError:
        return {'decision': 'unsure', 'score': None, 'reason': 'unsupported_domain', 'coverage': None}
    counts = Counter(domain_terms(domain))
    total = sum(counts.values())
    terms = [(g, count, model['terms'][g]) for g, count in counts.items() if g in model['terms']]
    known = sum(count for _, count, _ in terms)
    coverage = {'known': known, 'total': total, 'fraction': known / total if total else 0}
    result = {'domain': domain, 'coverage': coverage, 'threshold': model['threshold'], 'score': None}
    if not known or coverage['fraction'] < model['min_coverage']:
        return {**result, 'decision': 'unsure', 'reason': 'low_domain_coverage'}
    norm = math.sqrt(math.fsum((count * pair[0]) ** 2 for _, count, pair in terms))
    contributions = [(term, count * pair[0] / norm * pair[1]) for term, count, pair in terms]
    margin = model['intercept'] + math.fsum(value for _, value in contributions)
    score = logistic(margin)
    return {**result, 'score': score, 'decision': decide(score, model['threshold']),
            'reason': 'no_usable_validation_threshold' if model['threshold'] is None else None,
            'contributions': [{'fragment': term, 'points': value} for term, value in sorted(contributions, key=lambda x: -abs(x[1]))[:8]]}


def fit_domain_model(flagged, ordinary, *, min_df=2, min_coverage=.2, max_fpr=.05, progress=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from training import choose_threshold, metrics

    rows, seen = [], {}
    for label, domains in ((0, ordinary), (1, flagged)):
        for raw in domains:
            try:
                domain = canonical_domain(raw)
            except ValueError as exc:
                raise ValueError(f'Domain {raw!r}: {exc}') from None
            if domain in seen:
                if seen[domain] != label:
                    raise ValueError(f'Domain {domain!r} appears in both flagged and ordinary domain files, including subdomains. Resolve the conflicting labels.')
                continue
            seen[domain] = label
            rows.append((domain, label))
    rows.sort()
    for label, name in ((0, 'ordinary'), (1, 'flagged')):
        n = sum(y == label for _, y in rows)
        if n < 5:
            raise ValueError(f'The {name} domain file needs at least 5 independent registrable domains; found {n}.')
    labels = [y for _, y in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    folds = [test.tolist() for _, test in splitter.split(rows, labels, groups=[d for d, _ in rows])]
    split = {'train':folds[2]+folds[3]+folds[4], 'validation':folds[1], 'test':folds[0]}
    for name, indices in split.items():
        if {labels[i] for i in indices} != {0,1}:
            raise ValueError(f'Domain {name} split lacks one class; add more independent domains in both files.')
    docs = [' '.join(domain_terms(domain)) for domain, _ in rows]
    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=min_df,
                          max_features=50000, norm='l2', smooth_idf=True)
    train_idx = split['train']
    X = vec.fit_transform([docs[i] for i in train_idx])
    clf = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=500, random_state=43)
    clf.fit(X, [labels[i] for i in train_idx])
    model = {'format':DOMAIN_FORMAT,'terms':{key:[float(idf),float(coef)] for key,idf,coef in zip(vec.get_feature_names_out(),vec.idf_,clf.coef_[0],strict=True)},
             'intercept':float(clf.intercept_[0]),'min_coverage':min_coverage,'threshold':None}
    def samples(indices):
        output = [(labels[i], score_domain('test@'+rows[i][0], model)['score']) for i in indices]
        expected = clf.predict_proba(vec.transform([docs[i] for i in indices]))[:, 1]
        for (_, score), reference in zip(output, expected, strict=True):
            if score is not None and not math.isclose(score, reference, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError('Exported domain model disagrees with scikit-learn')
        return output
    validation = samples(split['validation'])
    model['threshold'] = choose_threshold(validation, max_fpr)
    test = samples(split['test'])
    report = {'split_counts':{key:len(value) for key,value in split.items()},'feature_count':len(model['terms']),
              'input_counts':{'ordinary':sum(y==0 for _,y in rows),'flagged':sum(y==1 for _,y in rows)},
              'min_df':min_df,'min_coverage':min_coverage,'validation_max_ordinary_false_flag_rate':max_fpr,
              'chosen_threshold':model['threshold'],'validation':metrics(validation,model['threshold']),'test':metrics(test,model['threshold'])}
    return model, report
