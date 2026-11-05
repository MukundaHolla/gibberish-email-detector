"""Explicit full-domain flagging rules, separate from spelling-model predictions."""
import csv
import io
from engine import DNS_LABEL, LOCAL


def normalize_domain(value):
    host = value.strip().rstrip('.').lower()
    if not host or any(c in host for c in '@/:*?#\\'):
        raise ValueError('enter a full domain such as example.com, without a URL, email, or wildcard')
    try:
        host = host.encode('idna').decode('ascii')
    except UnicodeError:
        raise ValueError('invalid internationalized domain') from None
    labels = host.split('.')
    if len(host) > 253 or len(labels) < 2 or not all(DNS_LABEL.fullmatch(label) for label in labels):
        raise ValueError('invalid domain; use a full domain such as example.com')
    return host


def parse_domain_file(data):
    name = str(data.get('name', ''))
    content = str(data.get('content', '')).lstrip('\ufeff')
    column = str(data.get('column', '')).strip()
    if name.lower().endswith('.txt'):
        rows = [(i, line) for i, line in enumerate(content.splitlines(), 1) if line.strip()]
    elif name.lower().endswith('.csv'):
        reader = csv.reader(io.StringIO(content, newline=''), strict=True)
        try:
            records = [(reader.line_num, row) for row in reader if any(v.strip() for v in row)]
        except csv.Error as e:
            raise ValueError(f'Invalid CSV: {e}') from None
        if not records:
            raise ValueError('The domain file is empty.')
        headers = records[0][1]
        if column:
            if column not in headers:
                raise ValueError(f'Column {column!r} was not found in the CSV header.')
            idx, records = headers.index(column), records[1:]
        else:
            matches = [i for i, h in enumerate(headers) if h.strip().lower() in ('domain', 'domain_name', 'domain name', 'host', 'hostname')]
            if len(matches) == 1:
                idx, records = matches[0], records[1:]
            elif len(headers) == 1:
                idx = 0
            else:
                raise ValueError('Enter the domain column name from the CSV header.')
        rows = [(line, row[idx] if idx < len(row) else '') for line, row in records]
    else:
        raise ValueError('Choose a .txt or .csv domain file.')
    domains, seen, errors, duplicates = [], set(), [], 0
    for line, value in rows:
        try:
            domain = normalize_domain(value)
        except ValueError as exc:
            errors.append(f'Line {line}: {exc}')
            continue
        if domain in seen:
            duplicates += 1
        else:
            seen.add(domain)
            domains.append(domain)
    if errors:
        raise ValueError(f'{len(errors)} invalid domain row(s). No changes saved. ' + '; '.join(errors[:5]))
    if not domains:
        raise ValueError('The domain file has no domains.')
    return {'domains': domains, 'duplicates': duplicates}


def email_host(address):
    raw = address.strip()
    if not raw.isascii() or raw.count('@') != 1 or len(raw) > 254:
        return None
    local, host = raw.lower().split('@')
    if (not LOCAL.fullmatch(local) or len(local) > 64 or local.startswith('.')
            or local.endswith('.') or '..' in local or host.endswith('.')):
        return None
    try:
        return normalize_domain(host)
    except ValueError:
        return None


def match_domain(address, domains, include_subdomains=False):
    host = email_host(address)
    if host is None:
        return None
    if host in domains:
        return host
    if include_subdomains:
        labels = host.split('.')
        for i in range(1, len(labels) - 1):
            suffix = '.'.join(labels[i:])
            if suffix in domains:
                return suffix
    return None


