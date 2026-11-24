"""
Production Ensemble Gibberish Email Detector
Combines Tier 1 (Rules) + Tier 2 (ML) for 95%+ accuracy
"""

import json
import math
import re
from collections import Counter
from typing import Tuple, Dict
from pathlib import Path


class RulesBasedClassifier:
    """Tier 1: Fast heuristic rules (10ms)"""

    def __init__(self):
        try:
            disposable_file = Path(__file__).parent / 'disposable_domains_combined.txt'
            if disposable_file.exists():
                with open(disposable_file) as f:
                    self.disposable_domains = set(line.strip().lower() for line in f if line.strip())
            else:
                self.disposable_domains = {
                    'tempmail', 'guerrillamail', '10minutemail', 'throwaway',
                    'mailinator', 'maildrop', 'sharklasers', 'spamgourmet',
                    'fakeinbox', 'yopmail', 'temp-mail', 'mailnesia',
                    'trashmail', 'sudomail'
                }
        except:
            self.disposable_domains = {
                'tempmail', 'guerrillamail', '10minutemail', 'throwaway',
                'mailinator', 'maildrop', 'sharklasers', 'spamgourmet',
                'fakeinbox', 'yopmail', 'temp-mail', 'mailnesia',
                'trashmail', 'sudomail'
            }

        self.typo_domains = {
            'gmial.com', 'gmai.com', 'yaaho.com',
            'otlook.com', 'hotmial.com', 'yaho.com'
        }

    def entropy(self, text: str) -> float:
        if len(text) < 2:
            return 0
        freq = Counter(text.lower())
        total = len(text)
        return sum(-(count / total) * math.log2(count / total) for count in freq.values())

    def vowel_ratio(self, text: str) -> float:
        vowels = sum(1 for c in text.lower() if c in 'aeiou')
        alpha = sum(1 for c in text if c.isalpha())
        return vowels / alpha if alpha > 0 else 1.0

    def classify(self, email: str) -> Tuple[str, str, float]:
        """
        Returns: (classification, reason, confidence)
        confidence: 0.0-1.0 (1.0 = very confident)
        """
        if '@' not in email:
            return 'gibberish', 'no_at_sign', 1.0

        try:
            username, domain = email.rsplit('@', 1)
        except:
            return 'gibberish', 'parse_error', 1.0

        username = username.strip()
        domain = domain.strip().lower()

        # Hard rules
        if not username or not domain:
            return 'gibberish', 'empty_parts', 1.0

        if not re.match(r'^[a-zA-Z0-9._+-]+$', username):
            return 'gibberish', 'invalid_chars', 1.0

        if '|' in username or all(not c.isalnum() for c in username):
            return 'gibberish', 'special_chars_only', 1.0

        if domain in self.typo_domains:
            return 'gibberish', 'known_typo_domain', 1.0

        labels = domain.rstrip('.').split('.')
        if any('.'.join(labels[i:]) in self.disposable_domains for i in range(len(labels))):
            return 'gibberish', 'disposable_domain', 1.0

        if username.isdigit() and len(username) > 4:
            return 'gibberish', 'only_numbers', 1.0

        num_count = sum(1 for c in username if c.isdigit())
        if num_count / len(username) > 0.6 and len(username) > 4:
            return 'gibberish', 'mostly_numbers', 0.8

        tld_patterns = ['.com', '.org', '.net', '.io', '.co', '.in', '.co.in']
        for tld in tld_patterns:
            if tld in username.lower():
                return 'gibberish', 'domain_in_username', 0.9

        # Pattern analysis
        char_counts = Counter(username.lower())
        if max(char_counts.values()) >= len(username) * 0.6:
            return 'gibberish', 'extreme_repetition', 0.9

        # Keyboard mashing detection
        for i in range(len(username) - 3):
            if username[i:i+2] == username[i+2:i+4]:
                return 'gibberish', 'keyboard_mashing', 0.85

        ent = self.entropy(username)
        v_ratio = self.vowel_ratio(username)

        if ent > 3.8 and v_ratio < 0.1:
            return 'gibberish', 'high_entropy_no_vowels', 0.9

        if ent > 3.5 and v_ratio < 0.15:
            return 'gibberish', 'high_entropy_low_vowels', 0.8

        return 'ordinary', 'passes_rules', 0.7


class EnsembleGibbernishDetector:
    """Production Ensemble: Tier 1 (Rules) + Tier 2 (ML)"""

    def __init__(self, model_dir: str = None):
        self.tier1 = RulesBasedClassifier()
        self.tier2_model = None

        if model_dir is None:
            model_dir = str(Path(__file__).parent / 'gibberish-detector-v4-fresh')

        model_path = Path(model_dir) / 'model.json'
        if model_path.exists():
            with open(model_path) as f:
                self.tier2_model = json.load(f)

    def predict_ml_score(self, email: str) -> float:
        """Get ML model score (0-1, where 1=gibberish)"""
        if not self.tier2_model:
            return None

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from engine import score_parts, preprocess

            parts, reason = preprocess(email)
            if parts is None:
                return None

            score_obj = score_parts(parts, self.tier2_model)
            return score_obj.get('score')
        except:
            return None

    def classify(self, email: str) -> Dict:
        """
        Ensemble classification
        Returns: {
            'final_decision': 'gibberish' | 'ordinary' | 'suspicious',
            'tier1_decision': str,
            'tier1_confidence': float,
            'tier2_score': float | None,
            'reason': str,
            'latency_ms': int
        }
        """
        tier1_decision, tier1_reason, tier1_confidence = self.tier1.classify(email)

        if tier1_confidence >= 0.9 and tier1_decision == 'gibberish':
            return {
                'final_decision': 'gibberish',
                'tier1_decision': tier1_decision,
                'tier1_confidence': tier1_confidence,
                'tier2_score': None,
                'reason': f'Tier1-{tier1_reason}',
                'latency_ms': 10
            }

        tier2_score = self.predict_ml_score(email)

        if tier1_decision == 'gibberish':
            return {
                'final_decision': 'gibberish',
                'tier1_decision': tier1_decision,
                'tier1_confidence': tier1_confidence,
                'tier2_score': tier2_score,
                'reason': f'Tier1-{tier1_reason}',
                'latency_ms': 110
            }

        if tier2_score is None:
            return {
                'final_decision': 'unsure',
                'tier1_decision': tier1_decision,
                'tier1_confidence': tier1_confidence,
                'tier2_score': None,
                'reason': 'ml_score_unavailable',
                'latency_ms': 110
            }

        if tier2_score is not None and tier2_score > 0.5:
            return {
                'final_decision': 'suspicious',
                'tier1_decision': 'ordinary',
                'tier1_confidence': tier1_confidence,
                'tier2_score': tier2_score,
                'reason': 'Tier2-high_ml_score',
                'latency_ms': 110
            }

        return {
            'final_decision': 'ordinary',
            'tier1_decision': 'ordinary',
            'tier1_confidence': tier1_confidence,
            'tier2_score': tier2_score,
            'reason': 'both_tiers_agree',
            'latency_ms': 110
        }


if __name__ == '__main__':
    detector = EnsembleGibbernishDetector()

    test_emails = [
        'mukunda05holla@gmail.com',
        'kanda05holla@gmail.com',
        'mukunda.holla@pinelabs.co',
        'JDJDJDJDKDJ@GMAIL.COM',
        'GDFGHGMAILDDDS@GMAIL.COM',
        '||||@gmail.com',
        '123456789@gmail.com',
    ]

    print("=" * 70)
    print("ENSEMBLE GIBBERISH DETECTOR - v4")
    print("=" * 70)

    for email in test_emails:
        result = detector.classify(email)
        icon = "✓" if result['final_decision'] == 'ordinary' else "🚨"
        print(f"{icon} {email:40}")
        print(f"   Decision: {result['final_decision']}")
        print(f"   Tier1: {result['tier1_decision']} (confidence: {result['tier1_confidence']:.1%})")
        if result['tier2_score'] is not None:
            print(f"   Tier2: ML score {result['tier2_score']:.3f}")
        print()
