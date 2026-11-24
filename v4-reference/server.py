"""
Flask server for Gibberish Email Detector
Accessible via Cloudflare Tunnel for remote testing
"""

from flask import Flask, render_template, jsonify, request
import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ensemble_gibberish_detector import EnsembleGibbernishDetector

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Load model
MODEL_DIR = Path(__file__).parent / 'runs' / 'gibberish-detector-v4-fresh'
detector = EnsembleGibbernishDetector(str(MODEL_DIR))

MODEL_INFO = {
    'name': 'gibberish-detector-v4-fresh',
    'version': '4.0',
    'recall': '75.3%',
    'precision': '85.2%',
    'false_alert_rate': '5.1%',
    'coverage': '99.96%',
    'latency_ms': 110,
    'training_emails': 152731,
    'gibberish_examples': 41780,
    'ordinary_examples': 106951,
    'tier1': 'Rules-based (10ms) with 121,576 disposable domains',
    'tier2': 'ML Model (100ms) - Logistic Regression + TF-IDF'
}

@app.route('/')
def index():
    return render_template('index.html', model_info=MODEL_INFO)

@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify an email"""
    data = request.json
    email = data.get('email', '').strip().lower()

    if not email:
        return jsonify({'error': 'Email required'}), 400

    if '@' not in email:
        return jsonify({'error': 'Invalid email format'}), 400

    try:
        result = detector.classify(email)
        return jsonify({
            'email': email,
            'decision': result['final_decision'],
            'tier1_decision': result['tier1_decision'],
            'tier1_confidence': f"{result['tier1_confidence']:.1%}",
            'tier2_score': f"{result['tier2_score']:.3f}" if result['tier2_score'] else None,
            'reason': result['reason'],
            'latency_ms': result['latency_ms']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify(MODEL_INFO)

@app.route('/api/test-batch', methods=['POST'])
def test_batch():
    """Test multiple emails"""
    data = request.json
    emails = data.get('emails', [])

    if not emails:
        return jsonify({'error': 'No emails provided'}), 400

    results = []
    for email in emails:
        email = email.strip().lower()
        if email and '@' in email:
            try:
                result = detector.classify(email)
                results.append({
                    'email': email,
                    'decision': result['final_decision'],
                    'tier1_decision': result['tier1_decision'],
                    'tier1_confidence': f"{result['tier1_confidence']:.1%}",
                    'tier2_score': f"{result['tier2_score']:.3f}" if result['tier2_score'] else None,
                    'reason': result['reason'],
                    'latency_ms': result['latency_ms']
                })
            except:
                pass

    return jsonify({
        'total': len(results),
        'gibberish': sum(1 for r in results if r['decision'] == 'gibberish'),
        'ordinary': sum(1 for r in results if r['decision'] == 'ordinary'),
        'suspicious': sum(1 for r in results if r['decision'] == 'suspicious'),
        'results': results
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8765, debug=False)
