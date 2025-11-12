#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re, unicodedata, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize('NFC', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def main():
    parser = argparse.ArgumentParser(description='One-shot trainer for Cocoon chatbot intents.')
    parser.add_argument('--intents', required=True, help='CSV with columns: text,label')
    parser.add_argument('--entities', required=True, help='JSONL with {label, pattern} per line')
    parser.add_argument('--rules', required=True, help='rules.json (validated only)')
    parser.add_argument('--outdir', default='artifacts', help='Output directory')
    args = parser.parse_args()

    intents_csv = Path(args.intents)
    entities_jsonl = Path(args.entities)
    rules_json = Path(args.rules)
    outdir = Path(args.outdir)

    for p in [intents_csv, entities_jsonl, rules_json]:
        if not p.exists():
            sys.exit(f'[ERROR] Missing file: {p}')

    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(intents_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'text' not in df.columns or 'label' not in df.columns:
        sys.exit('[ERROR] CSV must have columns: text,label')

    df['text_norm'] = df['text'].astype(str).apply(normalize_text)

    # Split (robust to tiny datasets)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            df['text_norm'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            df['text_norm'], df['label'], test_size=0.2, random_state=42
        )

    # Model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))),
        ('clf', LogisticRegression(max_iter=2000, n_jobs=1))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, digits=3)

    # Save artifacts
    joblib.dump(pipeline, outdir / 'intent_model.joblib')
    labels = sorted(df['label'].unique().tolist())
    (outdir / 'intent_labels.json').write_text(
        json.dumps({'labels': labels}, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    # Validate rules.json
    json.loads(rules_json.read_text(encoding='utf-8'))

    # Compile entity patterns
    compiled = []
    with entities_jsonl.open('r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if 'label' not in o or 'pattern' not in o:
                sys.exit('[ERROR] Each JSONL line must contain keys: label, pattern')
            compiled.append({'label': o['label'], 'pattern': o['pattern']})
    (outdir / 'entity_patterns_compiled.json').write_text(
        json.dumps(compiled, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    # Write a simple report
    (outdir / 'training_report.txt').write_text(
        f'Accuracy: {acc:.3f}\n\n{report}', encoding='utf-8'
    )

    print('[OK] Training finished.')
    print(f'- Saved to: {outdir.resolve()}')
    print(f'- Validation accuracy: {acc:.3f}')

if __name__ == '__main__':
    main()
