# -*- coding: utf-8 -*-
"""
Portable inference script for Cocoon chatbot.
Usage examples:
    python infer_chatbot.py --artifacts artifacts --rules rules.json "Da dau mun an dung gi?"
    python infer_chatbot.py --artifacts artifacts "Serum nghe gia bao nhieu?"
Notes:
- --rules is optional. If omitted, the script will try: <artifacts>/rules.json then parent folder ./rules.json
"""
import sys, json, re, unicodedata, argparse
from pathlib import Path
import joblib

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def load_rules(artifacts: Path, rules_arg: str | None):
    # 1) explicit path
    if rules_arg:
        rp = Path(rules_arg)
        if not rp.exists():
            raise FileNotFoundError(f"rules.json not found at {rp}")
        with rp.open("r", encoding="utf-8") as f:
            return json.load(f)
    # 2) artifacts/rules.json
    cand = artifacts / "rules.json"
    if cand.exists():
        return json.loads(cand.read_text(encoding="utf-8"))
    # 3) project root rules.json
    cand = artifacts.parent / "rules.json"
    if cand.exists():
        return json.loads(cand.read_text(encoding="utf-8"))
    raise FileNotFoundError("Cannot locate rules.json (pass --rules or put it in artifacts/ or its parent)")

def extract_entities(text: str, entity_patterns: list[dict]) -> list[str]:
    txt = normalize_text(text)
    found = []
    for p in entity_patterns:
        rx = re.compile(rf"\b{re.escape(p['pattern'])}\b", flags=re.IGNORECASE)
        for _ in rx.finditer(txt):
            found.append(f"{p['label']}:{p['pattern']}")
    # unique, keep order
    seen = set()
    uniq = []
    for x in found:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def match_rule(intent: str, entities: list[str], rules_obj: dict) -> str:
    rules = rules_obj.get("rules", [])
    defaults = rules_obj.get("defaults", {})
    ent_set = set(entities)

    def _has_entity(val):  # "LABEL:pattern"
        return val in ent_set
    def _has_all(vals):
        return all(v in ent_set for v in vals)
    def _has_any(vals):
        return any(v in ent_set for v in vals)

    for r in rules:
        cond = r.get("if", {})
        if "intent" in cond and cond["intent"] != intent:
            continue
        checks = []
        if "has_entity" in cond:
            checks.append(_has_entity(cond["has_entity"]))
        if "has_all" in cond:
            checks.append(_has_all(cond["has_all"]))
        if "has_any" in cond:
            checks.append(_has_any(cond["has_any"]))
        if all(checks) if checks else True:
            return r.get("reply", "")

    if intent in defaults:
        return defaults[intent]
    return defaults.get("_fallback", "Xin lỗi, mình chưa hiểu ý bạn.")

def respond(q: str, artifacts: Path, rules_obj: dict) -> dict:
    model = joblib.load(artifacts / "intent_model.joblib")
    with (artifacts / "intent_labels.json").open("r", encoding="utf-8") as f:
        _ = json.load(f)  # labels not strictly needed in this script
    entity_patterns = json.loads((artifacts / "entity_patterns_compiled.json").read_text(encoding="utf-8"))

    intent = model.predict([normalize_text(q)])[0]
    ents = extract_entities(q, entity_patterns)
    reply = match_rule(intent, ents, rules_obj)
    return {"intent": intent, "entities": ents, "reply": reply}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts", help="Folder containing intent_model.joblib, intent_labels.json, entity_patterns_compiled.json")
    parser.add_argument("--rules", default=None, help="Path to rules.json (optional)")
    parser.add_argument("query", nargs="+", help="User question")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    if not artifacts.exists():
        raise FileNotFoundError(f"Artifacts folder not found: {artifacts}")

    rules_obj = load_rules(artifacts, args.rules)
    q = " ".join(args.query)
    out = respond(q, artifacts, rules_obj)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
