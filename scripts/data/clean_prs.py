import json
import re


# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "C:\Users\g4u4q\PycharmProjects\AIVoiceAgentSell\.venv\Scripts\data\datsets\original_version\dataSale_conv.jsonl"
OUTPUT_FILE = "clean_dataSale_conv.jsonl"

MIN_MSG_LENGTH = 3
REMOVE_DUPLICATES = True
MAX_MESSAGES = 30

BOILERPLATE_PATTERNS = [
    r"thank you for contacting .*",
    r"how can i help you today\??",
    r"please let me know if you need anything else",
]


# -----------------------------
# clean message
# -----------------------------
def clean_message(text):
    if not text:
        return None

    text = text.strip()
    text = text.strip('"')
    text = re.sub(r"\s+", " ", text)

    if len(text) < MIN_MSG_LENGTH:
        return None

    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, text.lower()):
            return None

    return text


# -----------------------------
# dedupe
# -----------------------------
def deduplicate(messages):
    seen = set()
    out = []

    for m in messages:
        key = (m["role"], m["content"])
        if key in seen:
            continue
        seen.add(key)
        out.append(m)

    return out


# -----------------------------
# fix role flow
# -----------------------------
def fix_flow(messages):
    fixed = []
    last_role = None

    for m in messages:
        role = m.get("role")
        content = clean_message(m.get("content"))

        if not content:
            continue

        if role == last_role:
            continue

        fixed.append({
            "role": role,
            "content": content
        })

        last_role = role

    return fixed


# -----------------------------
# clean dataset item
# -----------------------------
def clean_item(item):
    messages = item.get("messages", [])

    messages = fix_flow(messages)

    if REMOVE_DUPLICATES:
        messages = deduplicate(messages)

    if MAX_MESSAGES:
        messages = messages[:MAX_MESSAGES]

    if len(messages) < 2:
        return None

    return {
        "id": item.get("id"),
        "messages": messages
    }


# -----------------------------
# main pipeline
# -----------------------------
def run():
    cleaned = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            new_item = clean_item(item)
            if new_item:
                cleaned.append(new_item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item) + "\n")

    print(f"Input loaded: {INPUT_FILE}")
    print(f"Output saved: {OUTPUT_FILE}")
    print(f"Clean samples: {len(cleaned)}")


if __name__ == "__main__":
    run()