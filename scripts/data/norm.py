import json
import re

INPUT_FILE = "C:\Users\g4u4q\PycharmProjects\AIVoiceAgentSell\.venv\Scripts\data\datsets\original_version\dataSale_conv.jsonl"
OUTPUT_FILE = "normalized_dataSale_conv.jsonl"


def extract_messages(text):
    text = str(text)

    # find (role, content) pairs even if messy
    pattern = r'(user|assistant)\s*["\']?\s*,?\s*content["\']?\s*:\s*"(.*?)"'
    return [{"role": r, "content": c} for r, c in re.findall(pattern, text, re.I)]


def run():
    out = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except:
                continue

            raw = json.dumps(item)
            msgs = extract_messages(raw)

            if len(msgs) >= 2:
                out.append({"id": item.get("id"), "messages": msgs})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")

    print("Normalized:", len(out))


if __name__ == "__main__":
    run()