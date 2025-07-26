import csv
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    csv_path = os.path.join(PROJECT_ROOT, "train_data", "train_data.csv")
    output_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    en_vi_jsonl = os.path.join(output_dir, "en_vi_train.jsonl")
    vi_en_jsonl = os.path.join(output_dir, "vi_en_train.jsonl")
    combined_jsonl = os.path.join(output_dir, "train.jsonl")

    en_vi_samples = []
    vi_en_samples = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = row.get("english", "").strip()
            vi = row.get("vietnamese", "").strip()
            if not en or not vi:
                continue

            # English to Vietnamese
            en_vi_samples.append({
                "messages": [
                    {"role": "user", "content": f"Translate the following English sentence to Vietnamese: {en}"},
                    {"role": "assistant", "content": vi}
                ],
                "expert": "en_vi"
            })

            # Vietnamese to English
            vi_en_samples.append({
                "messages": [
                    {"role": "user", "content": f"Translate the following Vietnamese sentence to English: {vi}"},
                    {"role": "assistant", "content": en}
                ],
                "expert": "vi_en"
            })

    # Write each expert file
    for samples, out_path in [
        (en_vi_samples, en_vi_jsonl),
        (vi_en_samples, vi_en_jsonl)
    ]:
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Write combined file
    with open(combined_jsonl, "w", encoding="utf-8") as f:
        for sample in en_vi_samples + vi_en_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ Converted CSV to JSONL for MoE training.")
    print(f"EN-VI samples: {len(en_vi_samples)}")
    print(f"VI-EN samples: {len(vi_en_samples)}")

if __name__ == "__main__":
    main()