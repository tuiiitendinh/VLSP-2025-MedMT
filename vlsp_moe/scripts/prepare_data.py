import json
import os
import random
import re
import shutil
import sys
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def is_medical_text(text):
    """Simple heuristic to identify medical texts."""
    medical_keywords = [
        "patient",
        "disease",
        "treatment",
        "diagnosis",
        "symptom",
        "medicine",
        "doctor",
        "hospital",
        "medical",
        "clinical",
        "therapy",
        "surgery",
        "drug",
        "medication",
        "health",
        "illness",
        "condition",
        "infection",
        "virus",
        "bacteria",
        "cancer",
        "tumor",
        "blood",
        "heart",
        "lung",
        "brain",
        "liver",
        "kidney",
        "bone",
        "muscle",
        "skin",
        "eye",
        "bệnh",
        "thuốc",
        "bác sĩ",
        "bệnh viện",
        "điều trị",
        "chẩn đoán",
        "triệu chứng",
        "y tế",
        "sức khỏe",
        "nhiễm trùng",
        "virus",
        "vi khuẩn",
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in medical_keywords)


def prepare_expert_data(
    expert_name, src_texts, tgt_texts, output_file, task_description
):
    """Prepare data for a specific expert."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as outf:
        for src, tgt in zip(src_texts, tgt_texts):
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{task_description}: {src.strip()}",
                    },
                    {"role": "assistant", "content": tgt.strip()},
                ],
                "expert": expert_name,  
            }
            outf.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_parallel_data(en_file, vi_file):
    """Load parallel data from files."""
    with open(en_file, "r", encoding="utf-8") as enf, open(
        vi_file, "r", encoding="utf-8"
    ) as vif:
        en_lines = [line.strip() for line in enf if line.strip()]
        vi_lines = [line.strip() for line in vif if line.strip()]

    # Ensure equal length
    min_length = min(len(en_lines), len(vi_lines))
    return en_lines[:min_length], vi_lines[:min_length]


def split_data_by_experts(en_texts, vi_texts):
    """Split data into different expert categories."""
    medical_en, medical_vi = [], []
    general_en, general_vi = [], []

    for en, vi in zip(en_texts, vi_texts):
        if is_medical_text(en) or is_medical_text(vi):
            medical_en.append(en)
            medical_vi.append(vi)
        else:
            general_en.append(en)
            general_vi.append(vi)

    return (medical_en, medical_vi), (general_en, general_vi)


def main():
    """Main function to prepare MoE training data."""
    print("Loading parallel data...")

    # Load your parallel data
    en_file = os.path.join(PROJECT_ROOT, "data", "train.en.txt")
    vi_file = os.path.join(PROJECT_ROOT, "data", "train.vi.txt")

    if not os.path.exists(en_file) or not os.path.exists(vi_file):
        print(f"Error: Could not find {en_file} or {vi_file}")
        print("Please ensure your training files are in the correct location.")
        return

    en_texts, vi_texts = load_parallel_data(en_file, vi_file)
    print(f"Loaded {len(en_texts)} parallel sentences")

    # Split data into medical and general
    (medical_en, medical_vi), (general_en, general_vi) = split_data_by_experts(
        en_texts, vi_texts
    )

    print(f"Medical domain: {len(medical_en)} sentences")
    print(f"General domain: {len(general_en)} sentences")

    # If no medical data found, create a subset as medical domain
    if len(medical_en) == 0:
        print("No medical keywords found. Using first 20% of data as medical domain.")
        split_idx = len(general_en) // 5
        medical_en = general_en[:split_idx]
        medical_vi = general_vi[:split_idx]
        general_en = general_en[split_idx:]
        general_vi = general_vi[split_idx:]

    # Split general data for EN->VI and VI->EN experts
    general_split = len(general_en) // 2
    en_vi_en = general_en[:general_split]
    en_vi_vi = general_vi[:general_split]
    vi_en_en = general_en[general_split:]
    vi_en_vi = general_vi[general_split:]

    print(f"EN->VI expert: {len(en_vi_en)} sentences")
    print(f"VI->EN expert: {len(vi_en_en)} sentences")

    # Prepare data for Medical Domain Expert (EN->VI)
    print("Preparing medical domain expert data...")
    prepare_expert_data(
        expert_name="medical",
        src_texts=medical_en,
        tgt_texts=medical_vi,
        output_file=os.path.join(PROJECT_ROOT, "data", "processed", "medical_train.jsonl"),
        task_description="Translate the following English medical text to Vietnamese",
    )

    # Prepare data for EN->VI Translation Expert
    print("Preparing EN->VI expert data...")
    prepare_expert_data(
        expert_name="en_vi",
        src_texts=en_vi_en,
        tgt_texts=en_vi_vi,
        output_file=os.path.join(PROJECT_ROOT, "data", "processed", "en_vi_train.jsonl"),
        task_description="Translate the following English sentence to Vietnamese",
    )

    # Prepare data for VI->EN Translation Expert
    print("Preparing VI->EN expert data...")
    prepare_expert_data(
        expert_name="vi_en",
        src_texts=vi_en_vi,  # Vietnamese as source
        tgt_texts=vi_en_en,  # English as target
        output_file=os.path.join(PROJECT_ROOT, "data", "processed", "vi_en_train.jsonl"),
        task_description="Translate the following Vietnamese sentence to English",
    )

    # Combine all expert data into a single training file
    print("Combining all expert data...")
    combined_output = os.path.join(PROJECT_ROOT, "data", "processed", "train.jsonl")
    total_samples = 0

    for expert_file in [
        os.path.join(PROJECT_ROOT, "data", "processed", "medical_train.jsonl"),
        os.path.join(PROJECT_ROOT, "data", "processed", "en_vi_train.jsonl"),
        os.path.join(PROJECT_ROOT, "data", "processed", "vi_en_train.jsonl"),
    ]:
        if os.path.exists(expert_file):
            with open(expert_file, "r", encoding="utf-8") as inf:
                with open(combined_output, "a", encoding="utf-8") as outf:
                    for line in inf:
                        outf.write(line)
                        total_samples += 1

    print(f"\nData preparation completed!")
    print(f"Total training samples: {total_samples}")
    print(f"Files created:")
    print(f"  - Medical expert: {os.path.join(PROJECT_ROOT, 'data', 'processed', 'medical_train.jsonl')}")
    print(f"  - EN->VI expert: {os.path.join(PROJECT_ROOT, 'data', 'processed', 'en_vi_train.jsonl')}")
    print(f"  - VI->EN expert: {os.path.join(PROJECT_ROOT, 'data', 'processed', 'vi_en_train.jsonl')}")
    print(f"  - Combined: {os.path.join(PROJECT_ROOT, 'data', 'processed', 'train.jsonl')}")


if __name__ == "__main__":
    # Error handling for disk quota exceeded
    try:
        main()
    except OSError as e:
        if e.errno == 28:  # No space left on device
            print("Error: Disk quota exceeded. Please free up some space and try again.")
        else:
            print(f"Error: {e.strerror}")
            print("Unexpected error occurred. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Unexpected error occurred. Exiting.")
        sys.exit(1)

    # Redirect output to a temporary directory if needed
    temp_dir = tempfile.gettempdir()
    shutil.copy(os.path.join(PROJECT_ROOT, "data", "processed", "train.jsonl"), temp_dir)
    print(f"Output redirected to temporary directory: {temp_dir}")
