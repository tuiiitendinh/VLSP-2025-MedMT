import json
import os
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_validation_data():
    """Create validation data from training data for evaluation."""
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    train_file = os.path.join(processed_dir, "train.jsonl")
    val_file = os.path.join(processed_dir, "val.jsonl")

    if not os.path.exists(train_file):
        print(f"Training file {train_file} not found. Please run prepare_data.py first.")
        return

    # Read all training samples
    all_samples = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_samples.append(json.loads(line.strip()))

    # Shuffle and split
    random.shuffle(all_samples)
    split_point = int(len(all_samples) * 0.9)

    train_samples = all_samples[:split_point]
    val_samples = all_samples[split_point:]

    # Write training data
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Write validation data
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Created validation split:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")

    # Show expert distribution in validation set
    expert_counts = {}
    for sample in val_samples:
        expert = sample.get('expert', 'unknown')
        expert_counts[expert] = expert_counts.get(expert, 0) + 1

    print(f"  Expert distribution in validation set:")
    for expert, count in expert_counts.items():
        print(f"    {expert}: {count}")

def main():
    create_validation_data()

if __name__ == "__main__":
    main()
