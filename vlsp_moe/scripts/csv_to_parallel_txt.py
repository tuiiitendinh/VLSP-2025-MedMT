import csv
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def main():
    csv_path = os.path.join(PROJECT_ROOT, 'train_data', 'train_data.csv')
    output_dir = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(output_dir, exist_ok=True)

    en_txt_path = os.path.join(output_dir, 'train.en.txt')
    vi_txt_path = os.path.join(output_dir, 'train.vi.txt')

    with open(csv_path, 'r', encoding='utf-8') as csvfile, \
         open(en_txt_path, 'w', encoding='utf-8') as enf, \
         open(vi_txt_path, 'w', encoding='utf-8') as vif:
        reader = csv.DictReader(csvfile)
        for row in reader:
            en = row.get('english', '').strip()
            vi = row.get('vietnamese', '').strip()
            if en and vi:
                enf.write(en + '\n')
                vif.write(vi + '\n')
    print(f"Wrote parallel files: {en_txt_path}, {vi_txt_path}")

if __name__ == "__main__":
    main() 