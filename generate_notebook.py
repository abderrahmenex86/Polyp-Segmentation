import json
import os
import re

MODULE_FILES_ORDER = [
    "src/helpers.py",
    "src/dataset.py",
    "src/models.py",
    "src/factory.py",
    "src/profile.py",
    "src/trainer.py",
    "src/infer.py",
]


def build_monolith(entrypoint):
    code_lines = []
    for filepath in MODULE_FILES_ORDER + [entrypoint]:
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r") as f:
            text = f.read()

        pattern = r"(?:from\s+src[\.\w]*\s+import\s+(?:\([^\)]+\)|[^\n]+)|import\s+src[\.\w]*.*?)(?:\n|$)"
        text = re.sub(pattern, "", text)

        code_lines.append(f"\n# {'='*50}\n# ORIGINAL FILE: {filepath}\n# {'='*50}\n")
        code_lines.append(text.strip() + "\n")
    return "".join(code_lines)


def create_cell(source_text, cell_type="code"):
    lines = [line + "\n" for line in source_text.split("\n")]
    if lines and lines[-1] == "\n":
        lines.pop()
    cell = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def generate_ipynb():
    main_monolith = "%%writefile kaggle_main.py\n" + build_monolith("main.py")

    mapper_code = """
import os
import glob
import cv2
import shutil

input_dir = "/kaggle/input"
target_dir = "/kaggle/working/dataset"

# Purge any old directory containing stale .tif or .jpg files
shutil.rmtree(target_dir, ignore_errors=True)

os.makedirs(f"{target_dir}/images", exist_ok=True)
os.makedirs(f"{target_dir}/masks", exist_ok=True)

image_variants = ["images", "Original", "CVC-300/images", "CVC-ColonDB/images", "ETIS-LaribPolypDB/images", "PNG/Original"]
mask_variants = ["masks", "Ground Truth", "CVC-300/masks", "CVC-ColonDB/masks", "ETIS-LaribPolypDB/masks", "PNG/Ground Truth"]

print("Scanning mounted Kaggle datasets...")
for dataset_name in os.listdir(input_dir):
    dataset_path = os.path.join(input_dir, dataset_name)
    if not os.path.isdir(dataset_path): continue
    
    img_dir, msk_dir = None, None
    for root, dirs, files in os.walk(dataset_path):
        folder_name = os.path.basename(root)
        if folder_name in [os.path.basename(v) for v in image_variants] and not img_dir: img_dir = root
        if folder_name in [os.path.basename(v) for v in mask_variants] and not msk_dir: msk_dir = root
            
    if img_dir and msk_dir:
        print(f"Standardizing {dataset_name} to lossless PNG format...")
        img_files = sorted(glob.glob(f"{img_dir}/*.*"))
        msk_files = sorted(glob.glob(f"{msk_dir}/*.*"))
        
        for img_path, msk_path in zip(img_files, msk_files):
            img = cv2.imread(img_path)
            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None and msk is not None:
                base_img_name = os.path.splitext(os.path.basename(img_path))[0]
                base_msk_name = os.path.splitext(os.path.basename(msk_path))[0]
                
                new_img_path = os.path.join(target_dir, "images", f"{dataset_name}_{base_img_name}.png")
                new_msk_path = os.path.join(target_dir, "masks", f"{dataset_name}_{base_msk_name}.png")
                
                cv2.imwrite(new_img_path, img)
                cv2.imwrite(new_msk_path, msk)
    else:
        print(f"Skipped {dataset_name}: Could not resolve image/mask structure.")
print(f"Dataset consolidation complete. Files stored in {target_dir}")
""".strip()

    cells = [
        create_cell(
            "# 1. Ex86 Endoscopy Segmentation - Kaggle Environment\n\n**INSTRUCTIONS:**\n1. In the top right corner of Kaggle, click **Add Data**.\n2. Search for and add: `kvasir-seg`, `cvcclinicdb`, `cvc-colondb`, `etis-laribpolypdb`, `cvc-300`.\n3. Make sure **Accelerator** (GPU) is turned on in Notebook Settings.\n4. Run the cells below sequentially.",
            "markdown",
        ),
        create_cell("!pip install monai torchinfo"),
        create_cell(main_monolith),
        create_cell(mapper_code),
        create_cell("!python kaggle_main.py --profile --architecture PraNet --image_height 352 --image_width 352"),
        create_cell(
            "!python kaggle_main.py --train --architecture PraNet --batch_size 16 --epochs 100 --warmup_epochs 10 --dataset_directory /kaggle/working/dataset"
        ),
    ]

    notebook = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 4}

    with open("Ex86_Kaggle_Environment.ipynb", "w") as f:
        json.dump(notebook, f, indent=2)
    print("Successfully generated Ex86_Kaggle_Environment.ipynb")


if __name__ == "__main__":
    generate_ipynb()
