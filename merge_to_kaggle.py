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


def execute_monolith_compilation(entrypoint_filepath, output_filename):
    compiled_code_lines = []

    for filepath in MODULE_FILES_ORDER + [entrypoint_filepath]:
        if not os.path.exists(filepath):
            continue

        with open(filepath, "r") as file_handle:
            source_code_string = file_handle.read()

        source_code_string = re.sub(
            pattern=r"^(from\s+src[\.\w]*\s+import\s+.*|import\s+src[\.\w]*.*)$",
            repl="",
            string=source_code_string,
            flags=re.MULTILINE,
        )

        compiled_code_lines.append(f"\n# {'='*50}\n# ORIGINAL FILE: {filepath}\n# {'='*50}\n")
        compiled_code_lines.append(source_code_string.strip())
        compiled_code_lines.append("\n")

    with open(output_filename, "w") as file_handle:
        file_handle.write("\n".join(compiled_code_lines))

    print(f"Successfully compiled {output_filename}")


if __name__ == "__main__":
    execute_monolith_compilation("main.py", "kaggle_main.py")
    execute_monolith_compilation("tools.py", "kaggle_tools.py")
