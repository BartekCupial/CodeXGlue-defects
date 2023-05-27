import re

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


from tqdm.auto import tqdm


def extract_functions(file_string, function_name, line_numbers):
    functions = []
    bugs = []
    # Split the file string into lines
    lines = file_string.split('\n')

    # Remove comments from the lines
    lines = [re.sub(r'#.*', '', line) for line in lines]

    # Find function definitions
    for i, line in enumerate(lines):
        if line.strip().startswith("def " + function_name):
            bugs.append(0)
            bugs[-1] |= i in line_numbers
            function_body = [line]

            # Check if it's a single-line function
            if line.strip().endswith(":"):
                indent = len(line) - len(line.lstrip())
                for j in range(i + 1, len(lines)):
                    bugs[-1] |= j in line_numbers
                    current_line = lines[j]
                    current_indent = len(current_line) - len(current_line.lstrip())
                    if current_indent <= indent and current_line.strip() != "":
                        break
                    function_body.append(current_line)
                functions.append('\n'.join(function_body))
            else:
                # Single-line function without indented block
                functions.append(line)

    # remove docstrings from functions
    functions = [re.sub(r'""".*?"""', '', function, flags=re.DOTALL) for function in functions]

    return functions, bugs


dataset_dir = Path("defectors/line_bug_prediction_splits/random")
output_dir = Path("python")
output_dir.mkdir(parents=True, exist_ok=True)

for dataset_split in ["train", "val", "test"]:
    df = pq.read_table(dataset_dir / f"{dataset_split}.parquet.gzip").to_pandas()

    new_df = []
    for e, row in enumerate(tqdm(df.to_dict(orient="records"))):
        try:
            content = row["content"]
            file = content.decode("UTF-8")
        except Exception as e:
            continue
        methods = row["methods"]
        line_numbers = row["lines"]
        
        for function_name in methods:
            functions, bugs = extract_functions(file, function_name, line_numbers)

            for function, bug in zip(functions, bugs):
                new_row = {}
                new_row["func (string)"] = function
                new_row["target"] = bug
                new_row["commit"] = row["commit"]
                new_row["repo"] = row["repo"]
                new_row["filepath"] = row["filepath"]
                new_df.append(new_row)

    new_df = pd.DataFrame(new_df)
    new_df.to_csv(output_dir / f"{dataset_split}.csv", index=False)
