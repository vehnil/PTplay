import os
import pandas as pd
import re

def load_intellirehab_dataset(folder_path):
    """Loads all IntelliRehabDS CSV files, cleans them, and extracts joint data."""
    dataset = {}

    for file in os.listdir(folder_path):
        if file.endswith(".csv") or file.endswith(".txt"):  # Adjust if needed
            file_path = os.path.join(folder_path, file)
            dataset[file] = clean_and_parse_intellirehab_csv(file_path)

    return dataset

def clean_and_parse_intellirehab_csv(file_path):
    """Reads an IntelliRehabDS CSV file, removes unwanted lines, and extracts joint data."""
    cleaned_lines = []
    
    with open(file_path, 'r') as f:
        # print(file_path)
        for line in f:
            line = line.strip()  # Remove leading/trailing spaces
            if not line or "Version0.1" in line:  # Skip empty lines and unwanted metadata
                continue
            cleaned_lines.append(line)

    parsed_data = []
    for line in cleaned_lines:
        frame_id, joints = parse_intellirehab_line(line)
        parsed_data.append((frame_id, joints))

    return parsed_data


def parse_intellirehab_line(line):
    """Extracts joint data from a single line of IntelliRehabDS dataset."""
    # print(line)
    parts = line.strip().split(",", 3)
    frame_id, joint_id, timestamp, joint_data = parts

    # Regex to extract (JointName, State, X, Y, Z, u, v)
    joint_pattern = r"\(([^,]+),Tracked,([-\d\.]+),([-\d\.]+),([-\d\.]+),([\d\.]+),([\d\.]+)\)"
    matches = re.findall(joint_pattern, joint_data)

    joints = {joint[0]: (float(joint[1]), float(joint[2]), float(joint[3])) for joint in matches}
    return int(frame_id), joints

# Load dataset
dataset_folder = "dataset/SkeletonData/RawData/"
intellirehab_data = load_intellirehab_dataset(dataset_folder)

print("Loaded", len(intellirehab_data), "files.")
