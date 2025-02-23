import os
import cv2
import mediapipe as mp
import numpy as np
import re
import time
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def load_intellirehab_dataset(folder_path, movement_type):
    """Loads and cleans only valid IntelliRehabDS CSV files for a specific movement type."""
    dataset = {}
    print(f"Loading dataset for movement type: {movement_type}")
    
    for file in os.listdir(folder_path):
        parts = file.split('_')
        if len(parts) < 6:
            continue
        if parts[2] != str(movement_type) or parts[4][0] != '2':
            continue
        file_path = os.path.join(folder_path, file)
        dataset[file] = clean_and_parse_intellirehab_csv(file_path)
    
    print(f"Loaded {len(dataset)} files: {list(dataset.keys())}")
    return dataset


def clean_and_parse_intellirehab_csv(file_path):
    """Reads and cleans an IntelliRehabDS CSV file."""
    cleaned_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or "Version0.1" in line:
                continue
            cleaned_lines.append(line)
    parsed_data = [parse_intellirehab_line(line) for line in cleaned_lines]
    return parsed_data


def parse_intellirehab_line(line):
    """Extracts joint data from a single line."""
    parts = line.split(",", 3)
    frame_id, _, _, joint_data = parts
    joint_pattern = r"\(([^,]+),Tracked,([-.\d]+),([-.\d]+),([-.\d]+),[\d.]+,[\d.]+\)"
    matches = re.findall(joint_pattern, joint_data)
    joints = {joint[0]: (float(joint[1]), float(joint[2]), float(joint[3])) for joint in matches}
    return int(frame_id), joints


def compute_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def normalize_angle(angle, min_expected=50, max_expected=120):
    """Normalizes the calculated angle (0-180) to fit within the expected range (50-120)."""
    return ((angle - 0) / (180 - 0)) * (max_expected - min_expected) + min_expected


def extract_joint_angles(dataset):
    """Extracts angles between relevant joints from each parsed CSV dataset."""
    all_angles = []

    for file, data in dataset.items():
        angles = []
        for frame_id, joints in data:
            if not all(j in joints for j in ["ShoulderRight", "ElbowRight", "WristRight"]):
                continue  # Skip if required joints are missing
            
            shoulder, elbow, wrist = joints["ShoulderRight"], joints["ElbowRight"], joints["WristRight"]
            angle = compute_angle(shoulder, elbow, wrist)
            angles.append(angle)
        
        if angles:
            all_angles.append(normalize_time_series(angles))  # Ensure non-empty lists
    
    if not all_angles:
        raise ValueError("No valid angles extracted from dataset. Check data formatting.")

    return all_angles


def normalize_time_series(data, num_points=100):
    """Resamples a variable-length series to a fixed-length using interpolation."""
    x_old = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, num_points)
    interpolator = interp1d(x_old, data, kind='linear')
    return interpolator(x_new)


def compute_ground_truth_time_series(all_angles):
    """
    Computes the mean and standard deviation of joint angles at each time step.
    Ensures the input is a valid array before performing computations.
    """
    if not isinstance(all_angles, list) or len(all_angles) == 0:
        raise ValueError("Error: all_angles is empty. No data to compute ground truth.")

    all_angles = np.array(all_angles)  # Shape: (num_samples, num_time_steps)
    
    if all_angles.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {all_angles.shape}")

    mean_angles = np.mean(all_angles, axis=0)  # Average at each time step
    std_angles = np.std(all_angles, axis=0)    # Standard deviation at each time step

    if mean_angles.ndim == 0:  # If mean_angles is a scalar instead of an array
        raise ValueError("Error: computed ground truth is a scalar. Expected an array.")

    return mean_angles, std_angles


def extract_mediapipe_joints(landmarks):
    """Extracts required joint coordinates from Mediapipe."""
    joint_names = ["ShoulderLeft", "ElbowLeft", "WristLeft"]
    indices = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]
    
    joints = {}
    for name, idx in zip(joint_names, indices):
        lm = landmarks.landmark[idx]
        joints[name] = (lm.x, lm.y, lm.z)
    
    return joints


def compare_to_expected(angle, expected_angle):
    """Provides real-time feedback on movement accuracy."""
    
    if abs(angle - expected_angle) < 10:
        return "Good Form!"
    elif angle > expected_angle:
        return "Too Extended!"
    else:
        return "Speed Up!"


def process_live_video(ground_truth_angles, duration=5):
    """Processes live video, calculates angles, normalizes them, and provides feedback based on the expected trajectory."""
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    start_time = time.time()
    prev_expected_angle = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        progress = (elapsed_time % duration) / duration  # Normalize progress (0 to 1)
        expected_angle = np.interp(progress, np.linspace(0, 1, len(ground_truth_angles)), ground_truth_angles)

        if prev_expected_angle is not None:
            angle_trend = "lower" if expected_angle > prev_expected_angle else "raise"
        else:
            angle_trend = ""

        prev_expected_angle = expected_angle

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            joints = extract_mediapipe_joints(results.pose_landmarks)

            if all(j in joints for j in ["ShoulderLeft", "ElbowLeft", "WristLeft"]):
                raw_angle = compute_angle(joints["ShoulderLeft"], joints["ElbowLeft"], joints["WristLeft"])
                normalized_angle = normalize_angle(raw_angle)

                feedback = compare_to_expected(normalized_angle, expected_angle)

                # Display the computed angle
                cv2.putText(frame, f"Angle: {normalized_angle:.2f}°", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Display feedback
                cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display expected angle trend
                cv2.putText(frame, f"Expected: {expected_angle:.2f}°  {angle_trend}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        cv2.imshow("Live Feedback", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def plot_ground_truth(ground_truth):
    """Plots the averaged time series ground truth."""
    time_steps = np.linspace(0, 1, len(ground_truth))  # Normalize time axis

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, ground_truth, label="Averaged Joint Angle Progression", linewidth=2)
    plt.xlabel("Normalized Time (0 to 1)")
    plt.ylabel("Joint Angle (degrees)")
    plt.title("Averaged Time Series Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    dataset = load_intellirehab_dataset("dataset/SkeletonData/RawData", movement_type=0)
    
    print(f"Dataset loaded: {len(dataset)} files")  # Debugging output
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check file paths and format.")

    all_angles = extract_joint_angles(dataset)  # Normalized time series of angles

    print(f"Extracted {len(all_angles)} time series of angles")  # Debugging output

    ground_truth, _ = compute_ground_truth_time_series(all_angles)

    # plot_ground_truth(ground_truth)

    print(f"Ground truth computed. Shape: {ground_truth.shape}")  # Debugging output

    process_live_video(ground_truth)