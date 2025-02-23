import os
import cv2
import mediapipe as mp
import numpy as np
import re
import time
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class PracticeStudio:
    def __init__(self, movement_type=0, rep_duration=4, rest_duration=2):
        self.dataset_path = "dataset/SkeletonData/RawData"
        self.movement_type = movement_type
        self.rep_duration = rep_duration
        self.rest_duration = rest_duration

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.video = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.counter = 0
        self.stage = None
        self.ground_truth_angles = None

        if self.movement_type == 0:
            self.desired_joints = ["ShoulderRight", "ElbowRight", "WristRight"]
        if self.movement_type == 1:
            self.desired_joints = ["ShoulderLeft", "ElbowLeft", "WristLeft"]
        if self.movement_type == 4:
            self.desired_joints = ["ShoulderLeft", "ShoulderRight", "ElbowRight"]
        if self.movement_type == 5:
            self.desired_joints = ["ShoulderRight", "ShoulderLeft", "ElbowLeft"]

        self.load_and_process_dataset()

    def __del__(self):
        self.video.release()

    def load_and_process_dataset(self):
        """Loads the dataset, extracts angles, and computes ground truth."""
        dataset = self.load_intellirehab_dataset(self.dataset_path, self.movement_type)
        if not dataset:
            raise ValueError("Dataset is empty. Check file paths and format.")

        all_angles = self.extract_joint_angles(dataset)
        self.ground_truth_angles, _ = self.compute_ground_truth_time_series(all_angles)

    def load_intellirehab_dataset(self, folder_path, movement_type):
        """Loads and cleans only valid IntelliRehabDS CSV files for a specific movement type."""
        dataset = {}
        # print(f"Loading dataset for movement type: {movement_type}")

        for file in os.listdir(folder_path):
            parts = file.split('_')
            if len(parts) < 6 or parts[2] != str(movement_type) or parts[4][0] != '2':
                continue
            file_path = os.path.join(folder_path, file)
            joint_positions = self.clean_and_parse_intellirehab_csv(file_path)
            if len(joint_positions) < 10:
                continue
            dataset[file] = joint_positions

        # print(f"Loaded {len(dataset)} files: {list(dataset.keys())}")
        return dataset

    def clean_and_parse_intellirehab_csv(self, file_path):
        """Reads and cleans an IntelliRehabDS CSV file."""
        cleaned_lines = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or "Version0.1" in line:
                    continue
                cleaned_lines.append(line)
        return [self.parse_intellirehab_line(line) for line in cleaned_lines]

    def parse_intellirehab_line(self, line):
        """Extracts joint data from a single line."""
        parts = line.split(",", 3)
        frame_id, _, _, joint_data = parts
        joint_pattern = r"\(([^,]+),Tracked,([-.\d]+),([-.\d]+),([-.\d]+),[\d.]+,[\d.]+\)"
        matches = re.findall(joint_pattern, joint_data)
        joints = {joint[0]: (float(joint[1]), float(joint[2]), float(joint[3])) for joint in matches}
        return int(frame_id), joints

    def extract_joint_angles(self, dataset):
        """Extracts angles between relevant joints from each parsed CSV dataset."""
        all_angles = []

        for file, data in dataset.items():
            angles = []
            for frame_id, joints in data:
                if not all(j in joints for j in self.desired_joints):
                    continue

                joint1, joint2, joint3 = [joints[joint] for joint in self.desired_joints]
                angle = self.compute_angle(joint1, joint2, joint3)
                angles.append(angle)

            if angles and len(angles) > 10:
                all_angles.append(self.normalize_time_series(angles))  # Ensure non-empty lists

        if not all_angles:
            raise ValueError("No valid angles extracted from dataset. Check data formatting.")

        return all_angles

    def compute_ground_truth_time_series(self, all_angles):
        """Computes the mean and standard deviation of joint angles at each time step."""
        if not all_angles:
            raise ValueError("Error: all_angles is empty. No data to compute ground truth.")

        all_angles = np.array(all_angles)
        if all_angles.ndim != 2:
            raise ValueError(f"Expected a 2D array, but got shape {all_angles.shape}")

        return np.mean(all_angles, axis=0), np.std(all_angles, axis=0)

    def normalize_time_series(self, data, num_points=100):
        """Resamples a variable-length series to a fixed-length using interpolation."""
        print("hi", len(data))
        print(data)
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, num_points)
        interpolator = interp1d(x_old, data, kind='linear')
        return interpolator(x_new)

    def compute_angle(self, a, b, c):
        """Computes the angle between three joint points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle
    
    def normalize_angle(self, angle, min_expected=50, max_expected=120):
        """Normalizes the calculated angle (0-180) to fit within the expected range (50-120)."""
        return ((angle - 20) / (170 - 20)) * (max_expected - min_expected) + min_expected

    def extract_mediapipe_joints(self, landmarks):
        """Extracts required joint coordinates from Mediapipe."""
        if self.movement_type == 0:
            indices = [self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST]
        if self.movement_type == 1:
            indices = [self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST]
        if self.movement_type == 4:
            indices = [self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_WRIST]
        if self.movement_type == 5:
            indices = [self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_WRIST]

        joints = {}
        for name, idx in zip(self.desired_joints, indices):
            lm = landmarks.landmark[idx]
            joints[name] = (lm.x, lm.y, lm.z)

        return joints
    
    def plot_movement_trajectory(self):
        """Plots the expected movement trajectory for the selected movement type."""
        if self.ground_truth_angles is None or len(self.ground_truth_angles) == 0:
            raise ValueError("No ground truth data available for plotting.")

        time_steps = np.linspace(0, 1, len(self.ground_truth_angles))

        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, self.ground_truth_angles, label="Expected Joint Angle Progression", linewidth=2, color="b")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Joint Angle (degrees)")

        # Title based on movement type

        if self.movement_type == 0:
            movement_name = "Left Arm Curl"
        if self.movement_type == 1:
            movement_name = "Right Arm Curl"
        if self.movement_type == 4:
            movement_name = "Left Shoulder Abduction"
        if self.movement_type == 5:
            movement_name = "Right Shoulder Abduction"
            
        plt.title(f"Expected {movement_name} Trajectory")
        
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def get_frame(self):
        """Processes each frame with real-time feedback (time remaining, movement feedback, speed feedback)."""
        ret, frame = self.video.read()
        if not ret:
            return None

        elapsed_time = time.time() - self.start_time  # ✅ Now correctly initialized
        cycle_time = self.rep_duration + self.rest_duration
        cycle_progress = elapsed_time % cycle_time
        is_resting = cycle_progress >= self.rep_duration

        # Initialize variables
        results = None
        remaining_time_display = ""
        movement_feedback = ""
        speed_feedback = ""

        # Timer Display
        if is_resting:
            remaining_time_display = f"Rest... {max(0, self.rest_duration - (cycle_progress - self.rep_duration)):.1f}s"
        else:
            rep_progress = cycle_progress / self.rep_duration
            expected_angle = np.interp(rep_progress, np.linspace(0, 1, len(self.ground_truth_angles)), self.ground_truth_angles)
            remaining_time_display = f"Timer: {max(0, self.rep_duration - cycle_progress):.1f}s"
            movement_feedback = "Raise Your Arm!" if cycle_progress < self.rep_duration / 2 else "Lower Your Arm!"
            speed_feedback = ""

            # ✅ Ensure `results` is always defined
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            if results and results.pose_landmarks:
                joints = self.extract_mediapipe_joints(results.pose_landmarks)
                if all(j in joints for j in self.desired_joints):
                    raw_angle = self.compute_angle(*[joints[joint] for joint in self.desired_joints])
                    normalized_angle = self.normalize_angle(raw_angle)

                    # Speed feedback
                    if abs(normalized_angle - expected_angle) > 15:
                        speed_feedback = "Slow Down!" if raw_angle > expected_angle else "Speed Up!"
                    else:
                        speed_feedback = "Good Form!"

        # Convert the frame to RGB and overlay the feedback
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame, remaining_time_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, movement_feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, speed_feedback, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if results and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Encode the processed frame in JPEG format and return

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

# Usage
if __name__ == "__main__":
    processor = PracticeStudio(movement_type=1)
    processor.plot_movement_trajectory()
