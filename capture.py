import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class Capture:
    def __init__(self, mode: int):
        self.video = cv2.VideoCapture(0)

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.mode = mode

        self.counter = 0
        self.stage = None
        self.angles = []

    def calculate_angle(self, a, b):
        delta_x = b[0] - a[0]
        delta_y = b[1] - a[1]
        angle = np.arctan2(delta_y, delta_x) * (180.0 / np.pi)
        return angle if angle >= 0 else angle + 360

    def __del__(self):
        self.video.release()

    def bicep_curls(self, angle: float):
        # Curl counter logic for left arm (example)
        if angle > 160:
            self.stage = "down"
        if angle < 30 and self.stage == 'down':
            self.stage = "up"
            self.counter += 1
            print("Rep Count:", self.counter)

    def bicep_curls_r(self, angle_r: float):
        # Curl counter logic for right arm (example)
        if angle_r > 160:
            self.stage = "down"
        if angle_r < 30 and self.stage == 'down':
            self.stage = "up"
            self.counter += 1
            print("Rep Count:", self.counter)

    def lat_raises(self, avg_shoulder: float, avg_wrist: float):
        # Lat raise counter logic for left arm (example)
        if avg_wrist > avg_shoulder + 0.1:  # Arms Down (0.1 is buffer)
                self.stage = "down"
        if avg_wrist < avg_shoulder - 0.2 and self.stage == "down":  # Arms Up
            self.stage = "up"
            self.counter += 1
            print("Rep Count:", self.counter)

    def band_pulls(self, wrist_distance: float, left_elbow_angle: float, right_elbow_angle: float):
        # band pull counter logic for left arm (example)
        if wrist_distance < 0.2:
                self.stage = "together"
        if wrist_distance > 0.4 and self.stage == "together":
            self.stage = "apart"
            self.counter += 1
            print("Rep Count:", self.counter)

    def hooks(self):
        # Hook counter logic for left arm (example)
        if len(self.angles) > 20:
            angle_range = max(self.angles) - min(self.angles)

            # Starting hook
            if angle_range > 90 and self.stage != 'in_progress':
                self.stage = 'in_progress'

            # Complete hook (full circle)
            if angle_range > 300 and self.stage == 'in_progress':
                self.counter += 1
                self.stage = 'complete'
                self.angles = []  # Reset for next rep

            # Reset stage when arm returns
            if angle_range < 50 and self.stage == 'complete':
                self.stage = 'ready'
        
    def slices(self, angle_l, angle_r):
        if angle_l < 30:
            self.l_stage = 'up'  # Hand moved up
        
        if angle_r < 30:
            self.r_stage = 'up'  # Hand moved up

        if angle_l > 130 and self.l_stage == 'up':
            self.l_stage = 'down'
            self.counter += 1
            
        if angle_r > 130 and self.r_stage == 'up':
            self.r_stage = 'down'
            self.counter += 1  # Count when going from up → down

    def detect_punch(self, angle: float, angle_r: float, wrist: float, wrist_r: float, elbow: float, elbow_r: float):
        # Check if either arm is extended forward (angle < 30)
        left_arm_extended = angle > 150 and wrist < elbow  # Left arm extended and wrist y > elbow y
        right_arm_extended = angle_r > 150 and wrist_r < elbow_r  # Right arm extended and wrist y > elbow y

        # Detect punch only when moving from 'down' to 'punch'
        if self.stage == 'rest' and (left_arm_extended or right_arm_extended):
            self.stage = 'punch'
            self.counter += 1
            print("Punch Detected! Rep Count:", self.counter)

        # Reset to 'down' when both arms are no longer extended
        elif not (left_arm_extended or right_arm_extended):
            self.stage = 'rest'

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.flip(image, 1)

        try:
            landmarks = results.pose_landmarks.landmark

            # Extract coordinates for left arm
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Extract coordinates for right arm
            shoulder_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles for left and right arms
            angle = calculate_angle(shoulder, elbow, wrist)
            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)

            avg_shoulder_y = (shoulder[1] + shoulder_r[1]) / 2
            avg_wrist_y = (wrist[1] + wrist_r[1]) / 2

            wrist_distance = np.sqrt(
                (wrist[0] - wrist_r[0]) ** 2 +
                (wrist[1] - wrist_r[1]) ** 2
            )

            shoulder_wrist_angle = self.calculate_angle(shoulder, wrist)
            self.angles.append(shoulder_wrist_angle)

            wrist_px_l = tuple(np.multiply(wrist, [640, 480]).astype(int))
            wrist_px_r = tuple(np.multiply(wrist_r, [640, 480]).astype(int))

            # Visualize the calculated angles on the frame
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle_r)),
                        tuple(np.multiply(elbow_r, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if self.mode == 0:
                self.bicep_curls(angle)
            elif self.mode == 1:
                self.lat_raises(avg_shoulder_y, avg_wrist_y)
            elif self.mode == 2:
                self.band_pulls(wrist_distance, angle, angle_r)
            elif self.mode == 3:
                self.hooks()
            elif self.mode == 4:
                self.slices(angle, angle_r)
            elif self.mode == 5:
                self.bicep_curls_r(angle_r)
            elif self.mode == 6:
                self.detect_punch(angle, angle_r, wrist[1], wrist_r[1], elbow[1], elbow_r[1])

        except Exception as e:
            # In case landmarks are not detected or any error occurs
            pass

        # Render counter and stage information on the frame
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, self.stage if self.stage is not None else "",
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw pose landmarks if available
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                           self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Encode the processed frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
        