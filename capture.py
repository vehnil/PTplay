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
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.counter = 0
        self.stage = None

    def __del__(self):
        self.video.release()

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

            # Visualize the calculated angles on the frame
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle_r)),
                        tuple(np.multiply(elbow_r, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic for left arm (example)
            if angle > 160:
                self.stage = "down"
            if angle < 30 and self.stage == 'down':
                self.stage = "up"
                self.counter += 1
                print("Rep Count:", self.counter)

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
        