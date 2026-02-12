import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch



class DataEngine:
    def __init__(self,device='cpu'):
        self.detector = MTCNN(select_largest=True, post_process=False,device=device)
        self.device = device

    def extract_frames(self,video_path, sample_rate=.5):
        frames = []
        cap = cv2.VideoCapture(video_path)
        #---error-handling---
        if not cap.isOpened():
            print(f"Error opening video stream or file {video_path} isn't open")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_step = int(fps * sample_rate)

        for frame_idx in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        return frames

    def align_face(self, frame, landmarks):
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']

        #---angle-between-eyes---
        dy = right_eye[0] - left_eye[0]
        dx = left_eye[1] - right_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        #--rotation-matrix---
        eye_center = (
                    (int(left_eye[0] + right_eye[0]) // 2),
                    (int(left_eye[1] + right_eye[1]) // 2))
        height, width = frame.shape[:2]
        Matrix = cv2.getRotationMatrix2D(eye_center, angle, scale =1)
        rotated_frame = cv2.warpAffine(frame, Matrix, (width, height))
        #---detect-face-on-aligned-frame---
        boxes,_ = self.detector.detect_face(rotated_frame)
        if boxes is not None:
            x1, y1, x2, y2 = boxes[0].astype(int)
            #---still-in-bounds-?---
            face_crop= rotated_frame[max(0, y1):y2, max(0, x1):x2]
            face_crop = cv2.resize(face_crop, (244, 244))
            return face_crop

        return None


    def process_video(self,video_path):
        pocessed_frames = []
        #---extract-frames---
        extracted_frames = self.extract_frames(video_path)
        if not extracted_frames:
            return []

        for frame in extracted_frames:
            boxes, probs,landmarks = self.detector.detect_face(frame,landmarks=True)
            if boxes is not None and probs[0]> .90:
                face_landmarks = landmarks[0]
                cropped_image = self.align_face(frame, face_landmarks)
                if cropped_image is not None:
                    pocessed_frames.append(cropped_image)
        return pocessed_frames


