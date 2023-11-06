import mediapipe as mp
import cv2 as cv

class Hand:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands()
        self.draw = mp.solutions.drawing_utils

    def detect_hands(self, img):
        rgbimg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.res = self.hands.process(rgbimg)
        multihand = self.res.multi_hand_landmarks
        if multihand:
            for hand in multihand:
                self.draw.draw_landmarks(img, hand,  self.mphands.HAND_CONNECTIONS)
        return img

    def give_hand_point(self, img, handNo=0, draw=True):
        lm_list = []
        rgbimg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.res = self.hands.process(rgbimg)
        if self.res.multi_hand_landmarks:
            hand  = self.res.multi_hand_landmarks[handNo]
            for id, location_point in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(location_point.x * w), int(location_point.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 3, (225, 4, 1), cv.FILLED)
        return lm_list
