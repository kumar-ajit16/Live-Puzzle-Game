import cv2
import numpy as np
import mediapipe as mp
from puzzle_ui import PuzzleUI


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def detect_hands_mediapipe(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_landmarks = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for lm in handLms.landmark:
                h, w, _ = frame.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))
            hand_landmarks.append(lm_list)

    return hand_landmarks


def main():
    cap = cv2.VideoCapture(0)
    captured = False
    puzzle_ui = None
    grid_size = 3

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        h, w = raw_frame.shape[:2]
        dark_bg = np.zeros_like(raw_frame)
        min_dim = min(h, w)
        cam_img = cv2.resize(raw_frame, (min_dim, min_dim))
        dark_bg[(h-min_dim)//2:(h+min_dim)//2,
                (w-min_dim)//2:(w+min_dim)//2] = cam_img

        frame = dark_bg.copy()

        if not captured:

            cv2.putText(frame, "LIVE PUZZLE",
                        (w//2 - 100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (57, 255, 20),
                        2)

            hand_landmarks = detect_hands_mediapipe(frame)

            for lm_list in hand_landmarks:
                for (x, y) in lm_list:
                    cv2.circle(frame, (x, y), 3, (57, 255, 20), -1)

            rect_pts = None

            if len(hand_landmarks) == 2:
                pts = [
                    hand_landmarks[0][0],
                    hand_landmarks[0][17],
                    hand_landmarks[1][17],
                    hand_landmarks[1][0]
                ]

                pts_np = np.array(pts, dtype=np.int32)

                x_min = np.min(pts_np[:, 0])
                y_min = np.min(pts_np[:, 1])
                x_max = np.max(pts_np[:, 0])
                y_max = np.max(pts_np[:, 1])

                rect_pts = (x_min, y_min, x_max, y_max)

                cv2.rectangle(frame,
                              (x_min, y_min),
                              (x_max, y_max),
                              (180, 255, 80),
                              2)

                cv2.putText(frame, "PINCH TO CAPTURE",
                            (x_min + 5, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (180, 255, 80),
                            2)
            pinch = False
            for lm in hand_landmarks:
                x4, y4 = lm[4]
                x8, y8 = lm[8]
                if np.hypot(x4 - x8, y4 - y8) < 40:
                    pinch = True

            if rect_pts and pinch:
                puzzle_img = raw_frame[
                    rect_pts[1]:rect_pts[3],
                    rect_pts[0]:rect_pts[2]
                ].copy()

                captured = True
                puzzle_ui = PuzzleUI(puzzle_img, grid_size)

                cv2.namedWindow("Puzzle")
                cv2.setMouseCallback("Puzzle", puzzle_ui.mouse_event)

            cv2.imshow("Live Puzzle", frame)

        else:
            solved = puzzle_ui.show(
                dark_theme=False,
                window_name="Puzzle"
            )

            if solved:
                cv2.putText(puzzle_ui.window,
                            "COMPLETE!",
                            (100, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 0),
                            4)
                cv2.imshow("Puzzle", puzzle_ui.window)
                cv2.waitKey(2000)
                break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
