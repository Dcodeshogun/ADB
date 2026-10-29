import cv2
import numpy as np

VIDEO_PATH = "traffic.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open traffic.mp4")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edges
    edges = cv2.Canny(blur, 50, 150)

    # Road region
    roi = np.array([
    [
        (0, h),
        (int(w * 0.35), int(h * 0.55)),
        (int(w * 0.65), int(h * 0.55)),
        (w, h)
    ]
    ], dtype=np.int32)

    mask = np.zeros_like(edges)

    cv2.fillPoly(mask, roi, 255)

    cropped_edges = cv2.bitwise_and(
        edges,
        mask
    )

    # Hough lines
    lines = cv2.HoughLinesP(
        cropped_edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    left_lines = []
    right_lines = []

    if lines is not None:

        for line in lines:

            # HoughLinesP normally returns [[x1,y1,x2,y2]]
            x1, y1, x2, y2 = line

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            slope = dy / dx

            # Ignore horizontal lines
            if abs(slope) < 0.4:
                continue

            if slope < 0:
                left_lines.append(
                    (x1, y1, x2, y2)
                )
            else:
                right_lines.append(
                    (x1, y1, x2, y2)
                )

    # Create averaged lane
    def make_lane_line(lines):

        if len(lines) == 0:
            return None

        points_x = []
        points_y = []

        for x1, y1, x2, y2 in lines:
            points_x.extend([x1, x2])
            points_y.extend([y1, y2])

        slope, intercept = np.polyfit(
            points_x,
            points_y,
            1
        )

        if abs(slope) < 0.01:
            return None

        y_bottom = h
        y_top = int(h * 0.55)

        x_bottom = int(
            (y_bottom - intercept) / slope
        )

        x_top = int(
            (y_top - intercept) / slope
        )

        return (
            x_bottom,
            y_bottom,
            x_top,
            y_top
        )

    left_lane = make_lane_line(left_lines)
    right_lane = make_lane_line(right_lines)

    # Draw left lane
    if left_lane is not None:

        x1, y1, x2, y2 = left_lane

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5
        )

    # Draw right lane
    if right_lane is not None:

        x1, y1, x2, y2 = right_lane

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5
        )

    # Draw ROI boundary
    cv2.polylines(
        frame,
        roi,
        True,
        (255, 0, 0),
        2
    )

    # Display
    cv2.imshow(
        "Lane Detection",
        frame
    )

    cv2.imshow(
        "Canny ROI",
        cropped_edges
    )

    # ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()