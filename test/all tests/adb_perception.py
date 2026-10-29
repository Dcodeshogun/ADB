from ultralytics import YOLO
import cv2
import numpy as np

# =========================================================
# YOLO
# =========================================================

model = YOLO("yolo26n.pt")

# =========================================================
# VIDEO
# =========================================================

cap = cv2.VideoCapture("traffic.mp4")

if not cap.isOpened():
    print("Could not open traffic.mp4")
    exit()

# =========================================================
# LOCKED CLASSIFICATION
# ByteTrack ID -> direction
# =========================================================

locked_direction = {}

# ByteTrack ID -> vehicle type
locked_type = {}

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    original = frame.copy()

    h, w = frame.shape[:2]

    # =====================================================
    # YOLO + BYTE TRACK
    # =====================================================

    results = model.track(
        original,
        tracker="bytetrack.yaml",
        classes=[2, 3, 7],      # car, motorcycle, truck
        persist=True,
        conf=0.05,
        iou=0.5,
        agnostic_nms=True,
        verbose=False
    )

    result = results[0]

    # YOLO boxes
    output = result.plot()

    # =====================================================
    # OPENCV LANE DETECTION
    # =====================================================

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    # Wider ROI
    roi = np.array([
        [
            (0, h),
            (int(w * 0.20), int(h * 0.55)),
            (int(w * 0.80), int(h * 0.55)),
            (w, h)
        ]
    ], dtype=np.int32)

    mask = np.zeros_like(edges)

    cv2.fillPoly(
        mask,
        roi,
        255
    )

    cropped = cv2.bitwise_and(
        edges,
        mask
    )

    # =====================================================
    # HOUGH
    # =====================================================

    lines = cv2.HoughLinesP(
        cropped,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=100
    )

    left_lines = []
    right_lines = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.3:
                continue

            if slope < 0:

                left_lines.append(
                    (x1, y1, x2, y2)
                )

            else:

                right_lines.append(
                    (x1, y1, x2, y2)
                )

    # =====================================================
    # MAKE LANE
    # =====================================================

    def make_lane(lines):

        if len(lines) == 0:
            return None

        xs = []
        ys = []

        for x1, y1, x2, y2 in lines:

            xs.extend([x1, x2])
            ys.extend([y1, y2])

        if len(xs) < 2:
            return None

        slope, intercept = np.polyfit(
            xs,
            ys,
            1
        )

        if abs(slope) < 0.001:
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
            x_top,
            y_top,
            x_bottom,
            y_bottom
        )

    left_lane = make_lane(left_lines)
    right_lane = make_lane(right_lines)

    # =====================================================
    # DRAW ROI
    # =====================================================

    cv2.polylines(
        output,
        roi,
        True,
        (255, 0, 0),
        3
    )

    # =====================================================
    # DRAW LANES
    # =====================================================

    if left_lane is not None:

        cv2.line(
            output,
            (left_lane[0], left_lane[1]),
            (left_lane[2], left_lane[3]),
            (0, 255, 0),
            5
        )

    if right_lane is not None:

        cv2.line(
            output,
            (right_lane[0], right_lane[1]),
            (right_lane[2], right_lane[3]),
            (0, 255, 0),
            5
        )

    # =====================================================
    # VEHICLES
    # =====================================================

    if result.boxes is not None:

        for box in result.boxes:

            # -------------------------------------------------
            # BYTE TRACK ID
            # -------------------------------------------------

            if box.id is not None:

                track_id = int(box.id[0])

            else:

                track_id = None

            # -------------------------------------------------
            # VEHICLE TYPE
            # -------------------------------------------------

            detected_class = int(box.cls[0])

            detected_type = model.names[detected_class]

            if track_id is not None:

                if track_id not in locked_type:

                    locked_type[track_id] = detected_type

                vehicle_type = locked_type[track_id]

            else:

                vehicle_type = detected_type

            # =================================================
            # BOX
            # =================================================

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # =================================================
            # ALREADY LOCKED DIRECTION?
            # =================================================

            if (
                track_id is not None
                and track_id in locked_direction
            ):

                direction = locked_direction[track_id]

                if direction == "SAME DIRECTION":

                    color = (0, 255, 0)

                else:

                    color = (0, 0, 255)

                red_ratio = 0.0
                white_ratio = 0.0

            else:

                direction = "UNKNOWN"

                # =================================================
                # CROP VEHICLE
                # =================================================

                car = original[y1:y2, x1:x2]

                red_ratio = 0.0
                white_ratio = 0.0

                if car.size != 0:

                    hsv = cv2.cvtColor(
                        car,
                        cv2.COLOR_BGR2HSV
                    )

                    # =================================================
                    # RED
                    # =================================================

                    red1 = cv2.inRange(
                        hsv,
                        np.array([0, 120, 100]),
                        np.array([10, 255, 255])
                    )

                    red2 = cv2.inRange(
                        hsv,
                        np.array([170, 120, 100]),
                        np.array([180, 255, 255])
                    )

                    red_mask = red1 | red2

                    # =================================================
                    # WHITE
                    # =================================================

                    white_mask = cv2.inRange(
                        hsv,
                        np.array([0, 0, 180]),
                        np.array([180, 80, 255])
                    )

                    # =================================================
                    # PIXEL COUNTS
                    # =================================================

                    red_pixels = cv2.countNonZero(
                        red_mask
                    )

                    white_pixels = cv2.countNonZero(
                        white_mask
                    )

                    total_pixels = (
                        car.shape[0] *
                        car.shape[1]
                    )

                    red_ratio = (
                        red_pixels /
                        max(total_pixels, 1)
                    )

                    white_ratio = (
                        white_pixels /
                        max(total_pixels, 1)
                    )

                    # =================================================
                    # CLASSIFY
                    # =================================================

                    if (
                        red_ratio > 0.005
                        and red_ratio > white_ratio * 0.5
                    ):

                        direction = "SAME DIRECTION"

                        if track_id is not None:

                            locked_direction[track_id] = direction

                    elif (
                        white_ratio > 0.015
                        and white_ratio > red_ratio
                    ):

                        direction = "ONCOMING"

                        if track_id is not None:

                            locked_direction[track_id] = direction

            # =================================================
            # COLOR
            # =================================================

            if direction == "SAME DIRECTION":

                color = (0, 255, 0)

            elif direction == "ONCOMING":

                color = (0, 0, 255)

            else:

                color = (255, 255, 255)

            # =================================================
            # CENTER
            # =================================================

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(
                output,
                (cx, cy),
                5,
                color,
                -1
            )

            # =================================================
            # VEHICLE TYPE + ID
            # =================================================

            if track_id is not None:

                cv2.putText(
                    output,
                    f"{vehicle_type} ID:{track_id}",
                    (
                        x1,
                        max(y1 - 45, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    output,
                    vehicle_type,
                    (
                        x1,
                        max(y1 - 45, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            # =================================================
            # DIRECTION
            # =================================================

            cv2.putText(
                output,
                direction,
                (
                    x1,
                    max(y1 - 20, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2
            )

            # =================================================
            # DEBUG
            # =================================================

            cv2.putText(
                output,
                f"R:{red_ratio:.3f} W:{white_ratio:.3f}",
                (
                    x1,
                    min(y2 + 20, h - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1
            )

    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "ADB Perception",
        output
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()