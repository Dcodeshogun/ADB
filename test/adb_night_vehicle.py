from ultralytics import YOLO
from huggingface_hub import hf_hub_download

import cv2
import numpy as np
import serial
import time

# =========================================================
# CONFIGURATION
# =========================================================
VIDEO_PATH = "traffic3.mp4"

# Night-trained YOLOv8s checkpoint hosted on Hugging Face.
MODEL_REPO = "G2mus/night-vehicle-yolov8s"
MODEL_FILENAME = "night-vehicle-yolov8s.pt"

# Initial tracking confidence. Tune this on your own night footage.
DETECTION_CONFIDENCE = 0.20

SERIAL_PORT = "COM4"
BAUD_RATE = 115200

NUM_COLUMNS = 16
COLUMN_PADDING = 1

# =========================================================
# NIGHT VEHICLE MODEL AND VIDEO
# =========================================================
# Downloads the checkpoint on first run and reuses the Hugging Face cache
# on subsequent runs.
weights_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME
)
model = YOLO(weights_path)

# Resolve class IDs from the checkpoint instead of using COCO IDs.
# This checkpoint has: car, truck, bus (no motorcycle class).
if isinstance(model.names, dict):
    name_to_id = {
        str(class_name).lower(): int(class_id)
        for class_id, class_name in model.names.items()
    }
else:
    name_to_id = {
        str(class_name).lower(): class_id
        for class_id, class_name in enumerate(model.names)
    }

VEHICLE_CLASSES = [
    name_to_id[name]
    for name in ("car", "truck", "bus")
    if name in name_to_id
]

if not VEHICLE_CLASSES:
    raise RuntimeError(
        f"No expected vehicle classes found in model. Model classes: {model.names}"
    )

print(f"Loaded night vehicle model: {weights_path}")
print(f"Model classes: {model.names}")
print(f"Tracking vehicle class IDs: {VEHICLE_CLASSES}")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}")

# ByteTrack ID -> locked direction / vehicle type / light ratios
locked_direction = {}
locked_type = {}
locked_ratios = {}

# =========================================================
# SERIAL CONNECTION
# =========================================================
board = None
last_sent_mask = None

try:
    board = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    board.reset_input_buffer()
    print(f"Connected to NodeMCU on {SERIAL_PORT}")

except (serial.SerialException, OSError) as exc:
    print(f"NodeMCU serial connection failed: {exc}")
    print("Detection and matrix overlay will still run.")
    board = None


def send_column_mask(column_mask):
    """
    Send M:<16 bits> to the NodeMCU.

    0 = column ON
    1 = column OFF
    """
    global last_sent_mask, board

    mask_tuple = tuple(bool(value) for value in column_mask)

    if len(mask_tuple) != NUM_COLUMNS:
        return

    if board is None or not board.is_open:
        return

    if mask_tuple == last_sent_mask:
        return

    bit_string = "".join(
        "1" if off else "0"
        for off in mask_tuple
    )

    try:
        board.write(f"M:{bit_string}\n".encode("ascii"))
        last_sent_mask = mask_tuple

    except (serial.SerialException, OSError) as exc:
        print(f"Serial write failed: {exc}")
        board = None


# =========================================================
# VEHICLE DIRECTION CLASSIFICATION
# =========================================================
def classify_direction(vehicle_crop):
    """
    Experimental heuristic:
    Red pixels suggest tail lights.
    White pixels suggest headlights.
    """

    hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)

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

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 180]),
        np.array([180, 80, 255])
    )

    total_pixels = max(
        vehicle_crop.shape[0] * vehicle_crop.shape[1],
        1
    )

    red_ratio = cv2.countNonZero(red_mask) / total_pixels
    white_ratio = cv2.countNonZero(white_mask) / total_pixels

    if (
        red_ratio > 0.005
        and red_ratio > white_ratio * 0.5
    ):
        direction = "SAME DIRECTION"

    elif (
        white_ratio > 0.015
        and white_ratio > red_ratio
    ):
        direction = "ONCOMING"

    else:
        direction = "UNKNOWN"

    return direction, red_ratio, white_ratio


# =========================================================
# LANE DETECTION HELPER
# =========================================================
def make_lane(lines, height):
    if not lines:
        return None

    xs = []
    ys = []

    for x1, y1, x2, y2 in lines:
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    if len(xs) < 2:
        return None

    slope, intercept = np.polyfit(xs, ys, 1)

    if abs(slope) < 0.001:
        return None

    y_bottom = height
    y_top = int(height * 0.55)

    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)

    return x_top, y_top, x_bottom, y_bottom


# =========================================================
# LED MATRIX VISUALIZATION
# =========================================================
def draw_matrix_overlay(column_mask, cell_size=28):
    """
    Display two 8x8 matrices side by side.

    GREEN = column ON
    RED   = column OFF
    """

    rows = 8
    cols = len(column_mask)

    padding = 18
    header = 58
    footer = 34

    panel = np.zeros(
        (
            header + rows * cell_size + footer + padding,
            padding * 2 + cols * cell_size,
            3
        ),
        dtype=np.uint8
    )

    cv2.putText(
        panel,
        "ADB LED MATRIX - 2 x 8x8",
        (padding, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Labels for the two physical matrix modules
    cv2.putText(
        panel,
        "M1",
        (padding + 3 * cell_size, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        panel,
        "M2",
        (padding + 11 * cell_size, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # Draw each LED in the 8x16 preview
    for col in range(cols):
        if column_mask[col]:
            color = (0, 0, 225)       # RED = OFF
        else:
            color = (0, 190, 0)       # GREEN = ON

        for row in range(rows):
            x = padding + col * cell_size + cell_size // 2
            y = header + row * cell_size + cell_size // 2

            cv2.circle(
                panel,
                (x, y),
                cell_size // 2 - 4,
                color,
                -1
            )

    # Divider between matrix 1 and matrix 2
    divider_x = padding + 8 * cell_size

    cv2.line(
        panel,
        (divider_x, header - 5),
        (divider_x, header + rows * cell_size),
        (230, 230, 230),
        2
    )

    off_count = sum(bool(value) for value in column_mask)

    cv2.putText(
        panel,
        f"OFF columns: {off_count}/{cols}   RED=OFF  GREEN=ON",
        (padding, header + rows * cell_size + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA
    )

    return panel


# =========================================================
# MAIN LOOP
# =========================================================
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        original = frame.copy()
        height, width = frame.shape[:2]

        # -------------------------------------------------
        # YOLO + BYTE TRACK
        # -------------------------------------------------
        results = model.track(
            original,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,  # car, truck, bus for this checkpoint
            persist=True,
            conf=DETECTION_CONFIDENCE,
            iou=0.5,
            agnostic_nms=True,
            verbose=False
        )

        result = results[0]
        output = result.plot()

        # -------------------------------------------------
        # LANE DETECTION
        # -------------------------------------------------
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

        roi = np.array([[
            (0, height),
            (int(width * 0.20), int(height * 0.55)),
            (int(width * 0.80), int(height * 0.55)),
            (width, height)
        ]], dtype=np.int32)

        roi_mask = np.zeros_like(edges)

        cv2.fillPoly(
            roi_mask,
            roi,
            255
        )

        cropped = cv2.bitwise_and(
            edges,
            roi_mask
        )

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
                x1, y1, x2, y2 = map(
                    int,
                    np.asarray(line).reshape(-1)[:4]
                )

                if x2 == x1:
                    continue

                slope = (y2 - y1) / (x2 - x1)

                if abs(slope) < 0.3:
                    continue

                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))

        left_lane = make_lane(left_lines, height)
        right_lane = make_lane(right_lines, height)

        cv2.polylines(
            output,
            roi,
            True,
            (255, 0, 0),
            2
        )

        for lane in (left_lane, right_lane):
            if lane is not None:
                cv2.line(
                    output,
                    (lane[0], lane[1]),
                    (lane[2], lane[3]),
                    (0, 255, 0),
                    4
                )

        # -------------------------------------------------
        # BUILD MATRIX COLUMN MASK
        # False = ON
        # True  = OFF
        # -------------------------------------------------
        column_mask = [False] * NUM_COLUMNS

        if result.boxes is not None:
            for box in result.boxes:

                # Get ByteTrack ID
                track_id = (
                    int(box.id[0])
                    if box.id is not None
                    else None
                )

                # Vehicle type
                detected_class = int(box.cls[0])
                detected_type = model.names[detected_class]

                if track_id is not None:
                    if track_id not in locked_type:
                        locked_type[track_id] = detected_type

                    vehicle_type = locked_type[track_id]
                else:
                    vehicle_type = detected_type

                # Bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                red_ratio = 0.0
                white_ratio = 0.0

                # Use locked direction if available
                if (
                    track_id is not None
                    and track_id in locked_direction
                ):
                    direction = locked_direction[track_id]

                    red_ratio, white_ratio = locked_ratios.get(
                        track_id,
                        (0.0, 0.0)
                    )

                else:
                    direction = "UNKNOWN"

                    vehicle_crop = original[y1:y2, x1:x2]

                    if vehicle_crop.size:
                        direction, red_ratio, white_ratio = (
                            classify_direction(vehicle_crop)
                        )

                        if (
                            track_id is not None
                            and direction != "UNKNOWN"
                        ):
                            locked_direction[track_id] = direction

                            locked_ratios[track_id] = (
                                red_ratio,
                                white_ratio
                            )

                # -------------------------------------------------
                # DRAW VEHICLE INFORMATION
                # -------------------------------------------------
                if direction == "SAME DIRECTION":
                    color = (0, 255, 0)

                elif direction == "ONCOMING":
                    color = (0, 0, 255)

                else:
                    color = (255, 255, 255)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.circle(
                    output,
                    (cx, cy),
                    5,
                    color,
                    -1
                )

                id_text = (
                    f"ID:{track_id}"
                    if track_id is not None
                    else "ID:?"
                )

                cv2.putText(
                    output,
                    f"{vehicle_type} {id_text}",
                    (x1, max(y1 - 42, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    output,
                    direction,
                    (x1, max(y1 - 20, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

                cv2.putText(
                    output,
                    f"R:{red_ratio:.3f} W:{white_ratio:.3f}",
                    (x1, min(y2 + 18, height - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1
                )

                # -------------------------------------------------
                # MAP ONCOMING VEHICLE TO MATRIX COLUMNS
                # -------------------------------------------------
                # Demo mapping:
                # horizontal video position -> matrix columns.
                # This is not true road-lane estimation.
                if direction == "ONCOMING":

                    first_col = int(
                        x1 * NUM_COLUMNS / width
                    )

                    last_col = int(
                        (x2 - 1) * NUM_COLUMNS / width
                    )

                    first_col = max(
                        0,
                        min(NUM_COLUMNS - 1, first_col)
                    )

                    last_col = max(
                        0,
                        min(NUM_COLUMNS - 1, last_col)
                    )

                    # Add neighbouring columns for a wider OFF zone
                    first_col = max(
                        0,
                        first_col - COLUMN_PADDING
                    )

                    last_col = min(
                        NUM_COLUMNS - 1,
                        last_col + COLUMN_PADDING
                    )

                    for col in range(first_col, last_col + 1):
                        column_mask[col] = True

        # -------------------------------------------------
        # SEND MASK TO NODEMCU
        # -------------------------------------------------
        send_column_mask(column_mask)

        off_count = sum(column_mask)

        cv2.putText(
            output,
            f"ADB: {off_count}/{NUM_COLUMNS} COLUMNS OFF",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        # Window 1: vehicle detection and lanes
        cv2.imshow(
            "ADB Perception",
            output
        )

        # Window 2: simulated LED states
        # Red = OFF, Green = ON
        matrix_view = draw_matrix_overlay(column_mask)

        cv2.imshow(
            "ADB Matrix Overlay",
            matrix_view
        )

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Restore all columns to ON before closing serial.
    if board is not None:
        try:
            all_on = [False] * NUM_COLUMNS

            bit_string = "".join(
                "1" if off else "0"
                for off in all_on
            )

            board.write(
                f"M:{bit_string}\n".encode("ascii")
            )

            board.close()

        except (serial.SerialException, OSError):
            pass

    cap.release()
    cv2.destroyAllWindows()