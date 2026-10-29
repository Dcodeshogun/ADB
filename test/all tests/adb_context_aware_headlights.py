from ultralytics import YOLO
import cv2
import numpy as np
import serial
import time

# =========================================================
# CONFIGURATION
# =========================================================
VIDEO_PATH = "traffic3.mp4"
MODEL_PATH = "yolo26n.pt"
SERIAL_PORT = "COM4"
BAUD_RATE = 115200

NUM_COLUMNS = 16
COLUMN_PADDING = 1

# Independent compact-light and persistent headlight-pair detector.
HB_WHITE_V_MIN = 225
HB_WHITE_S_MAX = 165
HB_YELLOW_V_MIN = 190
HB_YELLOW_S_MIN = 25
HB_YELLOW_S_MAX = 220
HB_YELLOW_H_MIN = 8
HB_YELLOW_H_MAX = 43
HB_MIN_AREA = 2
HB_MAX_AREA = 5000
HB_MIN_CONTRAST = 16
HB_DEBUG = True
HIGHBEAM_CAN_CONTROL_MATRIX = True
HB_PAIR_MIN_HITS = 2
HB_PAIR_MAX_MISSES = 2
HB_COLUMN_HOLD_FRAMES = 3
HB_IGNORE_SAME_DIRECTION_VEHICLES = True
HB_IGNORE_EGO_LANE = True
HB_VEHICLE_BOX_PADDING = 0.06
HB_EGO_LANE_MARGIN = 0.02

# =========================================================
# YOLO AND VIDEO
# =========================================================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}")

# ByteTrack ID -> locked direction / vehicle type / light ratios
locked_direction = {}
locked_type = {}
locked_ratios = {}

# Simple persistence for high-beam candidates; it is used for display
# and does not affect existing matrix control unless configured above.
highbeam_history = {}
next_highbeam_id = 1
highbeam_pair_history = {}
next_pair_id = 1
highbeam_column_hold = [0] * NUM_COLUMNS
frame_index = 0

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
    """Send M:<16 bits>; 0 = ON, 1 = OFF."""
    global last_sent_mask, board
    mask_tuple = tuple(bool(value) for value in column_mask)

    if len(mask_tuple) != NUM_COLUMNS:
        return
    if board is None or not board.is_open:
        return
    if mask_tuple == last_sent_mask:
        return

    bit_string = "".join("1" if off else "0" for off in mask_tuple)
    try:
        board.write(f"M:{bit_string}\n".encode("ascii"))
        last_sent_mask = mask_tuple
    except (serial.SerialException, OSError) as exc:
        print(f"Serial write failed: {exc}")
        board = None


# =========================================================
# ORIGINAL VEHICLE-CROP DIRECTION HEURISTIC (UNCHANGED)
# =========================================================
def classify_direction(vehicle_crop):
    """Red pixels suggest tail lights; white pixels suggest headlights."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "UNKNOWN", 0.0, 0.0

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

    total_pixels = max(vehicle_crop.shape[0] * vehicle_crop.shape[1], 1)
    red_ratio = cv2.countNonZero(red_mask) / total_pixels
    white_ratio = cv2.countNonZero(white_mask) / total_pixels

    if red_ratio > 0.005 and red_ratio > white_ratio * 0.5:
        direction = "SAME DIRECTION"
    elif white_ratio > 0.015 and white_ratio > red_ratio:
        direction = "ONCOMING"
    else:
        direction = "UNKNOWN"

    return direction, red_ratio, white_ratio


# =========================================================
# NEW FULL-FRAME HIGH-BEAM CANDIDATE CLASSIFIER
# Does not need a YOLO vehicle box and has NO beam ROI.
# It detects possible bright glare sources for inspection.
# =========================================================
def detect_highbeam_candidates(frame):
    """Find compact local light peaks; pair geometry is checked separately."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    value = hsv[:, :, 2]

    # Local contrast suppresses broad illumination. Exclude the bottom 30%,
    # where lane paint and near-road reflections dominate.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=7, sigmaY=7)
    local_contrast = cv2.subtract(gray, background)
    search_bottom = int(height * 0.70)
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[:search_bottom, :] = 255
    contrast_roi = cv2.bitwise_and(local_contrast, roi)
    region = contrast_roi[:search_bottom, :]
    if region.size == 0:
        empty = np.zeros_like(gray)
        return [], empty, empty, empty

    # Fixed floor prevents a mostly dark frame from making the threshold tiny.
    threshold = max(28, int(np.percentile(region, 99.65)))
    peak_mask = cv2.inRange(contrast_roi, threshold, 255)
    peak_mask = cv2.bitwise_and(peak_mask, roi)
    peak_mask = cv2.morphologyEx(peak_mask, cv2.MORPH_OPEN,
                                 np.ones((2, 2), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        peak_mask, connectivity=8
    )
    candidates = []
    max_area = max(120, int(width * height * 0.0005))
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3 or area > max_area:
            continue
        long_side, short_side = max(w, h), max(1, min(w, h))
        aspect = long_side / short_side
        # Reject lane-marker-like streaks and flat bright bars.
        if long_side >= 6 and aspect > 1.9:
            continue
        if area / float(w * h) < 0.30:
            continue

        component = labels[y:y+h, x:x+w] == label
        contrasts = local_contrast[y:y+h, x:x+w][component]
        values = value[y:y+h, x:x+w][component]
        if contrasts.size == 0:
            continue
        mean_contrast = float(np.mean(contrasts))
        brightness = float(np.mean(values))
        if mean_contrast < 18 or float(np.max(contrasts)) < 32 or brightness < 145:
            continue

        candidates.append({
            "x": x, "y": y, "w": w, "h": h,
            "cx": float(centroids[label][0]),
            "cy": float(centroids[label][1]),
            "area": area, "brightness": brightness,
            "contrast": mean_contrast, "paired": False,
        })

    # Only propose pairs that resemble two lamps on one vehicle: roughly same
    # row, comparable size/brightness, and spacing that grows with image scale.
    min_dx = max(8, int(width * 0.008))
    max_dx = int(width * 0.13)
    pair_candidates = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            dx = abs(a["cx"] - b["cx"])
            dy = abs(a["cy"] - b["cy"])
            avg_size = max(2.0, (max(a["w"], a["h"]) + max(b["w"], b["h"])) / 2)
            if not (min_dx <= dx <= max_dx):
                continue
            if dy > max(3.0, height * 0.012, avg_size * 0.8):
                continue
            size_ratio = max(a["area"], b["area"]) / max(1, min(a["area"], b["area"]))
            bright_ratio = max(a["brightness"], b["brightness"]) / max(1.0, min(a["brightness"], b["brightness"]))
            if size_ratio > 4.0 or bright_ratio > 2.8:
                continue
            # Avoid linking peaks from opposite sides of the frame.
            pair_candidates.append({
                "cx": (a["cx"] + b["cx"]) / 2,
                "cy": (a["cy"] + b["cy"]) / 2,
                "left": min(a["cx"], b["cx"]),
                "right": max(a["cx"], b["cx"]),
                "width": dx,
                "a": a, "b": b,
            })
    # Resolve overlapping hypotheses greedily: each peak can belong to one pair.
    pair_candidates.sort(key=lambda p: (p["a"]["contrast"] + p["b"]["contrast"]), reverse=True)
    used = set()
    pairs = []
    for pair in pair_candidates:
        ka, kb = id(pair["a"]), id(pair["b"])
        if ka in used or kb in used:
            continue
        used.update((ka, kb))
        pair["a"]["paired"] = pair["b"]["paired"] = True
        pairs.append(pair)

    warm_mask = np.zeros_like(peak_mask)
    warm_pixels = cv2.inRange(hsv, np.array([8, 35, 100]), np.array([43, 255, 255]))
    warm_mask = cv2.bitwise_and(peak_mask, warm_pixels)
    return candidates, pairs, peak_mask, warm_mask


def associate_headlight_pairs(pairs, frame_idx):
    """Track pair centers across frames; returns pairs stable for >=2 hits."""
    global highbeam_pair_history, next_pair_id
    for track in highbeam_pair_history.values():
        track["misses"] += 1

    assigned = set()
    for pair in pairs:
        best_id, best_score = None, float("inf")
        for tid, track in highbeam_pair_history.items():
            if tid in assigned or track["misses"] > HB_PAIR_MAX_MISSES:
                continue
            dist = float(np.hypot(pair["cx"] - track["cx"], pair["cy"] - track["cy"]))
            width_delta = abs(pair["width"] - track["width"])
            if dist <= max(35.0, pair["width"] * 0.65) and width_delta <= max(25.0, pair["width"] * 0.6):
                score = dist + 0.35 * width_delta
                if score < best_score:
                    best_id, best_score = tid, score
        if best_id is None:
            best_id = next_pair_id
            next_pair_id += 1
            highbeam_pair_history[best_id] = {
                "cx": pair["cx"], "cy": pair["cy"], "width": pair["width"],
                "hits": 1, "misses": 0, "last_seen": frame_idx,
                "left": pair["left"], "right": pair["right"],
            }
        else:
            track = highbeam_pair_history[best_id]
            track["cx"], track["cy"] = pair["cx"], pair["cy"]
            track["width"] = pair["width"]
            track["left"], track["right"] = pair["left"], pair["right"]
            track["hits"] += 1
            track["misses"] = 0
            track["last_seen"] = frame_idx
        assigned.add(best_id)
        pair["track_id"] = best_id
        pair["confirmed"] = highbeam_pair_history[best_id]["hits"] >= HB_PAIR_MIN_HITS

    for tid in list(highbeam_pair_history):
        if frame_idx - highbeam_pair_history[tid]["last_seen"] > 6:
            del highbeam_pair_history[tid]
    return pairs



def _point_inside_box(x, y, box, width, height):
    x1, y1, x2, y2 = box
    px = max(4, int((x2 - x1) * HB_VEHICLE_BOX_PADDING))
    py = max(4, int((y2 - y1) * HB_VEHICLE_BOX_PADDING))
    return (x1 - px <= x <= x2 + px and y1 - py <= y <= y2 + py)


def _lane_x_at_y(lane, y):
    x_top, y_top, x_bottom, y_bottom = lane
    if y_bottom == y_top:
        return None
    return x_top + (float(y - y_top) * (x_bottom - x_top) / (y_bottom - y_top))


def point_inside_ego_lane(x, y, left_lane, right_lane, width, height):
    """Use both detected lane boundaries; return False when geometry is dubious."""
    if left_lane is None or right_lane is None:
        return False
    # The lane model was fitted only down to 55% frame height. Modest
    # extrapolation upward is allowed, but not all the way to the horizon.
    if y < height * 0.30 or y > height * 0.78:
        return False
    xl = _lane_x_at_y(left_lane, y)
    xr = _lane_x_at_y(right_lane, y)
    if xl is None or xr is None:
        return False
    # Crossing boundaries indicate bad extrapolation / unreliable lane fits.
    if xl >= xr:
        return False
    lane_width = xr - xl
    if lane_width < width * 0.06 or lane_width > width * 0.82:
        return False
    margin = width * HB_EGO_LANE_MARGIN
    return xl - margin <= x <= xr + margin


def filter_headlight_pairs(pairs, same_direction_boxes, left_lane, right_lane, width, height):
    """Return eligible pairs and debug-labelled pairs rejected by context."""
    eligible, ignored = [], []
    for pair in pairs:
        reason = None
        a, b = pair["a"], pair["b"]

        # Rear lights are normally inside their YOLO vehicle box. Require both
        # proposed lamps to lie inside a same-direction box before rejecting.
        if HB_IGNORE_SAME_DIRECTION_VEHICLES:
            for vehicle_box in same_direction_boxes:
                if (_point_inside_box(a["cx"], a["cy"], vehicle_box, width, height) and
                    _point_inside_box(b["cx"], b["cy"], vehicle_box, width, height)):
                    reason = "SAME-DIRECTION VEHICLE / REAR LIGHTS"
                    break

        # For this ADB setup, a candidate located inside our detected lane is
        # treated as same-lane traffic / road glare, not an opposing-lane beam.
        if reason is None and HB_IGNORE_EGO_LANE and point_inside_ego_lane(
            pair["cx"], pair["cy"], left_lane, right_lane, width, height
        ):
            reason = "INSIDE EGO LANE"

        if reason:
            pair["ignored_reason"] = reason
            ignored.append(pair)
        else:
            pair.pop("ignored_reason", None)
            eligible.append(pair)
    return eligible, ignored


def draw_highbeam_debug(frame, candidates, pairs, peak_mask, warm_mask, ignored_pairs=None):
    debug = frame.copy()
    overlay = np.zeros_like(frame)
    overlay[peak_mask > 0] = (190, 190, 190)
    overlay[warm_mask > 0] = (0, 180, 255)
    debug = cv2.addWeighted(debug, 0.90, overlay, 0.25, 0)

    for pair in (ignored_pairs or []):
        a, b = pair["a"], pair["b"]
        color = (0, 0, 255)
        x1 = min(a["x"], b["x"])
        y1 = min(a["y"], b["y"])
        x2 = max(a["x"] + a["w"], b["x"] + b["w"])
        y2 = max(a["y"] + a["h"], b["y"] + b["h"])
        cv2.line(debug, (int(a["cx"]), int(a["cy"])),
                 (int(b["cx"]), int(b["cy"])), color, 2)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.putText(debug, "IGNORED: " + pair.get("ignored_reason", "CONTEXT"),
                    (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, color, 1, cv2.LINE_AA)

    for pair in pairs:
        a, b = pair["a"], pair["b"]
        color = (0, 255, 0) if pair.get("confirmed") else (255, 220, 0)
        cv2.line(debug, (int(a["cx"]), int(a["cy"])),
                 (int(b["cx"]), int(b["cy"])), color, 2)
        x1 = min(a["x"], b["x"])
        y1 = min(a["y"], b["y"])
        x2 = max(a["x"] + a["w"], b["x"] + b["w"])
        y2 = max(a["y"] + a["h"], b["y"] + b["h"])
        label = f'PAIR #{pair.get("track_id", "?")} ' + ("CONFIRMED" if pair.get("confirmed") else "CHECKING")
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.putText(debug, label, (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Unpaired peaks are visualized but never allowed to control the LEDs.
    for c in candidates:
        if c.get("paired"):
            continue
        cv2.rectangle(debug, (c["x"], c["y"]),
                      (c["x"] + c["w"], c["y"] + c["h"]), (0, 140, 255), 1)
    cv2.putText(debug, "GREEN=TRACKED / CYAN=NEW / RED=IGNORED BY YOLO OR LANE / ORANGE=UNPAIRED",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return debug


# =========================================================
# LANE DETECTION HELPER (ORIGINAL)
# =========================================================
def make_lane(lines, height):
    if not lines:
        return None

    xs, ys = [], []
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
# LED MATRIX VISUALIZATION (ORIGINAL)
# =========================================================
def draw_matrix_overlay(column_mask, cell_size=28):
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
        panel, "ADB LED MATRIX - 2 x 8x8", (padding, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        panel, "M1", (padding + 3 * cell_size, 49),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
    )
    cv2.putText(
        panel, "M2", (padding + 11 * cell_size, 49),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
    )

    for col in range(cols):
        color = (0, 0, 225) if column_mask[col] else (0, 190, 0)
        for row in range(rows):
            x = padding + col * cell_size + cell_size // 2
            y = header + row * cell_size + cell_size // 2
            cv2.circle(panel, (x, y), cell_size // 2 - 4, color, -1)

    divider_x = padding + 8 * cell_size
    cv2.line(
        panel, (divider_x, header - 5),
        (divider_x, header + rows * cell_size), (230, 230, 230), 2
    )

    off_count = sum(bool(value) for value in column_mask)
    cv2.putText(
        panel,
        f"OFF columns: {off_count}/{cols} RED=OFF GREEN=ON",
        (padding, header + rows * cell_size + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA
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

        frame_index += 1
        original = frame.copy()
        height, width = frame.shape[:2]

        # -------------------------------------------------
        # NEW: INDEPENDENT HIGH-BEAM CANDIDATES, FULL FRAME
        # Runs whether or not YOLO detects a vehicle.
        # -------------------------------------------------
        highbeam_candidates, pair_hypotheses, highbeam_white, highbeam_yellow = (
            detect_highbeam_candidates(original)
        )
        # Pair tracking is delayed until YOLO vehicle directions and lane
        # geometry are available, so taillights / ego-lane lights can be rejected.

        # -------------------------------------------------
        # YOLO + BYTE TRACK (ORIGINAL SETTINGS)
        # -------------------------------------------------
        results = model.track(
            original,
            tracker="bytetrack.yaml",
            classes=[2, 3, 7],
            persist=True,
            conf=0.05,
            iou=0.5,
            agnostic_nms=True,
            verbose=False
        )

        result = results[0]
        output = result.plot()

        # -------------------------------------------------
        # LANE DETECTION (ORIGINAL ROI; SEPARATE FROM BEAMS)
        # -------------------------------------------------
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        roi = np.array([[
            (0, height),
            (int(width * 0.20), int(height * 0.55)),
            (int(width * 0.80), int(height * 0.55)),
            (width, height)
        ]], dtype=np.int32)

        roi_mask = np.zeros_like(edges)
        cv2.fillPoly(roi_mask, roi, 255)
        cropped = cv2.bitwise_and(edges, roi_mask)

        lines = cv2.HoughLinesP(
            cropped,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=100
        )

        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, np.asarray(line).reshape(-1)[:4])
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

        cv2.polylines(output, roi, True, (255, 0, 0), 2)
        for lane in (left_lane, right_lane):
            if lane is not None:
                cv2.line(
                    output, (lane[0], lane[1]), (lane[2], lane[3]),
                    (0, 255, 0), 4
                )

        # -------------------------------------------------
        # ORIGINAL VEHICLE CLASSIFICATION / MATRIX CONTROL
        # -------------------------------------------------
        column_mask = [False] * NUM_COLUMNS
        same_direction_boxes = []

        if result.boxes is not None:
            for box in result.boxes:
                track_id = int(box.id[0]) if box.id is not None else None

                detected_class = int(box.cls[0])
                detected_type = model.names[detected_class]

                if track_id is not None:
                    if track_id not in locked_type:
                        locked_type[track_id] = detected_type
                    vehicle_type = locked_type[track_id]
                else:
                    vehicle_type = detected_type

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                red_ratio = 0.0
                white_ratio = 0.0

                if track_id is not None and track_id in locked_direction:
                    direction = locked_direction[track_id]
                    red_ratio, white_ratio = locked_ratios.get(
                        track_id, (0.0, 0.0)
                    )
                else:
                    direction = "UNKNOWN"
                    vehicle_crop = original[y1:y2, x1:x2]
                    if vehicle_crop.size:
                        direction, red_ratio, white_ratio = classify_direction(
                            vehicle_crop
                        )
                        if track_id is not None and direction != "UNKNOWN":
                            locked_direction[track_id] = direction
                            locked_ratios[track_id] = (red_ratio, white_ratio)

                if direction == "SAME DIRECTION":
                    # Save this vehicle box so the independent light detector
                    # cannot mistake its rear lamps for oncoming headlights.
                    same_direction_boxes.append((x1, y1, x2, y2))
                    color = (0, 255, 0)
                elif direction == "ONCOMING":
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 255)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(output, (cx, cy), 5, color, -1)

                id_text = f"ID:{track_id}" if track_id is not None else "ID:?"
                cv2.putText(
                    output,
                    f"{vehicle_type} {id_text}",
                    (x1, max(y1 - 42, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
                cv2.putText(
                    output,
                    direction,
                    (x1, max(y1 - 20, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )
                cv2.putText(
                    output,
                    f"R:{red_ratio:.3f} W:{white_ratio:.3f}",
                    (x1, min(y2 + 18, height - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
                )

                # Existing vehicle-based matrix trigger unchanged.
                if direction == "ONCOMING":
                    first_col = int(x1 * NUM_COLUMNS / width)
                    last_col = int((x2 - 1) * NUM_COLUMNS / width)

                    first_col = max(0, min(NUM_COLUMNS - 1, first_col))
                    last_col = max(0, min(NUM_COLUMNS - 1, last_col))
                    first_col = max(0, first_col - COLUMN_PADDING)
                    last_col = min(NUM_COLUMNS - 1, last_col + COLUMN_PADDING)

                    for col in range(first_col, last_col + 1):
                        column_mask[col] = True

        # -------------------------------------------------
        # CONTEXT FILTER: YOLO DIRECTION + EGO-LANE GEOMETRY
        # Reject pairs inside same-direction vehicle boxes (rear lamps), then
        # reject pairs whose center lies inside the detected ego-lane corridor.
        # If both lane boundaries are not available / plausible, lane filtering
        # is skipped rather than guessing.
        # -------------------------------------------------
        eligible_pairs, ignored_pair_hypotheses = filter_headlight_pairs(
            pair_hypotheses,
            same_direction_boxes,
            left_lane, right_lane,
            width, height
        )
        confirmed_pairs = associate_headlight_pairs(eligible_pairs, frame_index)

        # -------------------------------------------------
        # INDEPENDENT HEADLIGHT-PAIR CONTROL
        # Only context-approved persistent pairs can affect the matrix.
        # Single peaks and ignored pairs are debug-only. Hold briefly to prevent flicker.
        # -------------------------------------------------
        if HIGHBEAM_CAN_CONTROL_MATRIX:
            for col in range(NUM_COLUMNS):
                if highbeam_column_hold[col] > 0:
                    highbeam_column_hold[col] -= 1

            for pair in confirmed_pairs:
                if not pair.get("confirmed", False):
                    continue
                # Map the span of both lamps, with a small safety padding.
                left_x = max(0.0, pair["left"] - max(2.0, width * 0.004))
                right_x = min(float(width - 1), pair["right"] + max(2.0, width * 0.004))
                first_col = int(left_x * NUM_COLUMNS / width)
                last_col = int(right_x * NUM_COLUMNS / width)
                first_col = max(0, min(NUM_COLUMNS - 1, first_col - COLUMN_PADDING))
                last_col = max(0, min(NUM_COLUMNS - 1, last_col + COLUMN_PADDING))
                for col in range(first_col, last_col + 1):
                    highbeam_column_hold[col] = HB_COLUMN_HOLD_FRAMES

            for col in range(NUM_COLUMNS):
                if highbeam_column_hold[col] > 0:
                    column_mask[col] = True

        # -------------------------------------------------
        # SEND MATRIX + DISPLAY
        # -------------------------------------------------
        send_column_mask(column_mask)

        cv2.putText(
            output,
            f"ADB: {sum(column_mask)}/{NUM_COLUMNS} COLUMNS OFF",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )

        cv2.imshow("ADB Perception", output)
        cv2.imshow("ADB Matrix Overlay", draw_matrix_overlay(column_mask))

        if HB_DEBUG:
            cv2.imshow(
                "High-Beam Candidates (Full Frame)",
                draw_highbeam_debug(
                    original,
                    highbeam_candidates,
                    confirmed_pairs,
                    highbeam_white,
                    highbeam_yellow,
                    ignored_pair_hypotheses
                )
            )

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    if board is not None:
        try:
            all_on = [False] * NUM_COLUMNS
            bit_string = "".join("1" if off else "0" for off in all_on)
            board.write(f"M:{bit_string}\n".encode("ascii"))
            board.close()
        except (serial.SerialException, OSError):
            pass

    cap.release()
    cv2.destroyAllWindows()
