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

# Full-frame high-beam candidate detector.
# These identify possible glare sources, not guaranteed high beams.
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

# Keep False initially so new candidates cannot unexpectedly
# blank the matrix. Existing vehicle-based matrix control stays intact.
HIGHBEAM_CAN_CONTROL_MATRIX = False  # intentionally locked OFF; debug detector cannot control LEDs

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
    """Find compact local light peaks, not all bright pixels.

    The detector uses local contrast (high-pass image), rejects the lower
    road-glare zone and elongated patches, then scores plausible horizontal
    headlight pairs. Pairing is debug-only; single lights remain visible.
    """
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    value = hsv[:, :, 2]

    # Road glare and painted markings are often broad, slowly varying regions.
    # Subtracting a large blur leaves compact bright peaks much more prominent.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=9)
    local_contrast = cv2.subtract(gray, background)

    # Keep the scene above the nearest road foreground. This is deliberately
    # generous: distant headlights can still appear low in the image.
    search_bottom = int(height * 0.76)
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[:search_bottom, :] = 255
    contrast_roi = cv2.bitwise_and(local_contrast, roi)

    # A compact source must be locally bright, not merely white.
    threshold = max(22, int(np.percentile(contrast_roi[:search_bottom, :], 99.3)))
    peak_mask = cv2.inRange(contrast_roi, threshold, 255)
    peak_mask = cv2.bitwise_and(peak_mask, roi)
    peak_mask = cv2.morphologyEx(
        peak_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        peak_mask, connectivity=8
    )
    candidates = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 2 or area > max(500, int(width * height * 0.0015)):
            continue

        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side
        # Reject road stripes, lane markings and long reflected streaks.
        if long_side >= 7 and aspect > 2.4:
            continue
        fill = area / float(w * h)
        if fill < 0.18:
            continue

        component = labels[y:y+h, x:x+w] == label
        component_contrast = local_contrast[y:y+h, x:x+w][component]
        component_value = value[y:y+h, x:x+w][component]
        if component_contrast.size == 0:
            continue
        mean_contrast = float(np.mean(component_contrast))
        peak_contrast = float(np.max(component_contrast))
        brightness = float(np.mean(component_value))
        if mean_contrast < 12 or peak_contrast < 25 or brightness < 100:
            continue

        # Color is descriptive only. White/yellow classification does not
        # decide whether a region is a headlight.
        sat = hsv[y:y+h, x:x+w, 1][component]
        hue = hsv[y:y+h, x:x+w, 0][component]
        warm = int(np.count_nonzero((hue >= 8) & (hue <= 43) & (sat > 35)))
        kind = "WARM" if warm > component.size * 0.25 else "WHITE"
        candidates.append({
            "x": x, "y": y, "w": w, "h": h,
            "cx": float(centroids[label][0]),
            "cy": float(centroids[label][1]),
            "area": area, "brightness": brightness,
            "contrast": mean_contrast, "kind": kind,
            "paired": False,
        })

    # Pairing is a confidence boost, never a hard requirement. It only links
    # compact peaks on roughly the same row with plausible horizontal spacing.
    for i, a in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            dx = abs(a["cx"] - b["cx"])
            dy = abs(a["cy"] - b["cy"])
            size = max(a["w"], a["h"], b["w"], b["h"], 1)
            if dx < size * 1.4 or dx > width * 0.16:
                continue
            if dy > max(5.0, size * 1.1):
                continue
            if min(a["brightness"], b["brightness"]) < 105:
                continue
            a["paired"] = b["paired"] = True

    # Keep the existing debug renderer's interface. White mask = local peaks;
    # warm mask marks warm-colored pixels only within those peaks.
    white_mask = peak_mask.copy()
    warm_mask = np.zeros_like(peak_mask)
    warm_pixels = cv2.inRange(
        hsv,
        np.array([8, 35, 100]),
        np.array([43, 255, 255])
    )
    warm_mask = cv2.bitwise_and(peak_mask, warm_pixels)
    return candidates, white_mask, warm_mask, peak_mask


def associate_highbeam_candidates(candidates, frame_idx):
    """Lightweight temporal association used for DEBUG confirmation only."""
    global highbeam_history, next_highbeam_id

    max_distance = 30.0
    assigned_this_frame = set()
    for track in highbeam_history.values():
        track["misses"] += 1

    for candidate in candidates:
        best_id = None
        best_distance = float("inf")

        for track_id, track in highbeam_history.items():
            if track_id in assigned_this_frame:
                continue
            if track["misses"] > 2:
                continue
            dist = float(np.hypot(
                candidate["cx"] - track["cx"],
                candidate["cy"] - track["cy"]
            ))
            if dist < max_distance and dist < best_distance:
                best_id = track_id
                best_distance = dist

        if best_id is None:
            best_id = next_highbeam_id
            next_highbeam_id += 1
            highbeam_history[best_id] = {
                "cx": candidate["cx"],
                "cy": candidate["cy"],
                "hits": 1,
                "misses": 0,
                "last_seen": frame_idx,
            }
        else:
            track = highbeam_history[best_id]
            track["cx"] = candidate["cx"]
            track["cy"] = candidate["cy"]
            track["hits"] += 1
            track["misses"] = 0
            track["last_seen"] = frame_idx

        assigned_this_frame.add(best_id)
        candidate["track_id"] = best_id
        candidate["confirmed"] = highbeam_history[best_id]["hits"] >= 3

    stale = [
        tid for tid, track in highbeam_history.items()
        if frame_idx - track["last_seen"] > 8
    ]
    for tid in stale:
        del highbeam_history[tid]

    return candidates


def draw_highbeam_debug(frame, candidates, white_mask, yellow_mask):
    """Show local-contrast peaks and pair hypotheses; never controls LEDs."""
    debug = frame.copy()
    overlay = np.zeros_like(frame)
    overlay[white_mask > 0] = (190, 190, 190)
    overlay[yellow_mask > 0] = (0, 180, 255)
    debug = cv2.addWeighted(debug, 0.88, overlay, 0.30, 0)

    paired = [c for c in candidates if c.get("paired", False)]
    # Draw pair links first so boxes stay visible above them.
    for i, a in enumerate(paired):
        for b in paired[i + 1:]:
            dx = abs(a["cx"] - b["cx"])
            dy = abs(a["cy"] - b["cy"])
            size = max(a["w"], a["h"], b["w"], b["h"], 1)
            if size * 1.4 <= dx <= frame.shape[1] * 0.16 and dy <= max(5.0, size * 1.1):
                cv2.line(debug, (int(a["cx"]), int(a["cy"])),
                         (int(b["cx"]), int(b["cy"])), (255, 220, 0), 1)

    for c in candidates:
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        confirmed = c.get("confirmed", False)
        if c.get("paired", False):
            color = (255, 220, 0)  # plausible pair, not a final detection
            tag = "PAIR?"
        elif confirmed:
            color = (0, 255, 0)
            tag = "TRACKED"
        else:
            color = (0, 165, 255)
            tag = "PEAK"
        tid = c.get("track_id", "-")
        cv2.rectangle(debug, (x, y), (x+w, y+h), color, 2)
        cv2.putText(debug, f'{tag} #{tid}', (x, max(16, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)

    cv2.putText(debug, "LOCAL-CONTRAST HEADLIGHT CANDIDATES | DEBUG ONLY",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(debug, "Cyan = plausible pair; orange = unpaired peak; green = persistent",
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
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
        highbeam_candidates, highbeam_white, highbeam_yellow, _ = (
            detect_highbeam_candidates(original)
        )
        highbeam_candidates = associate_highbeam_candidates(
            highbeam_candidates, frame_index
        )

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

        # Independent headlight candidates are debug-only; they never alter column_mask.

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
                    highbeam_white,
                    highbeam_yellow
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
