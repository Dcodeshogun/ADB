from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")

name_map = {}
next_car = 1

# DroidCam camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera")
        break

    # Vehicle detection + tracking
    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        persist=True,
        verbose=False
    )

    r = results[0]

    # Draw tracked cars
    if r.boxes.id is not None:

        for box in r.boxes:
            tid = int(box.id)
            cls = int(box.cls)

            # Only cars
            if cls != 2:
                continue

            # Give friendly name
            if tid not in name_map:
                name_map[tid] = f"Car {next_car}"
                next_car += 1

            label = name_map[tid]

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    # Display
    cv2.imshow("DroidCam Vehicle Tracking", frame)

    # ESC to quit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()