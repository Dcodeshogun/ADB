from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture("traffic.mp4")

if not cap.isOpened():
    print("ERROR: Could not open traffic.mp4")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.05,
        verbose=False
    )

    result = results[0]

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # YOLO class name
            name = model.names[cls]

            # Track ID
            if box.id is not None:
                track_id = int(box.id[0])
                label = f"{name} ID:{track_id} {conf:.2f}"
            else:
                label = f"{name} {conf:.2f}"

            print(label)

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
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    cv2.imshow("YOLO ONLY", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()