import cv2
import numpy as np
from ultralytics import YOLO

video_path = "../videos/sample_1.mp4"
cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8n.pt")

xr, yr, wr, hr = 500, 300, 400, 600
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[xr : (xr + wr), yr : (yr + hr)]

    cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 240, 240), 2)
    res = model.track(roi, persist=True, conf=0.75, iou=0.8, classes=0, vid_stride=10)[0]
    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
    res_id = res.boxes.id
    if res_id is not None:
        conf = res.boxes.conf.cpu().numpy()
        print(conf)
        ids = res_id.cpu().numpy().astype(int)
        for box, id in zip(boxes, ids):
            cv2.rectangle(roi, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(
                roi,
                f"Id {id}",
                (box[0] + 20, box[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
