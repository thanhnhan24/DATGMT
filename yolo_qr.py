import cv2
import numpy as np
import threading
from queue import Queue
from pyzbar.pyzbar import decode
from ultralytics import YOLO
import pyrealsense2 as rs
import time

# Khởi tạo model YOLO
model = YOLO(r'C:\Users\Thanh Nhan\Desktop\DATGMT\runs-s-1000img\content\runs\segment\train\weights\best.pt')
label_names = model.model.names

# Config RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Hàng đợi ảnh QR
qr_queue = Queue()
stop_event = threading.Event()

# Hàm đọc QR từ queue
def qr_reader_thread(q, stop_event):
    while not stop_event.is_set():
        if not q.empty():
            frame = q.get()
            decoded_objects = decode(frame)
            for obj in decoded_objects:
                data = obj.data.decode("utf-8")
                print(f"[QR] Đã giải mã: {data}")
        time.sleep(0.05)

# Khởi chạy luồng đọc QR
qr_thread = threading.Thread(target=qr_reader_thread, args=(qr_queue, stop_event))
qr_thread.start()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image, verbose=False)[0]
        annotated_frame = color_image.copy()

        for box in results.boxes:
            conf = box.conf.item()
            if conf < 0.7:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = label_names[cls_id]

            # Vẽ bounding box
            color = (0, 255, 0) if label == "qr" else (255, 0, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Nếu label là QR thì crop và gửi vào hàng đợi QR
            if label == "qr":
                qr_crop = color_image[y1:y2, x1:x2].copy()
                if qr_crop.size > 0:
                    qr_queue.put(qr_crop)

        cv2.imshow("YOLO + QR", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop_event.set()
    qr_thread.join()
    pipeline.stop()
    cv2.destroyAllWindows()
