from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import torch

# Check GPU
print("Sử dụng GPU:", torch.cuda.is_available())

# Cấu hình RealSense
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors)
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load YOLO model (sử dụng GPU nếu có)
model_path = r'C:\Users\Thanh Nhan\Desktop\DATGMT\runs-s-1000img\content\runs\segment\train\weights\best.pt'
model = YOLO(model_path)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert to numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Detect với YOLO (tự động sử dụng GPU nếu có)
        results = model(color_image, verbose=False)[0]

        # Annotated frame
        annotated_frame = color_image.copy()

        for box in results.boxes:
            conf = box.conf[0].item()
            if conf < 0.65:  # lọc confidence < 65%
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
            text = f'{label} {conf:.2f}'

            # Vẽ bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Xử lý depth colormap
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)

        if depth_colormap.shape != annotated_frame.shape:
            annotated_frame = cv2.resize(annotated_frame, (depth_colormap.shape[1], depth_colormap.shape[0]))

        images = np.hstack((annotated_frame, depth_colormap))
        cv2.imshow('RealSense YOLO Detection', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
