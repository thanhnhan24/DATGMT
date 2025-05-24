from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import time

from deep_sort_realtime.deepsort_tracker import DeepSort

# Cấu hình RealSense
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors)
if not found_rgb:
    print("Camera cần phải có RGB sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load model YOLO
model = YOLO(r'D:\DATGMT-main\runs-m-1000image\content\runs\segment\train\weights\best.pt')  # Thay bằng đường dẫn model của bạn

# Khởi tạo Deep SORT - chỉ track các bbox label 'box' (sẽ lọc phía dưới)
tracker = DeepSort(max_age=30, n_init=3)

DEBUG_DISTANCE = False

def iou(box1, box2):
    x1_min = box1[0]
    y1_min = box1[1]
    x1_max = box1[0] + box1[2]
    y1_max = box1[1] + box1[3]

    x2_min = box2[0]
    y2_min = box2[1]
    x2_max = box2[0] + box2[2]
    y2_max = box2[1] + box2[3]

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def filter_overlapping_boxes(detections, iou_threshold=0.75):
    filtered = []
    for det in detections:
        keep = True
        for fdet in filtered:
            if iou(det[0], fdet[0]) > iou_threshold:
                keep = False
                break
        if keep:
            filtered.append(det)
    return filtered

def get_distance_avg(depth_frame, x, y, size=5):
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    x = int(x)
    y = int(y)

    sum_dist = 0
    count = 0

    half = size // 2

    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            px = x + dx
            py = y + dy
            if 0 <= px < width and 0 <= py < height:
                dist = depth_frame.get_distance(px, py)
                if dist > 0:
                    sum_dist += dist
                    count += 1

    if count == 0:
        return None
    avg_dist = sum_dist / count
    if DEBUG_DISTANCE:
        print(f"Distance avg at ({x},{y}) size {size} = {avg_dist:.4f} m from {count} points")
    return round(avg_dist * 100, 2)  # cm

# Hàm lấy label an toàn từ track.det_class
def get_label_from_track(track, model):
    if hasattr(model, 'names') and track.det_class is not None:
        try:
            cls_id = int(track.det_class)
            if 0 <= cls_id < len(model.names):
                return model.names[cls_id]
            else:
                return str(cls_id)
        except Exception:
            return str(track.det_class)
    else:
        return str(track.det_class)

# Vòng lặp chính với tính FPS
prev_time = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Tính FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Detect với YOLO
        results = model(color_image)[0]

        dets = []
        for box in results.boxes:
            conf = box.conf.item()
            if conf < 0.7:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            label = model.names[cls_id] if hasattr(model, 'names') and 0 <= cls_id < len(model.names) else str(cls_id)
            dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id, label))  # Thêm label vào để lọc

        # Lọc bbox chồng lấn
        dets_filtered = filter_overlapping_boxes(dets, iou_threshold=0.75)

        # Chỉ giữ detections có label 'box' để track
        dets_to_track = [ (det[0], det[1], det[2]) for det in dets_filtered if det[3] == "box" ]

        # Cập nhật tracker
        tracks = tracker.update_tracks(dets_to_track, frame=color_image)

        annotated_frame = color_image.copy()

        # Vẽ tất cả bbox (bao gồm cả không track) - không hiển thị khoảng cách cho các label khác 'box'
        for det in dets_filtered:
            bbox, conf, cls_id, label = det
            x, y, w, h = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            color = (255, 0, 0) if label == "box" else (0, 255, 255)  # Xanh dương cho box, vàng cho khác
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Vẽ bbox, ID và khoảng cách cho các track confirmed (chỉ label 'box')
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltwh()
            x1, y1, x2, y2 = int(l), int(t), int(l + w), int(t + h)

            cx = int(l + w / 2)
            cy = int(t + h / 2)
            cx = max(0, min(cx, depth_frame.get_width() - 1))
            cy = max(0, min(cy, depth_frame.get_height() - 1))

            dist_cm = get_distance_avg(depth_frame, cx, cy, size=5)
            dist_text = f"{dist_cm} cm" if dist_cm is not None else "N/A"

            label = get_label_from_track(track, model)

            # Vẽ bbox, ID, label và khoảng cách
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"ID:{track_id} {label} {dist_text}"
            cv2.putText(annotated_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Chữ vàng sáng
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

        # Hiển thị FPS màu vàng ở góc trên cùng bên trái
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Hiển thị depth colormap cạnh ảnh màu
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
        if depth_colormap.shape != annotated_frame.shape:
            annotated_frame = cv2.resize(annotated_frame, (depth_colormap.shape[1], depth_colormap.shape[0]))

        images = np.hstack((annotated_frame, depth_colormap))
        cv2.imshow('YOLO + Deep SORT + RealSense', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
