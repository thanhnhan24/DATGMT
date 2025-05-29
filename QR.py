import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Giải mã mã QR từ khung hình
    decoded_objects = decode(frame)
    for obj in decoded_objects:
        # In nội dung mã QR
        print("Phát hiện mã QR:", obj.data.decode("utf-8"))
        # Vẽ khung xung quanh mã QR
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = hull
        points = [point for point in points]
        for j in range(len(points)):
            cv2.line(frame, tuple(points[j]), tuple(points[(j + 1) % len(points)]), (0, 255, 0), 3)
    # Hiển thị khung hình
    cv2.imshow("QR Code Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
