# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:31:05 2024

@author: Admin
"""

# detect.py
import argparse
import datetime
import cv2
import imutils

# Tham số đầu vào
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Nguồn video mặc định 0 là sử dụng webcam", default='0', type=str)
ap.add_argument("-a", "--min-area", type=int, default=300, help="Kích thước nhỏ nhất của đối tượng")
ap.add_argument("-b", "--background", type=str, default=None, help="Ảnh background")
args = vars(ap.parse_args())

# Chọn nguồn video đầu vào
if args['video'] == '0':
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture(args['video'])
# Khởi tạo ảnh background nếu không chuyền vào thì ta lấy ảnh đầu tiên làm background
first_frame = None
if args['background'] is not None:
    first_frame = cv2.imread(args['background'])
    #  Chuyển ảnh đầu vào về ảnh xám
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # Làm mờ ảnh
    first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

idx = 0
#  Xác định vùng quan sát trên ảnh
top_left, bottom_right = (330, 50), (450, 280)
#  Lặp lần lượt từng frame của video
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    text = "An toan"
    # Resize ảnh về kích thước cố định
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # cv2.imshow("Current frame", frame)

    # Nếu tạo ảnh background nếu không được chuyền vào
    if first_frame is None:
        first_frame = frame
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

    # Tính sự khác biệt giữa ảnh hiện tại và ảnh background
    frameDelta = cv2.absdiff(first_gray, gray)
    # Sử dụng threshold để chuyển ảnh về dụng nhị phân
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # Thực hiện dãn nở ảnh để làm rõ các cùng vùng màu trắng trong ảnh
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Xác định đường bao cho các đối tượng thu được
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # Xác định vùng cần kiểm tra
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    for c in cnts:
        # Loại bỏ các đối tượng có kích thước nhỏ
        if cv2.contourArea(c) < args["min_area"]:
            continue
        # Lấy tọa độ của hình chữ nhật bao quanh đối tượng
        (x, y, w, h) = cv2.boundingRect(c)
        # Xác định tâm của đối tượng
        center_x = x + w / 2
        center_y = y + h / 2
        # Kiểm tra đối tượng có nằm trong khu vực quan sát hay không
        logic = top_left[0] < center_x < bottom_right[0] and top_left[1] < center_y < bottom_right[1]
        if logic:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = "Co xam nhap"
            # Hiện cảnh báo lên hình
            cv2.putText(frame, "Tinh trang: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # show the frame and record if the user presses a key
    cv2.imshow("Camera an ninh", frame)
    # cv2.imshow("Thresh", thresh)
    cv2.imwrite('../images/{}_result.jpg'.format(idx), frame)
    # cv2.imwrite('../images/{}_delta.jpg'.format(idx), frameDelta)
    # cv2.imshow("Frame Delta", frameDelta)
    idx += 1
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
cv2.destroyAllWindows()