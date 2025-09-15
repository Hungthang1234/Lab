import time
import cv2
import argparse
import numpy as np
import imutils

# Cai dat tham so doc weight, config va class name
ap = argparse.ArgumentParser()
ap.add_argument('-v', "--video", default='0',
                help='Nguồn video mặc định 0 là sử dụng webcam')

args = vars(ap.parse_args())

# Ham tra ve output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# Ham ve cac hinh chu nhat va ten class
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Doc tu webcam

# Chọn nguồn video đầu vào
if args['video'] == '0':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args['video'])

frame_id = 0
starting_time = time.time()

# Doc ten cac class
classes = None
with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')


# Bat dau doc tu webcam
i=1
while (True):

    # Doc frame
    ret, frame = cap.read()
    frame_id += 1
    image = imutils.resize(frame, width=600)
    i+=1
    if i%10==0:
        # Resize va dua khung hinh vao mang predict
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        # Loc cac object trong khung hinh
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (confidence > 0.5):
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Ve cac khung chu nhat quanh doi tuong
        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            draw_prediction(image, class_ids[i], round(x), round(y), round(x + w), round(y + h))

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(image, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.imshow("object detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()