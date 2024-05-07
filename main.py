from ultralytics import YOLO
import math
import cv2
import cvzone
import torch
from image_to_text import predict_number_plate
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from paddleocr import PaddleOCR

cap = cv2.VideoCapture("videos/22.mp4")  # For videos

model = YOLO("best.pt") # after training update the location of best.pt

device = torch.device("mps") # change to cuda for windows gpu or keep it as cpu

classNames = ["with helmet", "without helmet", "rider", "number plate"]
num = 0
old_npconf = 0

# grab the width, height, and fps of the frames in the video stream.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))


ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory

while True:
    success, img = cap.read()
    # Check if the frame was read successfully
    if not success:
        break
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device="mps")
    for r in results:
        boxes = r.boxes
        li = dict()
        rider_box = list()
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)
        try:
            new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
            # Get the indices of the rows where the value in column 1 is equal to 5.
            indices = torch.where(new_boxes[:, -1] == 2)
            # Select the rows where the mask is True.
            rows = new_boxes[indices]
            # Add rider details in the list
            for box in rows:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                rider_box.append((x1, y1, x2, y2))
        except:
            pass
        for i, box in enumerate(new_boxes):
            # Bounding box
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box[4] * 100)) / 100
            # Class Name
            cls = int(box[5])
            if classNames[cls] == "without helmet" and conf >= 0.5 or classNames[cls] == "rider" and conf >= 0.45 or \
                    classNames[cls] == "number plate" and conf >= 0.5:
                if classNames[cls] == "rider":
                    rider_box.append((x1, y1, x2, y2))
                if rider_box:
                    for j, rider in enumerate(rider_box):
                        if x1 + 10 >= rider_box[j][0] and y1 + 10 >= rider_box[j][1] and x2 <= rider_box[j][2] and \
                                y2 <= rider_box[j][3]:
                            # highlight or outline objects detected by object detection models
                            cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                            cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                               offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                            li.setdefault(f"rider{j}", [])
                            li[f"rider{j}"].append(classNames[cls])
                            if classNames[cls] == "number plate":
                                npx, npy, npw, nph, npconf = x1, y1, w, h, conf
                                crop = img[npy:npy + h, npx:npx + w]
                        if li:
                            for key, value in li.items():
                                if key == f"rider{j}":
                                    if len(list(set(li[f"rider{j}"]))) == 3:
                                        try:
                                            # crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) # for easy ocr
                                            vechicle_number, conf = predict_number_plate(crop, ocr)
                                            if vechicle_number and conf:
                                                cvzone.putTextRect(img, f"{vechicle_number} {round(conf*100, 2)}%",
                                                                   (x1, y1 - 50), scale=1.5, offset=10,
                                                                   thickness=2, colorT=(39, 40, 41),
                                                                   colorR=(105, 255, 255))
                                        except Exception as e:
                                            print(e)
        # Display the frame
        output.write(img)
        cv2.imshow('Video', img)
        li = list()
        rider_box = list()

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            output.release()
            break
