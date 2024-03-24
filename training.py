from ultralytics import YOLO

# yolo model creation
model = YOLO("yolo-weights/yolov8l.pt")
model.train(data="coco128.yaml", imgsz=320, batch=4, epochs=20, workers=0)