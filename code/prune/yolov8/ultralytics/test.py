from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/home/DONG/PRUNE/ultralytics/runs/detect/train4/weights/best.pt")  # load a pretrained model (recommended for training)

# Use the model
model.val(data="XX.yaml", batch=64)  

print("success")
