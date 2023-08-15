from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/home/DONG/PRUNE/ultralytics/runs/detect/train6/weights/best.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="XX.yaml", epochs=100, batch=32)  # train the model

metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

