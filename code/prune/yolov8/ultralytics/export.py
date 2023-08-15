from ultralytics import YOLO
import sys

# Load a model
model = YOLO("/home/DONG/PRUNE/ultralytics/runs/detect/train6/weights/best.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="VOC.yaml", epochs=100)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx", opset=13, half=True)  # export the model to ONNX format
success = model.export(format="onnx", simplify=True, opset=13, half=True) 
print(success)
