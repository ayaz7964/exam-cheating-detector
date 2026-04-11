from ultralytics import YOLO

model = YOLO("../model/best.pt")

metrics = model.val()

print("\n📊 Evaluation Results:")
print(metrics)