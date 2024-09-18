from ultralytics import YOLOv10
#model = YOLOv10("runs/detect/train8/weights/best.pt")
import os
os.environ['WANDB_MODE'] = 'disabled'
model = YOLOv10("weights/best_finetuned_9c.pt")
model.train(
    data='datasets/SailbotVT-4/data.yaml', device="cuda:0", epochs=10, batch=10, imgsz=640, dropout=0.05, 
    freeze=15, crop_fraction=0., warmup_epochs=0.5, erasing=0.1, hsv_s=0.5, hsv_v=0.2, fliplr=1, flipud=1, mosaic=0.
)

