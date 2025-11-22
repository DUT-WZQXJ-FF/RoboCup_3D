from ultralytics import YOLO

# Load a COCO-pretrained YOLO12n model
model = YOLO(r"F:\mything\比赛代码\yolov8\model\office\yolo11x.pt")

if __name__ == '__main__':
# Fine-tune on your detection dataset
  results = model.train(
      data=r"F:\mything\比赛代码\yolov8\data\all\data.yaml",  # Detection dataset
      epochs=1000,
      batch=8,
      patience=30,
      name="yolo",  
      save_period=100,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
      hsv_s=0.1,  
      hsv_h=0.4, 
      degrees=0.0,  
      scale=0.1,  
      mosaic=0.0,  
      erasing=0.0,   
      cls=10, 
      dfl=1,
      box=3,
      amp=False,
      optimizer='SGD',
      workers=1,
  )