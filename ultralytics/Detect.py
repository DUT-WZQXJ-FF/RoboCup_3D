import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'F:\mything\比赛代码\yolov8\best.pt') # select your model.pt path
    model.predict(source=r'F:\mything\比赛代码\yolov8\data\all\images\test',
                  task='detect',
                  imgsz=640,
                  project='runs/杨',
                  name='3-0-0',
                  save=True,
                  device=0,  # 选择GPU设备
                  show_labels=False,
                  conf=0.3,
                  save_txt=True,
                  iou=0.7,
                  # classes=0, 是否指定检测某个类别.
                )