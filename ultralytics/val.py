from ultralytics import YOLO

if __name__ == "__main__":
    # 加载训练好的模型
    model_path = r"F:\yolov8shoot\cls.pt"
    model = YOLO(model_path)

    # 验证模型性能
    metrics = model.val() # 自动加载验证集路径
    print("mAP50-95:", metrics.box.map) # 输出mAP50-95
    print("mAP50:", metrics.box.map50) # 输出mAP50