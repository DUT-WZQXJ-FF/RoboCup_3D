from ultralytics import YOLO

# 加载模型
model = YOLO(r"F:\mything\比赛代码\yolov8\best.pt")

# 导出ONNX，指定opset版本
model.export(
    format="onnx",       # 导出格式
    opset=13,            # 指定opset版本
)