from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg-pf.pt")

# Run prediction. No prompts required.
results = model.predict(r"F:\mything\比赛代码\yolov8\data\all\images\train\0-0-0-0-0-1-180121.png",
                        save=True,
                        show_labels=False,)

# Show results
results[0].show()