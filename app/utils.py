# def predict_frame(model, frame):
#     results = model.predict(frame, verbose=False)

#     probs = results[0].probs
#     label_index = probs.top1
#     confidence = probs.top1conf.item()

#     label = results[0].names[label_index]

#     return label, confidence



def predict_frame(model, frame):
    results = model(frame)[0]   # get first result

    detections = []

    # Loop through detected boxes
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = results.names[cls]

            detections.append({
                "label": label,
                "confidence": conf,
                "box": (x1, y1, x2, y2)
            })

    return detections