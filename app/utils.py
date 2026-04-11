def predict_frame(model, frame):
    results = model.predict(frame, verbose=False)

    probs = results[0].probs
    label_index = probs.top1
    confidence = probs.top1conf.item()

    label = results[0].names[label_index]

    return label, confidence