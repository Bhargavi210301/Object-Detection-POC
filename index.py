from flask import Flask, request, jsonify, send_file
import torch
from PIL import Image, ImageDraw
import os
from ultralytics import YOLO

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model1 = YOLO('./best.pt')

@app.route('/predict_combined', methods=['POST'])
def predict_combined():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    img = Image.open(image_path)

    # Model 1 (YOLOv5) predictions
    results_v5 = model(img)
    detected_objects_v5 = results_v5.xyxy[0][:, -1].cpu().numpy().astype(int)
    detected_object_names_v5 = [results_v5.names[i] for i in detected_objects_v5]
    boxes_v5 = results_v5.xyxy[0][:, :4].cpu().numpy()

    draw = ImageDraw.Draw(img)
    for box, name in zip(boxes_v5, detected_object_names_v5):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), name, fill="red")

    # Model 2 (Custom YOLO model) predictions
    results_custom = model1.predict(source=image_path, save=False)
    
    detected_objects_custom = []
    boxes_custom = results_custom[0].boxes  # Get boxes
    for box in boxes_custom.data:  # Access the underlying data
        if box.size(0) < 6:
            continue  # Skip if not enough values

        x1, y1, x2, y2, conf, class_id = box.cpu().numpy()  # Unpack values
        if conf > 0.5:  # Confidence threshold
            detected_object_name = results_custom[0].names[int(class_id)]
            detected_objects_custom.append(detected_object_name)

            # Draw the bounding box and label
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            draw.text((x1, y1), f"{detected_object_name} ({conf:.2f})", fill="blue")

    output_image_path = os.path.join('uploads', 'combined_result_' + file.filename)
    img.save(output_image_path)

    os.remove(image_path)

    # Combine detected objects from both models
    combined_detected_objects = list(set(detected_object_names_v5 + detected_objects_custom))

    response = {
        "detected_objects": combined_detected_objects,
        "image_path": f"/uploads/combined_result_{file.filename}"
    }

    return jsonify(response)

@app.route('/uploads/<filename>', methods=['GET'])
def get_image(filename):
    return send_file(os.path.join('uploads', filename), mimetype='image/jpeg')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)