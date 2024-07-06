import os
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import datetime
from transformers import pipeline

app = Flask(__name__)

# Model paths
model1_path = os.path.join('.', 'runs', 'detect', 'Model-Mix 1', 'train2', 'weights', 'best.pt')
model2_path = os.path.join('.', 'runs', 'detect', 'Model-Vegetables', 'train4', 'weights', 'best.pt')

# Load the YOLO models
model1 = YOLO(model1_path)
model2 = YOLO(model2_path)

# Threshold for object detection confidence
threshold = 0.5

# Class names dictionary for Model 1
class_names = {
    0: 'almond', 1: 'apple', 2: 'asparagus', 3: 'avocado', 4: 'baking powder', 5: 'baking soda', 6: 'banana',
    7: 'beef', 8: 'beet', 9: 'bell pepper', 10: 'bread', 11: 'broccoli', 12: 'butter', 13: 'cabbage', 14: 'carrot',
    15: 'cauliflower', 16: 'celery', 17: 'cheese', 18: 'chicken', 19: 'chicken breast', 20: 'cilantro', 21: 'cinnamon',
    22: 'corn', 23: 'cream', 24: 'cucumber', 25: 'egg', 26: 'eggplant', 27: 'flour', 28: 'garlic', 29: 'ginger',
    30: 'grapes', 31: 'grated_cheese', 32: 'green onion', 33: 'jam', 34: 'ketchup', 35: 'lemon', 36: 'lettuce',
    37: 'lime', 38: 'mango', 39: 'mayonnaise', 40: 'milk', 41: 'mustard', 42: 'nuts', 43: 'oil', 44: 'onion',
    45: 'orange', 46: 'paprika', 47: 'parsley', 48: 'pasta', 49: 'peanutbutter', 50: 'pear', 51: 'peas',
    52: 'pineapple', 53: 'plum', 54: 'potato', 55: 'rice', 56: 'rosemary', 57: 'salmon', 58: 'sausage',
    59: 'shrimp', 60: 'soy sauce', 61: 'spaghetti', 62: 'spinach', 63: 'strawberries', 64: 'sugar',
    65: 'sweetpotato', 66: 'tomato', 67: 'tomato sauce', 68: 'vanilla_extract', 69: 'watermelon', 70: 'whole_chicken',
    71: 'yogurt'
}

# Class names dictionary for Model 2
model2_class_names = {
    0: 'avocado', 1: 'beans', 2: 'beet', 3: 'bell pepper', 4: 'broccoli', 5: 'brus capusta', 6: 'cabbage',
    7: 'carrot', 8: 'cauliflower', 9: 'celery', 10: 'corn', 11: 'cucumber', 12: 'eggplant', 13: 'fasol',
    14: 'garlic', 15: 'hot pepper', 16: 'onion', 17: 'peas', 18: 'potato', 19: 'pumpkin', 20: 'rediska',
    21: 'redka', 22: 'salad', 23: 'squash-patisson', 24: 'tomato', 25: 'vegetable marrow'
}

# Load the recipe generation model
model_path = "recipe_gen_model"
pl = pipeline(task='text-generation', model=model_path)

# Directory for saving captured images
save_dir = os.path.join('Images', 'Saved')

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Start video capture from webcam
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def create_prompt(ingredients):
    ingredients = ','.join([x.strip().lower() for x in ingredients.split(',')])
    ingredients = ingredients.strip().replace(',', '\n')
    s = f"Ingredients:\n{ingredients}\n"
    return s

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = cap.read()
    if success:
        # Flip the frame horizontally back to its original orientation
        frame = cv2.flip(frame, 1)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_path = os.path.join(save_dir, f"captured_image_{timestamp}.jpg")

        cv2.imwrite(image_path, frame)
        if os.path.exists(image_path):
            # Replace backslashes with forward slashes
            image_path = image_path.replace('\\', '/')
            return redirect(url_for('predict', image_path=image_path))
        else:
            return "Failed to save image", 500
    return "Failed to capture image", 500

@app.route('/predict')
def predict():
    image_path = request.args.get('image_path')
    if not os.path.exists(image_path):
        return "Image not found", 404

    captured_image = cv2.imread(image_path)

    # First model prediction
    results1 = model1(captured_image)[0]
    detection_mask = np.zeros(captured_image.shape[:2], dtype=np.uint8)
    class_names_detected = set()

    for result in results1.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(captured_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            class_name = class_names.get(int(class_id), "Unknown")
            cv2.putText(captured_image, class_name, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            detection_mask[int(y1):int(y2), int(x1):int(x2)] = 1
            class_names_detected.add(class_name)

    # Invert mask to get undetected regions
    detection_mask_inv = cv2.bitwise_not(detection_mask)

    # Second model prediction
    results2 = model2(captured_image)[0]

    for result in results2.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold and np.mean(detection_mask_inv[int(y1):int(y2), int(x1):int(x2)]) > 0:
            cv2.rectangle(captured_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            class_name = model2_class_names.get(int(class_id), "Unknown")
            cv2.putText(captured_image, class_name, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            class_names_detected.add(class_name)

    predicted_image_path = image_path.replace('.jpg', '_predicted.jpg')
    predicted_image_path = predicted_image_path.replace('\\', '/')
    cv2.imwrite(predicted_image_path, captured_image)

    class_names_detected = list(class_names_detected)

    return render_template('predict.html', original_image=image_path, predicted_image=predicted_image_path,
                           class_names_detected=class_names_detected)


@app.route('/generate_recipes', methods=['POST'])
def generate_recipes():
    ingredients = request.form.get('ingredients')
    if not ingredients:
        return "No ingredients provided", 400

    prompt = create_prompt(ingredients)

    # Generate 6 recipes
    generated_texts = pl(prompt, max_new_tokens=512, penalty_alpha=0.6, top_k=4, num_return_sequences=6)

    recipes = []
    for generated_text in generated_texts:
        generated_text = generated_text['generated_text']
        lines = generated_text.split('Instructions:')
        ingredients_text = lines[0].strip().replace("Ingredients:", "Ingredients:\n").replace(',', ',\n')
        instructions_text = lines[1].strip() if len(lines) > 1 else ""

        steps = [step.strip() for step in instructions_text.split('.') if step.strip()]

        # Join steps into a single string with line breaks
        instructions_formatted = "Instructions:\n" + "\n".join(steps)

        recipes.append({
            'ingredients_text': ingredients_text,
            'instructions_text': instructions_formatted
        })

    return render_template('recipes.html', recipes=recipes)

@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/<filename>')
def serve_image(filename):
    return send_from_directory(save_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
