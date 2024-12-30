
import cv2
import numpy as np
import os

import tensorflow as tf
import pandas as pd

from tqdm import tqdm

classes = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Afraid', 'Disgust', 'Anger', 'Contemptuous']
# Define the target dimensions
target_width = 224
target_height = 224

# Load the SavedModel
saved_model_path = 'saved_model'
model = tf.saved_model.load(saved_model_path)
# model = tf.keras.models.load_model(saved_model_path)

# Access the model's signatures
signatures = model.signatures

# Print available signature keys
# print("Available signatures:", list(signatures.keys()))

# Access the inference function
infer = model.signatures['serving_default']

def emotion_recognition(img):

    # Resize the image
    resized_image = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Convert to float32
    resized_image = np.asarray([resized_image]).astype(np.float32)

    # Normalize pixel values to [0, 1]
    resized_image /= 255.0

    # print("Input Image Shape:", resized_image.shape)

    input_tensor = tf.convert_to_tensor(resized_image)

    # print(input_tensor.shape)
    # Perform inference
    outputs = infer(input_tensor)
    # print(outputs['output'])
    predictions = outputs['output'].numpy()  

    pr = str(np.max(predictions)*100)[0:4]+"%"
    class_id = np.argmax(predictions)
    # Print predictions
    # print(predictions[0])
    # print("Predicted class:", classes[class_id], class_id)
    # print("Probability:",pr)


    return classes[class_id], pr


directory_path = 'dataset/AffectNet_test/'
ground_truth = os.listdir(directory_path)

# print(ground_truth)
file_paths = []
ground_truths = []
system_predictions = []
predicted_probabilities = []

for gt_class_name in ground_truth:
    class_folder_path = directory_path + gt_class_name
    population = os.listdir(class_folder_path)
    print("Ground Truth:", gt_class_name.lower())

    pbar = tqdm(total=len(population), position=0)

    for file_name in population:
        file_path = class_folder_path + "/" + file_name
        # print(file_path)
        file_paths.append(file_path)

        ground_truth_lower = gt_class_name.lower()
        # we are changing fear to afraid for dataset AffectNet
        if gt_class_name.lower() == 'fear':
            ground_truth_lower = 'afraid'
            
        ground_truths.append(ground_truth_lower)

        # calling emotion recognition
        img = cv2.imread(file_path)
        emotion_name, probability = emotion_recognition(img)

        system_predictions.append(emotion_name.lower())
        predicted_probabilities.append(probability)

        pbar.update(1)
        # break

# store it into a disctionary
model_predictions = {'File Path':file_paths, 'Ground Truth': ground_truths , 'System Prediction': system_predictions, ' Predicted Probability':predicted_probabilities}

data = pd.DataFrame.from_dict(model_predictions)

data.to_excel("results/AffectNet_test.xlsx", index=False) 


