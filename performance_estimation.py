import pandas as pd
from sklearn.preprocessing import LabelEncoder
# import all the metrics we'll use later on
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score 

import matplotlib.pyplot as plt
import seaborn as sns

classes = ['neutral', 'happy', 'sad', 'surprise', 'afraid', 'disgust', 'anger', 'contemptuous']

data = pd.read_excel('results/AffectNet_test.xlsx')

ground_truth = data['Ground Truth']
model_prediction = data['System Prediction']

# Initialize LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(classes)

# Convert class names to categorical values
ground_truth_encoded = label_encoder.transform(ground_truth)
model_prediction_encoded = label_encoder.transform(model_prediction)

# Generate confusion matrix for the predictions
conf_matrix = confusion_matrix(ground_truth_encoded, model_prediction_encoded)
 
# plt.figure(figsize=(8,8))
# sns.set(font_scale = 1.5)
 
# ax = sns.heatmap(
#     conf_matrix, # confusion matrix 2D array 
#     annot=True, # show numbers in the cells
#     fmt='g', # show numbers as integers,
#     cbar=False, # don't show the color bar
#     cmap='flag', # customize color map
#     vmax=175 # to get better color contrast
# )
 
# ax.set_xlabel("Predicted", labelpad=20)
# ax.set_ylabel("Actual", labelpad=20)
# plt.show()

# Calculate metrics
precision = precision_score(ground_truth_encoded, model_prediction_encoded, average='weighted')
recall = recall_score(ground_truth_encoded, model_prediction_encoded, average='weighted')
f1 = f1_score(ground_truth_encoded, model_prediction_encoded, average='weighted')
accuracy = accuracy_score(ground_truth_encoded, model_prediction_encoded)
conf_matrix = confusion_matrix(ground_truth_encoded, model_prediction_encoded)

# Display metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")


# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=classes, yticklabels=classes)

# Add labels and title
plt.xlabel('Predicted Classes')
plt.ylabel('True Classes')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig('results/AffectNet_confusion_matrix.png', dpi=300, bbox_inches='tight')

# plt.show()