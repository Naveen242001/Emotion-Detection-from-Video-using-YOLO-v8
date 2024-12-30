import tensorflow as tf
import numpy as np
import cv2
from retinaface import RetinaFace
import os
import mysql.connector
from tqdm import tqdm

# MySQL Config
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'avinash'
DB_NAME = 'test_database'
PROCESSED_FOLDER = 'static/processed'
######################################################################################################
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

# print("Input:",infer.inputs)
# print("Output:",infer.outputs)

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
######################################################################################################
def process_video(video_id, filepath):
    try:
        print("Input File Name:",filepath)
        # Open the video file
        cap = cv2.VideoCapture(filepath)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = str(int(frame_count/fps)) + " seconds"
        file_name = os.path.basename(filepath).split(".")[0]+".mp4"
        processed_path = PROCESSED_FOLDER+  "/processed_"+ file_name
        tmp_path = PROCESSED_FOLDER+  "/temp.mp4"

        print("Output File :", processed_path)
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (frame_width, frame_height))

        pbar = tqdm(total=frame_count, position=0)
        # Process the video frame by frame

        emotions= []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform face detection
            faces = RetinaFace.detect_faces(frame)
            
            # Draw rectangles around detected faces
            if isinstance(faces, dict):  # If faces are detected
                for key, face in faces.items():
                    facial_area = face['facial_area']
                    x1, y1, x2, y2 = facial_area
                    emotion_name, probability = emotion_recognition(frame[y1:y2,x1:x2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                    cv2.putText(frame, emotion_name + " | " + probability, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    if emotion_name not in emotions:
                        emotions.append(emotion_name)
            # Write the processed frame to the output video
            out.write(frame)
            # pbar.total += 1
            pbar.update(1)
            # Display the frame (optional, for debugging purposes)
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        cmd = f"ffmpeg -i static/processed/temp.mp4 -vcodec libx264 -acodec aac {processed_path}"
        os.system(cmd)
        # Process video (e.g., create a grayscale version)
        # Update the database with processed video details
        # Establishing connection
        # connection = mysql.connector.connect(
        #     host=DB_HOST,
        #     user=DB_USER,
        #     password=DB_PASSWORD,
        #     database=DB_NAME
        # )
        cursor = connection.cursor()
        cursor.execute("UPDATE videos SET video_duration = %s, detected_emotions=%s, status = %s, processed_video_path = %s WHERE id = %s", (duration, ", ".join(emotions), 'Completed', processed_path, video_id))
        connection.commit()
        # cursor.close()
        # connection.close()

        print("Video ID", video_id)
        print("Filepath:", filepath)

    except Exception as e:
        # Update the status to failed if any error occurs
        # connection = mysql.connector.connect(
        #     host=DB_HOST,
        #     user=DB_USER,
        #     password=DB_PASSWORD,
        #     database=DB_NAME
        # )
        cursor = connection.cursor()
        cursor.execute("UPDATE videos SET status = %s WHERE id = %s", ('Failed', video_id))
        connection.commit()
        # cursor.close()
        # connection.close()
        print("Error...")
        raise e
######################################################################################################

processing_flag = []
connection = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)

while True:

    # creating the cursor object
    cursor = connection.cursor()

    # query string
    select_query = "SELECT * FROM videos WHERE status = 'Processing'"
    
    # execute query
    cursor.execute(select_query)

    # Fetching all results

    records = cursor.fetchall()

    for record in records:
        video_id  = record[0]
        video_path  = record[2]
        if video_id not in processing_flag:
            # print("Processing... Video ID ", video_id)
            processing_flag.append(video_id)
            process_video(video_id, video_path)

    



