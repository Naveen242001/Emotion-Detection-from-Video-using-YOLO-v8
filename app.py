from flask import Flask, request, render_template, redirect, url_for, jsonify
import mysql.connector
import os
import cv2
# from tasks import process_video

app = Flask(__name__)

# MySQL Config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'avinash'
app.config['MYSQL_DB'] = 'test_database'

# Directories for videos
UPLOAD_FOLDER = 'static/videos'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Establishing connection
connection = mysql.connector.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB']
)
# Check if connection is successful
if connection.is_connected():
    print("Connection established successfully!")



@app.route('/')
def index():

    # Establishing connection
    connection = mysql.connector.connect(
        host=app.config['MYSQL_HOST'],
        user=app.config['MYSQL_USER'],
        password=app.config['MYSQL_PASSWORD'],
        database=app.config['MYSQL_DB']
    )
    # Check if connection is successful
    if connection.is_connected():
        print("Connection established successfully!")

    # creating the cursor object
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM videos")
    videos = cursor.fetchall()
    cursor.close()
    return render_template('index.html', videos=videos, response=True)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No video file uploaded!', 400
    
    file = request.files['video']
    if file.filename == '':
        return 'No selected file!', 400
    
    filepath = UPLOAD_FOLDER+ "/" + file.filename
    print("File Path:", filepath)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = str(int(frame_count/fps)) + " seconds"
    
    # Add video entry to database
    cursor = connection.cursor()
    cursor.execute("INSERT INTO videos (filename, input_video_path, video_duration, status) VALUES (%s, %s, %s, %s)", (file.filename, filepath, duration, 'Processing'))
    connection.commit()
    # video_id = cursor.lastrowid
    cursor.close()

    # Trigger Celery task
    # process_video.delay(video_id, filepath)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)


