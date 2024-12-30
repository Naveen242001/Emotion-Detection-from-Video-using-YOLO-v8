CREATE TABLE videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    input_video_path VARCHAR(255) DEFAULT NULL,
    video_duration VARCHAR(50) DEFAULT NULL,
    processed_video_path VARCHAR(255) DEFAULT NULL,
    status ENUM('Processing', 'Completed', 'Failed') NOT NULL,
    detected_emotions VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
