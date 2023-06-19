import cv2
from fer import FER

# Initialize the FER detector
detector = FER(mtcnn=True)

# Open a video capture object for the default camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If frame reading is not successful, break the loop
    if not ret:
        print("Failed to capture frame")
        break

    # Detect faces and classify emotions in the frame
    result = detector.detect_emotions(frame)

    # Process the detected faces and emotions
    for face in result:
        x, y, width, height = face["box"]
        emotions = face["emotions"]

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display the dominant emotion label on the face
        dominant_emotion = max(emotions, key=emotions.get)
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Facial Expression Recognition para el MINEDU", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
