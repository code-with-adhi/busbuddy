import cv2
import time
import torch

# Load YOLOv5 model (change the model path if necessary)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to process the image
def detect_person(image):
    results = model(image)  # Run inference
    person_count = sum([1 for x in results.xyxy[0] if int(x[5]) == 0])  # class_id 0 is 'person'

    # Annotate image
    annotated_image = results.render()[0]  # Render the results on the image
    
    return annotated_image, person_count

#Capture images every 30 seconds
cap = cv2.VideoCapture(0)  # Change to your video source if needed

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        annotated_image, count = detect_person(frame)
        cv2.imshow("Detected Persons", annotated_image)
        print(f"Number of persons detected: {count}")

        # Wait for 30 seconds
        time.sleep(10)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    cap.release()
    cv2.destroyAllWindows()
