
import cv2
import pyttsx3
from ultralytics import YOLO
from color_detection import get_color_palette, get_color_name

# Load the model (update with your traine/Users/prad model path)
model = YOLO("/Users/prajjwal/Library/CloudStorage/OneDrive-UniversityofAppliedSciencesEuropeGmbH(ehem.BiTSbtk)-Berlin,Hamburg,Iserlohn/Final Thesis/Final/Prototype/best-22.pt")  # Replace with your model path

# Text-to-speech engine
engine = pyttsx3.init()

# Configure speech properties
engine.setProperty('rate', 165)  # Slow speech for clarity
engine.setProperty('volume', 0.9)  # Set volume (0.0 to 1.0)

# Select a female voice (Samantha, Voice 132)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[132].id)  # Samantha (en-US)

# Initialize webcam (iPhone camera on index 2)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam on index 2. Check if the iPhone camera is connected.")
    exit()

# Create OpenCV window
cv2.namedWindow("Assistive Object + Color Detection", cv2.WINDOW_NORMAL)

# Show live feed and wait for spacebar to capture
print("Live webcam feed active. Press spacebar to capture and process a frame, or 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Check webcam connection.")
        break

    # Show live feed
    cv2.imshow("Assistive Object + Color Detection", frame)

    # Wait for key press (50ms for smooth display)
    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):  # Quit on 'q'
        print("Quitting program.")
        break
    elif key == ord(' '):  # Capture and process on spacebar
        print("Processing frame...")
        processed_frame = frame.copy()
        
        # Perform object detection
        results = model(processed_frame)
        detected = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = processed_frame[y1:y2, x1:x2]

                # Get color palette
                color_names = get_color_palette(crop, k=3)
                if not color_names:
                    color_names = [get_color_name(get_dominant_colors(crop, k=1)[0])]
                label = f"{', '.join(color_names)} {class_name}"
                print("Detected:", label)

                # Draw rectangle and label
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                detected = True

        # Display processed frame
        cv2.imshow("Assistive Object + Color Detection", processed_frame)

        # Provide speech feedback
        if detected:
            for r in results:  # Re-loop to speak all detections
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = processed_frame[y1:y2, x1:x2]
                    color_names = get_color_palette(crop, k=3)
                    if not color_names:
                        color_names = [get_color_name(get_dominant_colors(crop, k=1)[0])]
                    speech_text = f"A {class_name} with {', '.join(color_names)} colors"
                    engine.say(speech_text)
                    engine.runAndWait()  # Wait for speech to complete
        else:
            print("No objects detected in this frame.")
            engine.say("No objects detected")
            engine.runAndWait()

        print("Press any key to exit.")
        cv2.waitKey(0)  # Wait for any key to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()