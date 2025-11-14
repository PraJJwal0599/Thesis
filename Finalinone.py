import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import webcolors
import pyttsx3
from ultralytics import YOLO

# Expanded fashion-relevant color database with RGB values
COLOR_NAMES = {
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 128, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'brown': (139, 69, 19),
    'orange': (255, 165, 0),
    'pink': (255, 182, 193),
    'purple': (128, 0, 128),
    'navy': (0, 0, 128),
    'lime': (50, 205, 50),
    'maroon': (128, 0, 0),
    'coral': (255, 127, 127),
    'teal': (0, 128, 128),
    'olive': (128, 128, 0),
    'beige': (245, 245, 220),
    'turquoise': (64, 224, 208),
    'lavender': (230, 230, 250),
    'gold': (255, 215, 0),
    'khaki': (240, 230, 140),
    'indigo': (75, 0, 130),
    'salmon': (250, 128, 114),
    'crimson': (220, 20, 60),
    'violet': (238, 130, 238),
    'charcoal': (54, 69, 79),
    'ivory': (255, 255, 240),
    'mint': (189, 252, 201),
    'plum': (221, 160, 221)
}

def preprocess_image(image):
    """Preprocess image for robust color detection in HSV space."""
    # Convert to HSV for better color perception
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split HSV channels
    h, s, v = cv2.split(img_hsv)
    
    # Apply adaptive histogram equalization to V channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    # Merge back and convert to BGR
    img_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img

def get_dominant_colors(image, k=3):
    """Extract k dominant colors using KMeans in HSV space."""
    # Preprocess image
    img = preprocess_image(image)
    
    # Convert to HSV for clustering
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Resize for efficiency
    img_hsv = cv2.resize(img_hsv, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Reshape for KMeans
    pixels = img_hsv.reshape((-1, 3))
    
    # Filter out low-saturation and extreme brightness pixels
    saturation = pixels[:, 1]
    pixels = pixels[saturation > 20]
    value = pixels[:, 2]
    pixels = pixels[(value > 30) & (value < 230)]
    
    # Fallback if too few pixels
    if len(pixels) < k:
        return [(0, 0, 0)] * k
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get cluster centers and sort by frequency
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors = [tuple(map(int, kmeans.cluster_centers_[i])) for i in sorted_indices]
    
    # Convert HSV to RGB for naming
    dominant_colors_rgb = []
    for hsv_color in dominant_colors:
        hsv_img = np.array([[hsv_color]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)[0][0]
        dominant_colors_rgb.append(tuple(map(int, rgb_color)))
    
    # Pad with black if needed
    while len(dominant_colors_rgb) < k:
        dominant_colors_rgb.append((0, 0, 0))
    
    return dominant_colors_rgb[:k]

def get_color_name(rgb_tuple):
    """Match RGB to closest fashion-relevant color name."""
    try:
        color_name = webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        min_dist = float("inf")
        color_name = "unknown"
        for name, rgb in COLOR_NAMES.items():
            dist = distance.euclidean(np.array(rgb), np.array(rgb_tuple))
            if dist < min_dist:
                min_dist = dist
                color_name = name
    return color_name

def get_color_palette(image, k=3):
    """Return a list of color names for the dominant colors."""
    dominant_colors = get_dominant_colors(image, k)
    color_names = [get_color_name(color) for color in dominant_colors]
    seen = set()
    color_names = [x for x in color_names if not (x in seen or seen.add(x))]
    return color_names if color_names else ["unknown"]

def main():
    # Load YOLO model
    model_path = "/Users/prajjwal/Library/CloudStorage/OneDrive-UniversityofAppliedSciencesEuropeGmbH(ehem.BiTSbtk)-Berlin,Hamburg,Iserlohn/Final Thesis/Final/Prototype/best-22.pt"
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Initialize text to speech engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)
    engine.setProperty('volume', 0.8)
    voices = engine.getProperty('voices')
    try:
        engine.setProperty('voice', voices[132].id)  # Samantha (en-US) as found to be very similar to Siri.
    except IndexError:
        print("Voice index 132 was not found. Now, using default voice.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam on index 1. Check if the iPhone camera is connected.")
        return
    
    # Create OpenCV window
    cv2.namedWindow("Assistive Object + Color Detection", cv2.WINDOW_NORMAL)
    print("Live webcam feed active. Press spacebar to process a frame, or 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Check webcam connection.")
            break
        
        # Show live feed
        cv2.imshow("Assistive Object + Color Detection", frame)
        
        # Wait for key press
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            print("Quitting program.")
            break
        elif key == ord(' '):
            print("Processing frame...")
            processed_frame = frame.copy()
            detected = False
            
            # Perform YOLO detection
            results = model(processed_frame)
            speech_texts = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Validate bounding box
                    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                        continue
                    
                    # Crop ROI
                    crop = processed_frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    # Get color palette
                    color_names = get_color_palette(crop, k=3)
                    
                    # Prepare label and speech
                    label = f"{', '.join(color_names)} {class_name}"
                    speech_text = f"A {class_name} with {', '.join(color_names)} colors"
                    print(f"Detected: {label}")
                    speech_texts.append(speech_text)
                    
                    # Draw rectangle and label
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    detected = True
            
            # Display processed frame
            cv2.imshow("Assistive Object + Color Detection", processed_frame)
            
            # Provide speech feedback
            if detected:
                for speech_text in speech_texts:
                    engine.say(speech_text)
                    engine.runAndWait()
            else:
                print("No objects detected in this frame.")
                engine.say("No objects detected")
                engine.runAndWait()
            
            # Wait briefly to allow user to view results
            cv2.waitKey(5000)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()