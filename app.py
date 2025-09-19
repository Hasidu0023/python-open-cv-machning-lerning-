import numpy as np
import cv2
import os
import urllib.request

# Configuration options
thres = 0.5  # Threshold to detect object
nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress, 0.1 means high suppress 
show_confidence = True
draw_bbox = True
flip_camera = False
use_colored_bbox = True
bbox_color = (0, 255, 0)
min_confidence = 0.3

# Download required files if they don't exist
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete!")
    else:
        print(f"{filename} already exists.")

# Download model files
model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
config_url = "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt"

download_file(model_url, "frozen_inference_graph.pb")
download_file(config_url, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

# Create objects.txt if it doesn't exist
if not os.path.exists("objects.txt"):
    print("Creating objects.txt with COCO class names...")
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]
    
    with open("objects.txt", "w") as f:
        for class_name in coco_classes:
            f.write(f"{class_name}\n")
    print("objects.txt created successfully!")

# Initialize camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
cap.set(cv2.CAP_PROP_CONTRAST, 50)
cap.set(cv2.CAP_PROP_SATURATION, 50)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# Verify camera settings
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {actual_width}x{actual_height}")

# Load class names
classNames = []
with open('objects.txt', 'r') as f:
    classNames = f.read().splitlines()
print(f"Loaded {len(classNames)} class names")

font = cv2.FONT_HERSHEY_PLAIN
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

# Load the model
weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# Check if model files exist
if not os.path.exists(weightsPath) or not os.path.exists(configPath):
    print("Error: Model files not found. Please check the file paths.")
    exit()

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Function to handle keyboard input
def handle_keypress(key):
    global thres, nms_threshold, show_confidence, draw_bbox, flip_camera, use_colored_bbox, min_confidence
    
    if key == ord('1'):
        thres = max(0.1, thres - 0.05)
        print(f"Detection threshold decreased to: {thres:.2f}")
    elif key == ord('2'):
        thres = min(0.9, thres + 0.05)
        print(f"Detection threshold increased to: {thres:.2f}")
    elif key == ord('3'):
        show_confidence = not show_confidence
        status = "ON" if show_confidence else "OFF"
        print(f"Confidence display: {status}")
    elif key == ord('4'):
        draw_bbox = not draw_bbox
        status = "ON" if draw_bbox else "OFF"
        print(f"Bounding boxes: {status}")
    elif key == ord('5'):
        flip_camera = not flip_camera
        status = "ON" if flip_camera else "OFF"
        print(f"Camera flip: {status}")
    elif key == ord('6'):
        use_colored_bbox = not use_colored_bbox
        status = "Colored" if use_colored_bbox else "Single Color"
        print(f"Bounding box mode: {status}")
    elif key == ord('7'):
        min_confidence = max(0.1, min_confidence - 0.1)
        print(f"Minimum confidence decreased to: {min_confidence:.2f}")
    elif key == ord('8'):
        min_confidence = min(0.9, min_confidence + 0.1)
        print(f"Minimum confidence increased to: {min_confidence:.2f}")
    elif key == ord('+') or key == ord('='):
        current = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, min(100, current + 10))
        print(f"Brightness increased to: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    elif key == ord('-') or key == ord('_'):
        current = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, max(0, current - 10))
        print(f"Brightness decreased to: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    elif key == ord('h') or key == ord('?'):
        print("\n=== Controls ===")
        print("1/2: Decrease/Increase detection threshold")
        print("3: Toggle confidence display")
        print("4: Toggle bounding boxes")
        print("5: Flip camera horizontally")
        print("6: Toggle colored bounding boxes")
        print("7/8: Decrease/Increase minimum confidence")
        print("+/-: Adjust brightness")
        print("h or ?: Show this help menu")
        print("q: Quit application")
        print("=================\n")

# Display help menu at start
handle_keypress(ord('h'))

print("Starting object detection. Point objects at the camera...")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
        
    # Flip image if enabled
    if flip_camera:
        img = cv2.flip(img, 1)
    
    # Apply some image processing to improve quality
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply slight sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    
    # Detect objects
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    
    # Debug information
    if len(classIds) == 0:
        print("No objects detected. Try adjusting the detection threshold with keys 1/2.")
    else:
        print(f"Detected {len(classIds)} objects")
    
    if len(classIds) > 0:
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        
        if len(indices) > 0:
            # Handle different return formats from NMSBoxes
            if isinstance(indices, tuple):
                indices = indices[0]
            indices = indices.flatten()
            
            for i in indices:
                if i >= len(confs) or i >= len(bbox):
                    continue
                    
                # Check if confidence meets minimum threshold
                if confs[i] < min_confidence:
                    continue
                    
                box = bbox[i]
                class_id = int(classIds[i]) - 1
                
                if class_id < 0 or class_id >= len(classNames):
                    continue
                
                # Choose color based on setting
                if use_colored_bbox:
                    color = Colors[class_id]
                else:
                    color = bbox_color
                    
                confidence = str(round(confs[i], 2))
                x, y, w, h = box[0], box[1], box[2], box[3]
                
                # Draw bounding box if enabled
                if draw_bbox:
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
                
                # Prepare text label
                if show_confidence:
                    label = f"{classNames[class_id]} {confidence}"
                else:
                    label = classNames[class_id]
                    
                # Draw text background for better visibility
                text_size = cv2.getTextSize(label, font, 1, 2)[0]
                cv2.rectangle(img, (x, y - text_size[1] - 5), (x + text_size[0] + 5, y), color, -1)
                
                # Draw text
                cv2.putText(img, label, (x + 5, y - 5), font, 1, (255, 255, 255), 2)

    # Display current settings on screen
    settings_text = f"Thres: {thres:.2f} | MinConf: {min_confidence:.2f} | Boxes: {'ON' if draw_bbox else 'OFF'}"
    cv2.putText(img, settings_text, (10, 30), font, 1, (0, 0, 255), 2)
    
    # Display brightness level
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    brightness_text = f"Brightness: {brightness:.1f}"
    cv2.putText(img, brightness_text, (10, 60), font, 1, (0, 0, 255), 2)
    
    # Display detection status
    status_text = f"Objects detected: {len(classIds) if 'classIds' in locals() else 0}"
    cv2.putText(img, status_text, (10, 90), font, 1, (0, 0, 255), 2)
    
    cv2.imshow("Enhanced Object Detection", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit application
        break
    else:
        handle_keypress(key)

cap.release()
cv2.destroyAllWindows()