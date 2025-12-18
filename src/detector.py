import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(image):
    """
    Detect faces in an image using Haar Cascades
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )
    return faces

def detect_objects(image):
    """
    Detect objects using contour detection
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    return boxes

def draw_boxes(image, boxes, label):
    """
    Draw green boxes, labels, and total count
    """
    img = image.copy()

    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        # Green bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label (Face-1, Object-1, etc.)
        text = f"{label}-{idx}"
        cv2.putText(
            img,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Total count text
    total_text = f"Total {label}s: {len(boxes)}"
    cv2.putText(
        img,
        total_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    return img
