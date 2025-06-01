import cv2
import numpy as np

# Load the two images (first and second frames)
img1 = cv2.imread('Portrait_1.jpg')  # First image
img2 = cv2.imread('Portrait_2.jpg')  # Second image

# Convert first image to grayscale for detection
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Initialize Haar Cascade face detector (autofocus equivalent)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Detect faces in the first image
faces = face_cascade.detectMultiScale(
    gray1,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Exit if no face is found
if len(faces) == 0:
    print("No face detected in the first image.")
    exit()

# Use the first detected face bounding box
x, y, w, h = faces[0]

# Draw a rectangle around the detected face
cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draw a circle around the detected face (center and radius)
center = (x + w // 2, y + h // 2)
radius = max(w, h) // 2
cv2.circle(img1, center, radius, (255, 0, 0), 2)

# Prepare for feature matching: crop the face region
face_roi = gray1[y : y + h, x : x + w]

# Initialize ORB detector for keypoints and descriptors
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(face_roi, None)

# Convert second image to grayscale for descriptor computation
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Match descriptors using BFMatcher with Hamming distance (for ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda m: m.distance)

# Use top N matches to compute homography
N_MATCHES = 10
good_matches = matches[:N_MATCHES]

# Extract matched keypoints' coordinates
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# Adjust coordinates relative to the full image (face_roi offset)
pts1 += np.array([[x, y]], dtype=float).reshape(-1, 1, 2)

pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute homography matrix mapping points from img1 to img2
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# Define the rectangle corners around the face in img1
rect_corners = np.float32(
    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
).reshape(-1, 1, 2)

# Map the rectangle corners to the second image using homography
mapped_corners = cv2.perspectiveTransform(rect_corners, H)

# Draw the mapped polygon on img2
img2_mapped = img2.copy()
pts = np.int32(mapped_corners)
cv2.polylines(img2_mapped, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

# Display results
cv2.imshow('First Image - Detected Subject', img1)
cv2.imshow('Second Image - Mapped Subject', img2_mapped)
cv2.waitKey(0)
cv2.destroyAllWindows()
