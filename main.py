import dlib
import cv2
import numpy as np
import os

# Load the face detector and face recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Download from dlib

# Load multiple reference images of your face
def load_reference_images(folder_path):
    reference_encodings = []
    reference_images = []

    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = sp(gray, face)
                face_encoding = face_rec_model.compute_face_descriptor(image, landmarks)
                reference_encodings.append(np.array(face_encoding))  # Store encoding
                reference_images.append(image)  # Store the corresponding image for later

    return reference_encodings, reference_images

# Load your face reference images (Add your images to this folder)
reference_folder = r"C:\Users\saich\Documents\testing_detect_face\reference_faces"
reference_encodings, reference_images = load_reference_images(reference_folder)

# Function to remove non-ASCII characters from a string
def remove_non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)

# Create 'my_face' folder if it doesn't exist
my_face_folder = r"C:\Users\saich\Documents\testing_detect_face\my_face"
if not os.path.exists(my_face_folder):
    os.makedirs(my_face_folder)


# Create 'frame_face' folder if it doesn't exist
frame_face_folder = r"C:\Users\saich\Documents\testing_detect_face\frame_face"
if not os.path.exists(frame_face_folder):
    os.makedirs(frame_face_folder)

# Loop through your album of images
album_folder = r"C:\Users\saich\Documents\testing_detect_face\source_pic"
for image_name in os.listdir(album_folder):
    sanitized_image_name = remove_non_ascii(image_name)
    
    image_path = os.path.join(album_folder, image_name)  # Original path for loading
    sanitized_image_path = os.path.join(album_folder, sanitized_image_name)  # New path with sanitized name
    
    # Rename the file if the sanitized name is different
    if image_name != sanitized_image_name:
        os.rename(image_path, sanitized_image_path)  # Rename the file

    image = cv2.imread(sanitized_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    for face in faces:
        landmarks = sp(gray, face)
        face_encoding = face_rec_model.compute_face_descriptor(image, landmarks)
        
           # Compare the detected face with all reference encodings
        for reference_encoding, reference_image in zip(reference_encodings, reference_images):
            distance = np.linalg.norm(np.array(reference_encoding) - np.array(face_encoding))
            
            if distance < 0.5:  # Threshold for face recognition
                print(f"Your face is detected in {sanitized_image_name}")
                
                # Check if the image has already been saved in the "my_face" folder
                saved_image_path = os.path.join(my_face_folder, sanitized_image_name)
                if not os.path.exists(saved_image_path):
                    cv2.imwrite(saved_image_path, image)  # Save the image
                    print(f"Image saved as {sanitized_image_name} in 'my_face' folder.")
                else:
                    print(f"Image {sanitized_image_name} already exists in 'my_face' folder, skipping save.")


                                # Draw a frame around the face on the image
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 225), 2)  # Green rectangle
                
                # Save the image with a frame in 'frame_face' folder
                frame_image_path = os.path.join(frame_face_folder, sanitized_image_name)
                if not os.path.exists(frame_image_path):
                    cv2.imwrite(frame_image_path, image)  # Save the framed image
                    print(f"Framed image saved as {sanitized_image_name} in 'frame_face' folder.")
                else:
                    print(f"Framed image {sanitized_image_name} already exists in 'frame_face' folder, skipping save.")
                
                break  # Exit loop once a match is found
        else:
            print(f"Your face is not in {sanitized_image_name}") 