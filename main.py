import face_recognition
import cv2
import numpy as np
import os

# Load the reference images and their encodings
def load_reference_images(folder_path):
    reference_encodings = []
    reference_images = []

    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)

            if encoding:  # Ensure there is at least one face in the image
                reference_encodings.append(encoding[0])  # Store the first encoding
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

    # Load the image
    image = face_recognition.load_image_file(sanitized_image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    for face_encoding in face_encodings:
        # Compare the detected face with all reference encodings
        for reference_encoding, reference_image in zip(reference_encodings, reference_images):
            distance = np.linalg.norm(reference_encoding - face_encoding)
            
            if distance < 0.5:  # Threshold for face recognition
                print(f"Your face is detected in {sanitized_image_name}")
                
                # Check if the image has already been saved in the "my_face" folder
                saved_image_path = os.path.join(my_face_folder, sanitized_image_name)
                if not os.path.exists(saved_image_path):
                    cv2.imwrite(saved_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Save the image
                    print(f"Image saved as {sanitized_image_name} in 'my_face' folder.")
                else:
                    print(f"Image {sanitized_image_name} already exists in 'my_face' folder, skipping save.")

                # Draw a frame around the face on the image
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)  # Red rectangle
                
                # Save the image with a frame in 'frame_face' folder
                frame_image_path = os.path.join(frame_face_folder, sanitized_image_name)
                if not os.path.exists(frame_image_path):
                    cv2.imwrite(frame_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Save the framed image
                    print(f"Framed image saved as {sanitized_image_name} in 'frame_face' folder.")
                else:
                    print(f"Framed image {sanitized_image_name} already exists in 'frame_face' folder, skipping save.")
                
                break  # Exit loop once a match is found
        else:
            print(f"Your face is not in {sanitized_image_name}")

    # Remove the processed image to save disk space
    try:
        os.remove(sanitized_image_path)
        print(f"Deleted original image: {sanitized_image_name}")
    except Exception as e:
        print(f"Failed to delete {sanitized_image_name}: {e}")
