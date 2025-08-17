import os
import cv2
import shutil

def create_yolo_format(image_dir, annotation_dir, output_dir, class_id):
    # Create output directories if they do not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.png'):  # Assuming PNG images
            image_path = os.path.join(image_dir, image_name)
            annotation_path = os.path.join(annotation_dir, image_name.replace('.png', '.txt'))
            output_txt_path = os.path.join(output_dir, image_name.replace('.png', '.txt'))

            # Process the image to get the center_x, center_y, width, height in normalized format
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # Assuming no bounding boxes are provided, so we use the whole image
            center_x = 0.5  # Middle of the image
            center_y = 0.5  # Middle of the image
            w = 1.0  # Full width
            h = 1.0  # Full height

            # Create YOLO format text file
            with open(output_txt_path, 'w') as file:
                file.write(f"{class_id} {center_x} {center_y} {w} {h}\n")

            print(f"Processed {image_name} into YOLO format.")

# Specify directories
fire_images_dir = 'D:/yoloFire/fire_dataset/fire_images'  #replace your directory
non_fire_images_dir = 'D:/yoloFire/fire_dataset/non_fire_images' #replace your directory
output_annotations_dir = 'D:/yoloFire/fire_dataset/annotations' #replace your directory

# Process fire images (class_id = 0)
create_yolo_format(fire_images_dir, fire_images_dir, output_annotations_dir, class_id=0)

# Process non-fire images (class_id = 1)
create_yolo_format(non_fire_images_dir, non_fire_images_dir, output_annotations_dir, class_id=1)
