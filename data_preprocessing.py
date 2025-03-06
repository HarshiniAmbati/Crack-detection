import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images(dataset_path, output_path, img_size=(224, 224)):
    """
    Preprocesses images by resizing, normalizing, and optionally augmenting them.

    Args:
        dataset_path: Path to the directory containing the original images.
        output_path: Path to the directory where preprocessed images will be saved.
        img_size: Desired size of the output images (tuple).

    Returns:
        None
    """
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

    for category in ['Negative', 'Positive']:
        category_path = os.path.join(dataset_path, category)
        output_category_path = os.path.join(output_path, category)

        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        for image_file in os.listdir(category_path):
            img_path = os.path.join(category_path, image_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize pixel values

            cv2.imwrite(os.path.join(output_category_path, image_file), img)

            # Augment the image (optional)
            img = img.reshape((1,) + img.shape)
            i = 0
            for batch in datagen.flow(img, batch_size=1, save_to_dir=output_category_path, save_prefix='aug',
                                      save_format='jpg'):
                i += 1
                if i > 5:  # Create 5 augmented images per original image
                    break

if __name__ == "__main__":
    preprocess_images('road_dataset', 'preprocessed_data')