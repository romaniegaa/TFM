import os

import numpy as np

import cv2
import aspose.words as aw

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

def get_paths(main_path):
    """Get paths from main directory.

    Args:
        main_path (str): Directory where the data is.

    Returns:
        tuple: Returns tuple containing paths and file names.
    """
    # Save lists
    paths = []
    file_names = []

    # Search for directories
    for dirs, sub, files in os.walk(main_path):
        for file in files:
            if file.endswith("dseg.svg"):
                paths.append(os.path.join(dirs, file))
                file_names.append(file)
    
    return paths, file_names

def convert_svg_to_png(paths: list):
    """Converts svg files to png.

    Args:
        paths (list<str>): List containing the paths of the images.    
    """
    for path in paths:
        doc = aw.Document()
        builder = aw.DocumentBuilder(doc)
        shapes = builder.insert_image(path)
        shapes.image_data.save(path.replace("svg", "png"))

def import_png(paths: list):
    """Import png data.

    Args:
        paths (list<str>): List containing the paths of the images.

    Returns:
        list<cv::Mat>: Returns list containing raw data.
    """
    # Open the .png
    data = []
    for path in paths:
            data.append(cv2.imread(path.replace("svg", "png"), cv2.IMREAD_COLOR))

    return data

def crop_data_and_norm(data: list, paths: list, img_index: list, img_size: tuple):
    """Crop the data given a coordinate and normalize it.

    Args:
        data (list<cv::Mt>): List containing raw data.
        paths (list<str>): List containing the paths of the images.
        img_number (list<int>): List containing the indexes of the desired image [x, y].
        img_size (tuple): Tuple containing the cropped image size.

    Returns:
        tuple: Returns a tuple containing the cropped data and the tags.
    """
    # Coordinates
    x1 = 2 + img_index[0] * (img_size[0] + 9)
    x2 = x1 + img_size[0]
    y1 = 10 + img_index[1] * (img_size[1] + 5)
    y2 = y1 + img_size[1]

    # Cropping
    cropped_data = []
    wrong_img_index = [638, 634, 628, 627, 625, 624, 623, 80, 79]

    for enum, img in enumerate(data):
        if  enum in wrong_img_index:
            if img_index[0] == 0:
                img_index[0] = 1
                cropped_data.append(img[x1:x2, y1:y2,:])
            elif img_index[0] == 1:
                img_index[0] = 0
                cropped_data.append(img[x1:x2, y1:y2,:])

        cropped_data.append(img[x1:x2, y1:y2,:])
    
    # Normalization
    cropped_data = np.array(cropped_data)
    cropped_data = cropped_data/255

    # Obtain the tags
    tags = np.zeros(len(paths), dtype = int)
    
    for enum, path in enumerate(paths):
        if "patient" in path:
            tags[enum] = 1

    tags = np.delete(tags, wrong_img_index)

    return cropped_data, tags

def save_data(folder, data, file_names):
    """Saves the data in the desired folder
    
    Args:
        folder (str): String containing the folder directory where the data will be stored.
        data (ndarray): Numpy array containing the data to be stored.
        file_names (list<str>): List containing the names of the files to be saved.
    """
    # Save data in folder
    if np.max(data) <= 1:
        data *= 255
        data = np.clip(data, 0, 255, dtype = int, casting = 'unsafe')

    for enum, img in enumerate(data):
        cv2.imwrite(folder + "\\" + file_names[enum].replace("svg", "png"), img)

def get_model_data(data: np.ndarray, tags: np.ndarray, split: np.ndarray):
    """Get training and testing dataset and their respective tags
    
    Args:
        data (ndarray): Numpy array containing the data to be split.
        tags (ndarray): Numpy array containing the tags to be split.
        split (ndarray): Numpy array containing the split.

    Returns:
        training_data (ndarray): Numpy array containing the training data.
        training_tags (ndarray): Numpy array containing the training tags.
        testing_data (ndarray): Numpy array containing the testing data.
        testing_tags (ndarray): Numpy array containing the testing tags.
    """
    # Training and testing datasets
    training_data = data[split,:,:] 
    testing_data = data[~split,:,:] 

    training_tags = tags[split]
    testing_tags = tags[~split]

    # One-hot encode
    # training_tags = np.eye(2)[training_tags]
    # testing_tags = np.eye(2)[testing_tags]
   
    return training_data, training_tags, testing_data, testing_tags

def data_augmentation(source, save_folder):
    """Generates images from source directory containing the classes and saves them in the desired folder.
    
    Args:
        source (str): Source directory where the images to be augmented are stored.
        save_folder (str): Directory where the augmented images will be stored.
    """
    datagen = ImageDataGenerator(shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

    i = 0
    for image in datagen.flow_from_directory(directory = source,
                                            batch_size = 256,
                                            target_size = (110, 110),
                                            color_mode = "rgb",
                                            save_to_dir = save_folder,
                                            save_prefix = "aug",
                                            class_mode= "binary",
                                            save_format = "png"):
        i += 1
        if i > 31:
            break

def load_append_aug(folder, split: np.ndarray, training_data: np.ndarray, training_tags: np.ndarray):
    """Load and append the augmented data and tags.
    
    Args:
        folder (str): Directory where the augmented data is stored.
        split (ndarray): Numpy array containing the split.
        training_data (ndarray): Numpy array containing the training data.
        training_tags (ndarray): Numpy array containing the training tags.

    Returns:
        tuple: Returns tuple containing the training data and the tags with augmented data.
    """
    # Find indexes for training images
    indexes = np.where(split == True)[0]

    # Load augmented images
    aug_imgs = []
    aug_tags = []
    for filename in os.listdir(folder):
        # If the index is on the file name
        i_img = filename.split("_")[1]
        if int(i_img) in list(indexes):
            # Import the file and save it in aug_imgs
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_COLOR)
            aug_imgs.append(img)
            # Add the tag
            i_tag = np.where(indexes == int(i_img))[0]
            tag = training_tags[i_tag][0]
            aug_tags.append(tag)

    print(training_data.shape)
    print(aug_imgs[0].shape)

    aug_imgs = np.array(aug_imgs)
    aug_tags = np.array(aug_tags)

    # Append aug_imgs to training_data
    data = np.append(training_data, aug_imgs, axis = 0)
    print(data.shape)
    tags = np.append(training_tags, aug_tags, axis = 0)
    print(tags.shape)

    return data, tags
