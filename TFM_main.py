import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import aspose.words as aw

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D 
from keras.layers import Flatten, Dropout, BatchNormalization
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

from TFM_utils import *


if __name__ == "__main__":
    # Set seed
    np.random.seed(12345)

    # Folder where the variables are stored
    folder = r"C:\Users\Usuario\OneDrive\Escritorio\MBB\TFM\Code\Variables3"

    # If True, the augmented data will be read/obtained; while False, original data will be read/obtained.
    read_augmented = True

    try:
        testing_data = np.load(folder+r"\testing_data.npy")
        testing_tags = np.load(folder+r"\testing_tags.npy")

        if read_augmented:
            training_data = np.load(folder+r"\aug_training_data.npy")   
            training_tags = np.load(folder+r"\aug_training_tags.npy")
        else:
            training_data = np.load(folder+r"\training_data.npy")   
            training_tags = np.load(folder+r"\training_tags.npy")

    except Exception as e:
        # Find the paths and file names with optional from .svg to .png convertor
        paths, file_names = get_paths(r"d:\fmriprep")
        
        # Import .png data
        data = import_png(paths)

        # Crop the data in desired coordinates and image sizes
        cropped_data, tags = crop_data_and_norm(data, paths, [2, 3], (100, 100))

        # Store the data
        save_data(r"d:\cropped_col", cropped_data, file_names)

        # Create training/testing split
        split = np.random.rand(cropped_data.shape[0]) <= 0.8

        # Obtain training and testing data and tags
        training_data, training_tags, testing_data, testing_tags = get_model_data(cropped_data, tags, split)

        # Create data augmentation
        #data_augmentation(r"d:\to_aug", r"d:\augmented")        

        # Save data
        np.save(folder+r"\testing_data.npy", testing_data)
        np.save(folder+r"\testing_tags.npy", testing_tags)

        if read_augmented:
            training_data, training_tags = load_append_aug(r"d:\augmented", split, training_data, training_tags)
            np.save(folder+r"\aug_training_data.npy", training_data)
            np.save(folder+r"\aug_training_tags.npy", training_tags)
        else:
            np.save(folder+r"\training_data.npy", training_data)
            np.save(folder+r"\training_tags.npy", training_tags)

    """
    Transfer learning
    """

    # Load pretrained model

    vgg16_model = Sequential()

    vgg16_pretrained = VGG16(input_shape = (110, 110, 3),
                      include_top = False,
                      weights = "imagenet",
                      classes = 2)
    
    # Convert layers to not trainable
    for layer in vgg16_pretrained.layers:
        layer.trainable = False

    vgg16_model.add(vgg16_pretrained)
    vgg16_model.add(Flatten())
    vgg16_model.add(Dense(1, activation = "sigmoid"))

    vgg16_model.summary()

    # Compile model
    vgg16_model.compile("rmsprop",
                        loss = BinaryCrossentropy(),
                        metrics = ["binary_accuracy"])

    # Callback
    callback = EarlyStopping(monitor = 'loss', patience = 10, start_from_epoch = 10,
                                mode = 'min', verbose = 0)

    # Fit model
    history = vgg16_model.fit(x = training_data, y = training_tags,
                        batch_size = 256, epochs = 100, validation_split = 0.2,
                        shuffle = True, verbose = 1, callbacks = callback)
                                
    # Evaluate model
    result = vgg16_model.evaluate(x = testing_data, y = testing_tags,
                            batch_size = 10, verbose = 0)

    # Output string
    output = ""
    output += f"Ev Loss : {float(result[0]):>12.8f} -- "
    output += f"Ev Acc : {float(result[1]):>12.8f}"
    
    print(output)

    fig, ax = plt.subplots(2, 1, figsize =(10, 5))
    ax[0].plot(history.history["loss"], label = "loss")
    ax[0].plot(history.history["val_loss"], label = "val_loss")
    ax[1].plot(history.history["binary_accuracy"], label = "accuracy")
    ax[1].plot(history.history["val_binary_accuracy"], label = "val_accuracy")

    ax[0].legend()
    ax[1].legend()

    plt.savefig(r'C:\\Users\\Usuario\\OneDrive\\Escritorio\\MBB\\TFM\\Code\\accuracy_loss.png')

    exit()

    """
    Training Hyperparameters
    """   

    # Hyperparameters    
    n_layers =  np.array([3]) #np.arange(1, 6, 1)
    n_filters = np.array([32]) #np.array([int(2**val) for val in np.arange(4, 9, 1)])
    n_sizes = np.array([3])
    n_dropouts = np.array([0.3]) #np.arange(0.1, 0.6, 0.1)
    n_units = np.zeros(1)
    
    try:
        # Load data
        saves_path = r'C:\\Users\\Usuario\\OneDrive\\Escritorio\\MBB\\TFM\\Code\\'
        
        tr_accuracy = np.load(f'{saves_path}tr_accuracy.npy')
        tr_loss = np.load(f'{saves_path}tr_loss.npy')
        ev_accuracy = np.load(f'{saves_path}ev_accuracy.npy')
        ev_loss = np.load(f'{saves_path}ev_loss.npy')
        checkpoints = np.load(f'{saves_path}checkpoints.npy')
        
    except Exception as e:
        dimensions = (n_layers.shape[0],
                      n_filters.shape[0],
                      n_sizes.shape[0],
                      n_dropouts.shape[0],
                      n_units.shape[0])
        
        tr_accuracy = np.zeros(dimensions)
        tr_loss = np.zeros(dimensions)
        ev_accuracy = np.zeros(dimensions)
        ev_loss = np.zeros(dimensions)
        checkpoints = np.zeros(dimensions, dtype = int)

    """
    Get Iterators
    """
    
    iter_grid = np.array(np.meshgrid(range(n_layers.shape[0]),
                                     range(n_filters.shape[0]),
                                     range(n_sizes.shape[0]),
                                     range(n_dropouts.shape[0]),
                                     range(n_units.shape[0])))
    iter_grid = iter_grid.T.reshape(-1, 5)
    
    for e_l, e_f, e_s, e_d, e_u in iter_grid:
        n_layer = n_layers[e_l]
        n_filter = n_filters[e_f]
        n_size = n_sizes[e_s]
        n_dropout = n_dropouts[e_d]
        n_unit = n_units[e_u]
        
        if checkpoints[e_l, e_f, e_s, e_d, e_u] == 1:
            continue
        
        try:
            model = Sequential()
            
            # Input Layer
            model.add(Conv2D(filters = n_filter,
                                kernel_size = (n_size, n_size),
                                input_shape = (110, 110, 3),
                                activation = "relu",
                                padding = "same"))
            model.add(MaxPooling2D((n_size, n_size), padding = "same"))
            # Uncomment next line to add Batch Normalization layer
            model.add(BatchNormalization())
            # model.add(Dropout(rate = n_dropout))
            
            # Convolutional Layers
            for n_l in range(n_layer):
                model.add(Conv2D(filters = n_filter,
                                    kernel_size = (n_size, n_size),
                                    activation = "relu",
                                    padding = "same"))
                model.add(MaxPooling2D((n_size, n_size), padding = "same"))
                # Uncomment next line to add Batch Normalization layer
                model.add(BatchNormalization())
                # model.add(Dropout(rate = n_dropout))
                
            model.add(Flatten())
            
            # Dense Layers
            if n_unit > 0:
                model.add(Dense(units = n_unit, activation = "relu"))
            
            # Output Layer
            model.add(Dense(units = 1, activation = "sigmoid"))
            
            # Compile model
            model.compile("rmsprop",
                            loss = BinaryCrossentropy(),
                            metrics = ["binary_accuracy"])
            
            # Callback
            callback = EarlyStopping(monitor = 'loss', patience = 10, start_from_epoch = 10,
                                        mode = 'min', verbose = 0)

            # Fit model
            history = model.fit(x = training_data, y = training_tags,
                                batch_size = 64, epochs = 100, validation_split = 0.2,
                                shuffle = True, verbose = 1, callbacks = callback)
                                        
            # Evaluate model
            result = model.evaluate(x = testing_data, y = testing_tags,
                                    batch_size = 10, verbose = 0)
            
            # Save data
            tr_accuracy[e_l, e_f, e_s, e_d, e_u] = float(history.history["binary_accuracy"][-1])
            tr_loss[e_l, e_f, e_s, e_d, e_u] = float(history.history["loss"][-1])
            ev_accuracy[e_l, e_f, e_s, e_d, e_u] = float(result[1])
            ev_loss[e_l, e_f, e_s, e_d, e_u] = float(result[0])
            checkpoints[e_l, e_f, e_s, e_d, e_u] = 1
            
            # Output string
            output = ""
            output += "Model fit ... "
            output += f"N Layer : {n_layer:>2} -- "
            output += f"N Filter : {n_filter:>4} -- "
            output += f"N Size : {n_size:>2}"
            output += f"N Dropout : {n_dropout:>3.1f}"
            output += f"N Unit : {n_unit:>2} -- "
            output += f"Tr Loss : {float(history.history['loss'][-1]):>12.8f}  -- "
            output += f"Tr Acc : {float(history.history['binary_accuracy'][-1]):>12.8f} -- "
            output += f"Ev Loss : {float(result[0]):>12.8f} -- "
            output += f"Ev Acc : {float(result[1]):>12.8f}"
            
            print(output)

            fig, ax = plt.subplots(2, 1, figsize =(10, 5))
            ax[0].plot(history.history["loss"], label = "loss")
            ax[0].plot(history.history["val_loss"], label = "val_loss")
            ax[1].plot(history.history["binary_accuracy"], label = "accuracy")
            ax[1].plot(history.history["val_binary_accuracy"], label = "val_accuracy")

            ax[0].legend()
            ax[1].legend()

            plt.savefig(r'C:\\Users\\Usuario\\OneDrive\\Escritorio\\MBB\\TFM\\Code\\accuracy_loss.png')
            
        except Exception as e:
            # Save results
            saves_path = r'C:\\Users\\Usuario\\OneDrive\\Escritorio\\MBB\\TFM\\Code\\'
            
            np.save(f'{saves_path}tr_accuracy.npy', tr_accuracy)
            np.save(f'{saves_path}tr_loss.npy', tr_loss)
            np.save(f'{saves_path}ev_accuracy.npy', ev_accuracy)
            np.save(f'{saves_path}ev_loss.npy', ev_loss)
            np.save(f'{saves_path}checkpoints.npy', checkpoints)
            
            raise e
        
    # Save results
    saves_path = r'C:\\Users\\Usuario\\OneDrive\\Escritorio\\MBB\\TFM\\Code\\'

    np.save(f'{saves_path}tr_accuracy.npy', tr_accuracy)
    np.save(f'{saves_path}tr_loss.npy', tr_loss)
    np.save(f'{saves_path}ev_accuracy.npy', ev_accuracy)
    np.save(f'{saves_path}ev_loss.npy', ev_loss)
    np.save(f'{saves_path}checkpoints.npy', checkpoints)

    exit()
 
    """
    Save training scatterplots
    """    

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize = (15, 7.5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    x = n_layers
    y = n_filters
    z = n_units
    x, y, z = np.meshgrid(x, y, z)
    c = tr_accuracy
    img = ax.scatter(x, y, z, c=c, cmap=plt.viridis())
    fig.colorbar(img)

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    x = n_layers
    y = n_filters
    z = n_units
    x, y, z = np.meshgrid(x, y, z)
    c = tr_loss
    img = ax.scatter(x, y, z, c=c, cmap=plt.viridis())
    fig.colorbar(img)

    plt.savefig(r'C:\Users\Usuario\OneDrive\Escritorio\MBB\TFM\Code\tr_scatterplot.png')

    from matplotlib import cm

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize = plt.figaspect(0.5))

    """
    Save evaluating scatterplots
    """ 
    
    fig = plt.figure(figsize = (15, 7.5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    x = n_layers
    y = n_filters
    z = n_units
    x, y, z = np.meshgrid(x, y, z)
    c = ev_accuracy
    img = ax.scatter(x, y, z, c=c, cmap=plt.viridis())
    fig.colorbar(img)

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    x = n_layers
    y = n_filters
    z = n_units
    x, y, z = np.meshgrid(x, y, z)
    c = ev_loss
    img = ax.scatter(x, y, z, c=c, cmap=plt.viridis())
    fig.colorbar(img)

    plt.savefig(r'C:\Users\Usuario\OneDrive\Escritorio\MBB\TFM\Code\ev_scatterplot.png')

    """
    Previous code
    """

    ##############################
    #     DATA PREPROCESSING     #
    ##############################

    # Find the directories
    main = r"D:\fmriprep"

    paths = []
    file_names = []
    for dirs, sub, files in os.walk(main):
        for file in files:
            if file.endswith("dseg.svg"):
                paths.append(os.path.join(dirs, file))
                file_names.append(file)

    # From .svg to .png
    for enum, path in enumerate(paths):
        doc = aw.Document()
        builder = aw.DocumentBuilder(doc)
        shapes = builder.insert_image(path)
        shapes.image_data.save(path.replace("svg", "png"))
        print(enum, path)

    # Import .png files, for grey use cv2.IMREAD_GRAYSCALE and for color use cv2.IMREAD_COLOR.
    # This will affect the dimension of the data.
    data = []

    for path in paths:
        data.append(cv2.imread(path.replace("svg", "png"), cv2.IMREAD_COLOR)) 

    # Crop images
    # Shape of full image is (353, 831)
    # Cropped image will be (100, 100)
    cropped_data = []
    wrong_img_index = [638, 634, 628, 627, 625, 624, 623, 80, 79]

    for enum, img in enumerate(data):
        if enum in wrong_img_index:
            cropped_data.append(img[15:115, 364:464,])
        else:
            cropped_data.append(img[130:230, 364:464,])

    plt.imshow(cropped_data[0], cmap = "gray")
    plt.show()

    # Scaling
    cropped_data = np.array(cropped_data)
    cropped_data = cropped_data/255

    # Obtain the tags
    tags = np.zeros(len(paths), dtype = int)

    for enum, path in enumerate(paths):
        if "patient" in path:
            tags[enum] = 1

    # Training and testing datasets
    np.random.seed(12345)

    split = np.random.rand(cropped_data.shape[0]) <= 0.67
    training_data = cropped_data[split,:] # 676 img
    testing_data = cropped_data[~split,:] # 348 img

    training_tags = tags[split] # 676 tags
    testing_tags = tags[~split] # 348 tags

    # One-hot encode
    training_tags = np.eye(2)[training_tags]
    testing_tags = np.eye(2)[testing_tags]

    #################################
    #        MODELS w/ COLOR        #
    #################################

    # Model 1 - COLOR
    
    model1c = Sequential()
    # Conv2D layer as input layer
    model1c.add(Conv2D(filters = 16,
                    kernel_size = (2, 2),
                    input_shape = (100, 100, 3),
                    activation = "relu",
                    padding = "same"))
    model1c.add(MaxPooling2D((2, 2)))
    model1c.add(Flatten())
    model1c.add(Dense(units = 2,
                    activation = "softmax"))
    model1c.compile("rmsprop",
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ["accuracy"])
  
    model1c.summary()

    history1c = model1c.fit(x = training_data,  
                    y = training_tags,
                    batch_size = 10,
                    epochs = 10,
                    verbose = 1,
                    shuffle = True,
                    validation_split = 0.1)
    
    fig, ax = plt.subplots(2, 1, figsize =(10, 5))
    ax[0].plot(history1c.history["loss"], label = "loss")
    ax[0].plot(history1c.history["val_loss"], label = "val_loss")
    ax[1].plot(history1c.history["accuracy"], label = "accuracy")
    ax[1].plot(history1c.history["val_accuracy"], label = "val_accuracy")

    ax[0].legend()
    ax[1].legend()

    plt.show()

    prediction1c = model1c.evaluate(testing_data, testing_tags)


    # Model 2 - COLOR

    model2c = Sequential()
    # Conv2D layer as input layer
    model2c.add(Conv2D(filters = 16,
                    kernel_size = 2,
                    input_shape = (100, 100, 3),
                    activation = "relu",
                    padding = "same"))
    model2c.add(MaxPooling2D())
    model2c.add(Conv2D(filters = 32,
                    kernel_size = 2,
                    activation = "relu",
                    padding = "same"))
    model2c.add(MaxPooling2D())
    model2c.add(Conv2D(filters = 64,
                    kernel_size = 2,
                    activation = "relu",
                    padding = "same"))
    model2c.add(MaxPooling2D())
    model2c.add(Conv2D(filters = 128,
                    kernel_size = 2,
                    activation = "relu",
                    padding = "same"))
    model2c.add(MaxPooling2D())
    model2c.add(Conv2D(filters = 256,
                    kernel_size = 2,
                    activation = "relu",
                    padding = "same"))
    model2c.add(MaxPooling2D())                                                                                
    model2c.add(Flatten())
    model2c.add(Dense(units = 2,
                    activation = "sigmoid"))
    model2c.compile("adam",
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ["accuracy"])

    model2c.summary()

    history2c = model2c.fit(x = training_data,
                    y = training_tags,
                    batch_size = 10,
                    epochs = 10,
                    verbose = 1,
                    shuffle = True,
                    validation_split = 0.1)

    fig, ax = plt.subplots(2, 1, figsize =(10, 5))
    ax[0].plot(history2c.history["loss"], label = "loss")
    ax[0].plot(history2c.history["val_loss"], label = "val_loss")
    ax[1].plot(history2c.history["accuracy"], label = "accuracy")
    ax[1].plot(history2c.history["val_accuracy"], label = "val_accuracy")

    ax[0].legend()
    ax[1].legend()

    plt.show()

    prediction2c = model2c.evaluate(testing_data, testing_tags)

    #################################
    #             MODELS            #
    #################################

    # Model 1

    model = Sequential()
    # Conv2D layer as input layer
    model.add(Conv2D(filters = 16,
                    kernel_size = (2, 2),
                    input_shape = (100, 100, 1),
                    activation = "relu",
                    padding = "same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 2,
                    activation = "softmax"))
    model.compile("rmsprop",
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ["accuracy"])

    model.summary()

    history = model.fit(x = training_data,  
                    y = training_tags,
                    batch_size = 50,
                    epochs = 10,
                    verbose = 1,
                    shuffle = True,
                    validation_split = 0.1)
    
    fig, ax = plt.subplots(2, 1, figsize =(10, 5))
    ax[0].plot(history.history["loss"], label = "loss")
    ax[0].plot(history.history["val_loss"], label = "val_loss")
    ax[1].plot(history.history["accuracy"], label = "accuracy")
    ax[1].plot(history.history["val_accuracy"], label = "val_accuracy")

    ax[0].legend()
    ax[1].legend()

    plt.show()

    prediction = model.evaluate(testing_data, testing_tags)

    # Model 2
   
    model2 = Sequential()
    # Conv2D layer as input layer
    model2.add(Conv2D(filters = 16,
                    kernel_size = (2, 2),
                    input_shape = (100, 100, 1),
                    activation = "relu",
                    padding = "same"))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Conv2D(filters = 32,
                    kernel_size = (2, 2),
                    activation = "relu",
                    padding = "same"))
    model2.add(MaxPooling2D((2, 2)))          
    model2.add(Flatten())
    model2.add(Dense(units = 2,
                    activation = "softmax"))
    model2.compile("rmsprop",
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ["accuracy"])

    model2.summary()

    history2 = model2.fit(x = training_data,
                   y = training_tags,
                   batch_size = 50,
                   epochs = 10,
                   verbose = 1,
                   shuffle = True,
                   validation_split = 0.2)
    
    fig, ax = plt.subplots(2, 1, figsize =(10, 5))
    ax[0].plot(history2.history["loss"], label = "loss")
    ax[0].plot(history2.history["val_loss"], label = "val_loss")
    ax[1].plot(history2.history["accuracy"], label = "accuracy")
    ax[1].plot(history2.history["val_accuracy"], label = "val_accuracy")

    ax[0].legend()
    ax[1].legend()

    plt.show()

    prediction2 = model2.evaluate(testing_data, testing_tags)

    # Model 3

    model3 = Sequential()
    # Conv2D layer as input layer
    model3.add(Conv2D(filters = 16,
                    kernel_size = (2, 2),
                    input_shape = (100, 100, 1),
                    activation = "relu",
                    padding = "same"))
    model3.add(MaxPooling2D((2, 2)))
    model3.add(Conv2D(filters = 32,
                    kernel_size = (2, 2),
                    activation = "relu",
                    padding = "same"))
    model3.add(MaxPooling2D((2, 2)))
    model3.add(Conv2D(filters = 64,
                    kernel_size = (2, 2),
                    activation = "relu",
                    padding = "same"))
    model3.add(MaxPooling2D((2, 2)))                    
    model3.add(Flatten())
    model3.add(Dense(units = 2,
                    activation = "softmax"))
    model3.compile("adam",
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ["accuracy"])
    
    model3.summary()

    history3 = model3.fit(x = training_data,
                   y = training_tags,
                   batch_size = 10,
                   epochs = 10,
                   verbose = 1,
                   shuffle = True,
                   validation_split = 0.1)
    
    fig, ax = plt.subplots(2, 1, figsize =(10, 5))
    ax[0].plot(history3.history["loss"], label = "loss")
    ax[0].plot(history3.history["val_loss"], label = "val_loss")
    ax[1].plot(history3.history["accuracy"], label = "accuracy")
    ax[1].plot(history3.history["val_accuracy"], label = "val_accuracy")

    ax[0].legend()
    ax[1].legend()

    plt.show()

    prediction3 = model3.evaluate(testing_data, testing_tags)

    # Model 4

    model4 = Sequential()
    # Conv2D layer as input layer
    model4.add(Conv2D(filters = 16,
                    kernel_size = (2, 2),
                    input_shape = (100, 100, 1),
                    activation = "relu",
                    padding = "same"))
    model4.add(MaxPooling2D((2, 2)))
    model4.add(Conv2D(filters = 32,
                    kernel_size = (2, 2),
                    activation = "relu",
                    padding = "same"))
    model4.add(MaxPooling2D((2, 2)))
    model4.add(Conv2D(filters = 64,
                    kernel_size = (2, 2),
                    activation = "relu",
                    padding = "same"))
    model4.add(MaxPooling2D((2, 2)))
    model4.add(Conv2D(filters = 128,
                    kernel_size = (2, 2),
                    activation = "relu",
                    padding = "same"))
    model4.add(MaxPooling2D((2, 2)))                                        
    model4.add(Flatten())
    model4.add(Dense(units = 2,
                    activation = "softmax"))
    model4.compile("adam",
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ["accuracy"])
    
    model4.summary()

    history4 = model4.fit(x = training_data,
                   y = training_tags,
                   batch_size = 10,
                   epochs = 10,
                   verbose = 1,
                   shuffle = True,
                   validation_split = 0.1)
    
    fig, ax = plt.subplots(2, 1, figsize =(10, 5))
    ax[0].plot(history4.history["loss"], label = "loss")
    ax[0].plot(history4.history["val_loss"], label = "val_loss")
    ax[1].plot(history4.history["accuracy"], label = "accuracy")
    ax[1].plot(history4.history["val_accuracy"], label = "val_accuracy")

    ax[0].legend()
    ax[1].legend()

    plt.show()

    prediction4 = model4.evaluate(testing_data, testing_tags)

    # Model 5

    model5 = Sequential()
    # Conv2D layer as input layer
    model5.add(Conv2D(filters = 16,
                    kernel_size = (2, 2),
                    input_shape = (100, 100, 1),
                    activation = "relu",
                    padding = "same"))
    model5.add(MaxPooling2D((2, 2)))
    model5.add(Conv2D(filters = 32,
                    kernel_size = (2, 2),
                    activation = "relu",
                    padding = "same"))
    model5.add(MaxPooling2D((2, 2)))
    model5.add(Conv2D(filters = 64,
                    kernel_size = (2, 2),
                    activation = "relu",
                    padding = "same"))
    model5.add(MaxPooling2D((2, 2)))
    model5.add(Conv2D(filters = 128,
                    kernel_size = (2, 2),
                    activation = "relu",
                    padding = "same"))
    model5.add(MaxPooling2D((2, 2)))
    model5.add(Conv2D(filters = 256,
                    kernel_size = (2, 2),
                    activation = "relu",
                    padding = "same"))
    model5.add(MaxPooling2D((2, 2)))                                                                                
    model5.add(Flatten())
    model5.add(Dense(units = 2,
                    activation = "softmax"))
    model5.compile("adam",
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ["accuracy"])
    
    model5.summary()

    history5 = model5.fit(x = training_data,
                   y = training_tags,
                   batch_size = 10,
                   epochs = 10,
                   verbose = 1,
                   shuffle = True,
                   validation_split = 0.1)
    
    fig, ax = plt.subplots(2, 1, figsize =(10, 5))
    ax[0].plot(history5.history["loss"], label = "loss")
    ax[0].plot(history5.history["val_loss"], label = "val_loss")
    ax[1].plot(history5.history["accuracy"], label = "accuracy")
    ax[1].plot(history5.history["val_accuracy"], label = "val_accuracy")

    ax[0].legend()
    ax[1].legend()

    plt.show()

    prediction5 = model5.evaluate(testing_data, testing_tags)
            
    