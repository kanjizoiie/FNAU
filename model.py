import tensorflow as tf

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.5,
    zoom_range = 0.5,
    horizontal_flip = True,
    rotation_range = 170)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 170)


training_set = train_datagen.flow_from_directory("training_set",
    target_size = (64, 64),
    batch_size = 32,
    class_mode = "categorical")

test_set = test_datagen.flow_from_directory("test_set",
    target_size = (64, 64),
    batch_size = 32,
    class_mode = "categorical")


class food_model:
    model = 0
    def __init__(self):
        self.model = tf.keras.models.Sequential();
    def create(self):
        self.model.add(tf.keras.layers.Conv2D(64, (7, 7), input_shape=(64, 64, 3))) # add a convolutional 2d layer to the self.model.
        self.model.add(tf.keras.layers.Activation("relu"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(128, (3, 3)))
        self.model.add(tf.keras.layers.Activation("relu"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        self.model.add(tf.keras.layers.Activation("relu"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


        # the self.model so far outputs 3D feature maps (height, width, features)

        self.model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors

        # 1st Dense layer
        self.model.add(tf.keras.layers.Dense(256))  #
        self.model.add(tf.keras.layers.Activation("relu"))
        self.model.add(tf.keras.layers.Dropout(0.5))

        # 2nd Dense layer
        self.model.add(tf.keras.layers.Dense(256))  #
        self.model.add(tf.keras.layers.Activation("relu"))
        self.model.add(tf.keras.layers.Dropout(0.5))

        # 3rd Dense layer
        self.model.add(tf.keras.layers.Dense(256))  #
        self.model.add(tf.keras.layers.Activation("relu"))
        self.model.add(tf.keras.layers.Dropout(0.5))

        # Output layer
        self.model.add(tf.keras.layers.Dense(len(training_set.class_indices)))
        self.model.add(tf.keras.layers.Activation("softmax"))
    def compile(self):
        # COMPILE
        self.model.compile(loss="categorical_crossentropy", optimizer= "rmsprop", metrics=["accuracy"])
    def get(self):
        return self.model

    def train(self):
        # TRAINING
        self.model.fit_generator(
            training_set,
            steps_per_epoch = 100,
            epochs = 25,
            validation_data = test_set,
            validation_steps = 25)
    def save(self, name):
        self.model.save_weights(name)
    def load(self, name):
        self.model.load_weights(name)
    def predict(self, x):
        return self.model.predict(x)
    def classes(self):
        return training_set.class_indices;