import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(data_dir, img_size=(256, 256), batch=32):
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        subset='validation'
    )

    return train_gen, val_gen

def build_model(classes, img_size=(256, 256)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir, model_path="model.h5", img_size=(256, 256), batch=32, epochs=4):
    train_gen, val_gen = load_dataset(data_dir, img_size, batch)
    
    model = build_model(len(train_gen.class_indices), img_size)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[early_stop])
    model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    data_dir = "data/valid"  
    train_model(data_dir)
