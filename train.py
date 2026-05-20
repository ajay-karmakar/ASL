import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import os

train_dir = 'asl_alphabet_train'
img_width, img_height = 299, 299
batch_size = 32
num_classes = 29
epochs = 30

train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.8, 1.0],
    zoom_range=[1.0, 1.2],
    validation_split=0.2 
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

for layer in base_model.layers[:248]:
    layer.trainable = False
for layer in base_model.layers[248:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

model.save('asl_inception_model.h5')
print("Model saved as asl_inception_model.h5")


class_indices = train_generator.class_indices
with open('class_indices.txt', 'w') as f:
    f.write(str(class_indices))
print("Class indices saved to class_indices.txt")