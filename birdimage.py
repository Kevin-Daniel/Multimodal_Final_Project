from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# Define the CNN architecture
# from keras.applications.vgg16 import VGG16
# model = VGG16(weights='imagenet')


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preprocess the data
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('FILENAME', target_size=(224, 224), batch_size=10, class_mode='categorical')
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory('FILENAME', target_size=(224, 224), batch_size=10, class_mode='categorical')


# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Test the model
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('FILENAME', target_size=(224, 224), batch_size=10, class_mode='categorical')
test_loss, test_acc = model.evaluate_generator(test_generator)
print('Test accuracy:', test_acc)