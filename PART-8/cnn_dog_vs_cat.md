# Convolutional Neural Network

### **Part 1 - Building the CNN**

### Importing the Keras libraries and packages
```python
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
```

### Initialising the CNN
```python
classifier = Sequential()
```

### Step 1 - Convolution
```python
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
```

### Step 2 - Pooling
```python
classifier.add(MaxPooling2D(pool_size = (2, 2)))
```

### Adding a second convolutional layer
```python
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
```

### Step 3 - Flattening
```python
classifier.add(Flatten())
```

### Step 4 - Full connection
```python
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
```

### Compiling the CNN
```python
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

### **Part 2 - Fitting the CNN to the images**

```python
from keras.preprocessing.image import ImageDataGenerator
```

### Image generator
```python
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
```

### loading our training set
```python
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
```

### loading our test set
```python
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
```

### model fitting
```python
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 2000)
```

### model saving for future use
```python
classifier.save('cnn_classifier.h5')
```
