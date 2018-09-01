### import packages
```python
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
```

### dimensions of our images    -----   are these then grayscale (black and white)?
```python
img_width, img_height = 64,64
```

### load the model we saved
```python
model = load_model('cnn_classifier.h5')
```

### Get test image ready
```python
test_image = image.load_img('what-does-it-mean-when-cat-wags-tail.jpg', target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image.reshape(1, 64, 64, 3)   # Ambiguity! Should this instead be: test_image.reshape(img_width, img_height, 3) ??
```

### model result
```python
result = model.predict(test_image, batch_size=1)
print(result)
if result == 1:
    print ('its dog..!')
else:
    print ('its cat..!')
```


# OUTPUT IS HERE..!!
```python
[[0.]]
its cat..!
```
