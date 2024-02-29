import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define class names
className_images = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
trainingData_length=len(train_labels)
total_class=max(train_labels)
# print(x)
# print(y)

#plotting images with its labels
plt.figure(figsize=(5, 5))
for i in range(total_class):
    #subplot - used so that we can have many images in one image
    # 1st row,2nd col, 3rd index of curr
    plt.subplot(3, 4, i + 1)
    sample_array = np.where(train_labels == i)
    label_index=sample_array[0][0]
    plt.imshow(train_images[label_index], cmap='gray')
    plt.title(className_images[i])
    plt.axis('off')

plt.show()