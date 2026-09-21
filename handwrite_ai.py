import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

# model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(x_train, y_train, epochs = 10)

# model.save('handwritten2.keras')




# model1 = tf.keras.models.load_model('handwritten.keras')

# loss, accuracy = model1.evaluate(x_test,y_test)

# print(f'loss = {loss}', f'accuracy = {accuracy}', sep='\n')

# # loss = 0.09267953783273697
# # accuracy = 0.97079998254776


# im_num = 0

# while os.path.isfile(f'digits/digit{im_num}.png'):
#     try:
#         img = cv2.imread(f'digits/digit{im_num}.png')[:,:,0]
#         img = np.invert(np.array([img]))
#         pred = model1.predict(img)
#         print(f'THIS DIGIT IS PROBABLY A {np.argmax(pred)}')
#         plt.imshow(img[0], cmap = plt.cm.binary)
#         plt.pause(3)
#         plt.close()
#     except:
#         print('ERROR')
#     finally:
#         im_num += 1





model2 = tf.keras.models.load_model('handwritten2.keras')

loss, accuracy = model2.evaluate(x_test,y_test)

print(f'loss = {loss}', f'accuracy = {accuracy}', sep='\n')

# loss = 0.10691659897565842
# accuracy = 0.9749000072479248


im_num = 0

while os.path.isfile(f'digits/digit{im_num}.png'):
    try:
        img = cv2.imread(f'digits/digit{im_num}.png')[:,:,0]
        img = np.invert(np.array([img]))
        pred = model2.predict(img)
        print(f'THIS DIGIT IS PROBABLY A {np.argmax(pred)}')
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.pause(2)
        plt.close()
    except:
        print('ERROR')
    finally:
        im_num += 1



