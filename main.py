import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import torchvision.transforms as trn
from cleverhans.tf2.attacks import fast_gradient_method, projected_gradient_descent, momentum_iterative_method

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, ytest) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(ytest, 10)

# Load pre-trained VGG16 model (excluding the top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add new classification layers on top of the base model
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')  # CIFAR-10 has 10 classes
])


# Compile the model
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Train the model
# model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

def FGSM(m, data, epsilon):
    data = np.expand_dims(data, axis=0)
    data_fgsm = fast_gradient_method.fast_gradient_method(m, data, epsilon, norm=np.inf, clip_min=None, clip_max=None)
    return data_fgsm

def PGD(m, data, epsilon):
    data = np.expand_dims(data, axis=0)
    data_pgd = projected_gradient_descent.projected_gradient_descent(m, x_test, epsilon,
    eps_iter=2.5*(epsilon/10), nb_iter=10, norm=np.inf, clip_min=None, clip_max=None)
    return data_pgd

# def MIM(m, data, epsilon):
#     data = np.expand_dims(data, axis=0)
#     data_MIM = momentum_iterative_method.momentum_iterative_method(m, data, epsilon, eps_iter = 2.5*(epsilon/10),
#     nb_iter=10, norm=np.inf, clip_min=0, clip_max=1)
#     return data_MIM

# #create a dictionary for storing the adversarial perturbation types
import collections
d = collections.OrderedDict()
d['fgsm'] = FGSM
d['pgd'] = PGD
# d['mim'] = MIM



# define the adversarial generation method:
# x_fgsm = fast_gradient_method.fast_gradient_method(model, x_test, eps=180.0/255, norm=np.inf, clip_min=0, clip_max=1)
# x_pgd = projected_gradient_descent.projected_gradient_descent(model, x_test, eps=180.0/255, eps_iter=0.01, 
# nb_iter=10, norm=np.inf, clip_min=0, clip_max=1)
# fig, (ax1, ax2) = plt.subplots(1,2, figsize =(4,4))
# ax1.imshow(x_test[0])
# ax2.imshow(x_pgd[0])
# plt.show()
convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])

for method_name in d.keys():
    print(f'Creating corruption for {method_name} perturbation')
    cifar_adv, labels = [], []

    for epsilon in [2.0/255, 4.0/255, 6.0/255, 8.0/255, 16.0/255]:
        corruption = lambda clean_img: d[method_name](model, clean_img, epsilon)
        for img, label in zip(x_test, ytest):
            labels.append(label.squeeze())
            cifar_adv.append(np.uint8(corruption(convert_img(img))))
    cifar_adv=np.squeeze(cifar_adv)
    np.save('PATH/TO/adversarial_corruption/'+ d[method_name].__name__ + '.npy',
            np.array(cifar_adv).astype(np.uint8))
    
    np.save('PATH/TO/adversarial_corruption/labels.npy',
            np.array(labels).astype(np.uint8))



