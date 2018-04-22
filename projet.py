
from keras.datasets import cifar10

#CR pour le 22/04
from keras.models import Model,Sequential,load_model
from keras.layers import Input, Dense,Flatten, Activation, Dropout
from keras.layers import Conv2D,MaxPooling2D, BatchNormalization
from keras.layers import UpSampling2D, Conv2DTranspose, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=(x_train-127.5)/127.5
x_test=(x_test-127.5)/127.5
import matplotlib as mpl
#mpl.use ( ’ Agg ’ ) # Uncomment this if you have problems to use plt.imshow
#In that case , replace plt.imshow by plt.imsave ( ’filename.png’, var ) ,
#so the image will be saved to a file instead of displayed .

import matplotlib.pyplot as plt
import numpy as np

print(x_train.shape)







def perceptronMul():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train - 127.5) / 127.5
    x_test = (x_test - 127.5) / 127.5
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(512))
    model.add(LeakyReLU(0.1))
    model.add(Dense(512))
    model.add(LeakyReLU(0.1))#decoupe l'espace de facon non lineaire
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    from keras.callbacks import TensorBoard
    tensorboard = TensorBoard(log_dir='./tboard')  # Create ’ tboard ’ before
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=5, callbacks=[tensorboard])
    model.summary()

    return 0


import matplotlib as mpl
#mpl.use ( ’ Agg ’ ) # Uncomment this if you have problems to use plt.imshow
#In that case , replace plt.imshow by plt.imsave ( ’filename.png’, var ) ,
#so the image will be saved to a file instead of displayed .
import matplotlib.pyplot as plt
import numpy as np

def CNN():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train - 127.5) / 127.5
    x_test = (x_test - 127.5) / 127.5
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),padding='same',input_shape=(32, 32, 3)))#Conv2D(nbDeFiltres utilisés,taille de decoupe,on garde la même taille ou non (valid= non),stripes=on decoupe tout les #stripes pixels)
    model.add(LeakyReLU())
    model.add(Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(filters=32,kernel_size=(3,3)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))


    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(512))


    model.add(Dense(10))
    model.add(Activation('softmax'))


    from keras.callbacks import TensorBoard
    from keras.preprocessing.image import ImageDataGenerator
    """ datagen = ImageDataGenerator(
    #         zoom_range=0.2, # randomly zoom into images
           rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
            """
    # datagen.fit(x_train)
    tensorboard = TensorBoard(log_dir='./tboard')  # Create ’ tboard ’ before
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=45, callbacks=[tensorboard])
    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, epochs=35,validation_data=(x_test,y_test),callbacks=[tensorboard])
    # model.predict(arrayImage,(size(Array),32,32,3))
    model.summary()


    return 0


def gen():

    generator = Sequential()
    generator.add(Dense(256,input_shape=(10,)))
    generator.add(LeakyReLU())
    generator.add(Dense(512))
    generator.add(LeakyReLU())
    generator.add(Dropout(0.1))
    generator.add(Dense(1024))
    generator.add(LeakyReLU())
    generator.add(Dropout(0.1))
    generator.add(Dense(3072))
    generator.add(Activation('tanh'))#to put the output between 0 and 1
    generator.add(Dropout(0.1))

    generator.add(Reshape((32,32,3)))



    return generator

def gen2():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train - 127.5) / 127.5
    x_test = (x_test - 127.5) / 127.5
    generator = Sequential()
    generator.add(Dense(7*7*128,input_shape=(10,)))
    generator.add(LeakyReLU(0.1))
    generator.add(Reshape((7,7,128)))
    generator.add(UpSampling2D(size=(2,2)))#multiply by 2 rows and columns
    generator.add(Conv2DTranspose(64,kernel_size=(3,3)))#add 2 rows and 2 columns
    generator.add(LeakyReLU())
    generator.add(Dropout(0.2))
    generator.add(UpSampling2D((2,2)))
    generator.add(Conv2D(128,(3,3)))
    generator.add(LeakyReLU())
    generator.add(Conv2DTranspose(3,(3,3)))
    generator.add(Activation('tanh'))
    generator.add(Reshape((32,32,3)))
    return generator


def discrim():
    discriminator = Sequential()
    discriminator.add(Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),padding='same',input_shape=(32, 32, 3)))
    discriminator.add(LeakyReLU(0.1))
    discriminator.add(Dropout(0.2))
    discriminator.add(Conv2D(filters=32, kernel_size=(3, 3)))
    discriminator.add(LeakyReLU(0.1))
    discriminator.add(MaxPooling2D((2, 2)))
    discriminator.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.1))
    discriminator.add(Dropout(0.2))
    discriminator.add(Conv2D(filters=32, kernel_size=(3, 3)))
    discriminator.add(LeakyReLU(0.1))
    discriminator.add(MaxPooling2D((2,2)))
    discriminator.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.1))
    discriminator.add(Dropout(0.1))
    discriminator.add(Flatten())
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.1))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.1))
    discriminator.add(Dense(2048))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.1))
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))#probabilité
    discriminator.compile(loss='binary_crossentropy',optimizer=Adam(1e-3,1e-5))
    discriminator.trainable = False
    return discriminator




def train():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train - 127.5) / 127.5
    x_test = (x_test - 127.5) / 127.5

    discriminator = discrim()
    generator = gen2()
    gan_input = Input(shape=(10,))
    fake_image = generator(gan_input)
    gan_output = discriminator(fake_image)
    gan = Model(gan_input, gan_output)  # this is the combined model
    gan.compile(loss='binary_crossentropy', optimizer=Adam(1e-4, 1e-5))
    gan.summary()
    batch_size = 16
    x_train_c = x_train[np.where(y_train == 8)[0]]  # Using ship only
    num_batches = int(len(x_train_c) / batch_size)

    for epoch in range(10):
        for batch in range(num_batches):
            # Select a random batch from x_train_c
            x = x_train_c[np.random.randint(0, len(x_train_c), size=batch_size)]
            # Gaussian noise for the generator model
            noise = np.random.normal(0, 1, size=[batch_size, 10])
            # Generate fake images
            gen_imgs = generator.predict(noise)

            disc_data = np.concatenate([x, gen_imgs])
            # True images are labeled 1, false ones are 0
            labels = [1] * batch_size + [0] * batch_size

            discriminator.trainable = True
            dloss = discriminator.train_on_batch(disc_data, labels)

            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, [1] * batch_size)

            print('\b' * 79 + '\r', end='')
            print('Epoch %d, batch %d/%d: ' % \
                  (epoch + 1, batch, num_batches) + \
                  ' gloss=%.4f,dloss=%.4f' % (gloss, dloss), end='')
            print('')
    generator.save_weights(filepath='weights/geneTConv2D.h5')
    noise = np.random.normal(0, 1, size=[4, 10])
    gen_imgs = generator.predict(noise)
    for i in range(len(noise)):
        plt.imshow((gen_imgs[i] + 1) / 2)
        plt.show()
    return generator

def useModel():

    generator = gen2()
    generator.load_weights('weights/geneTConv2D.h5')

    noise = np.random.normal(0, 1, size=[4, 10])
    gen_imgs = generator.predict(noise)
    for i in range(len(noise)):
        plt.imshow((gen_imgs[i] + 1) / 2)
        plt.show()
    return 0

