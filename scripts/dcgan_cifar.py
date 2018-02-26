'''
DCGAN on CIFAR10 using Keras
Author: Philip Ball with thanks to Rowel Atienza
'''

import numpy as np
import time
from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
import glob
import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
	def __init__(self, img_rows=32, img_cols=32, channel=3):

		self.img_rows = img_rows
		self.img_cols = img_cols
		self.channel = channel
		self.D = None	# discriminator
		self.G = None	# generator
		self.AM = None	# adversarial model
		self.DM = None	# discriminator model

	def discriminator(self):
		if self.D:
			return self.D
		self.D = Sequential()
		depth = 64
		dropout = 0.4
		# In: 28 x 28 x 1, depth = 1
		# Out: 14 x 14 x 1, depth=64
		input_shape = (self.img_rows, self.img_cols, self.channel)
		self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
		#self.D.add(BatchNormalization(momentum=0.9))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
		#self.D.add(BatchNormalization(momentum=0.9))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		self.D.add(Conv2D(depth*8, 3, strides=1, padding='same'))
		#self.D.add(BatchNormalization(momentum=0.9))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		# Out: 1-dim probability
		self.D.add(Flatten())
		self.D.add(Dense(1))
		#self.D.add(BatchNormalization(momentum=0.9))
		self.D.add(Activation('sigmoid'))
		#self.D.summary()
		return self.D

	def generator(self):
		if self.G:
			return self.G
		self.G = Sequential()
		dropout = 0.4
		depth = 1024
		dim = 4
		# In: 100
		# Out: dim x dim x depth
		self.G.add(Dense(dim*dim*depth, input_dim=100))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))
		self.G.add(Reshape((dim, dim, depth)))
		self.G.add(Dropout(dropout))

		# In: dim x dim x depth
		# Out: 2*dim x 2*dim x depth/2
		self.G.add(UpSampling2D())
		self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		self.G.add(UpSampling2D())
		self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		self.G.add(UpSampling2D())
		self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image (-1.0,1.0) per pix
		self.G.add(Conv2DTranspose(3, 5, padding='same'))
		self.G.add(Activation('tanh'))
		self.G.summary()
		return self.G

	def discriminator_model(self):
		if self.DM:
			return self.DM
		optimizer = Adam(lr=0.0001, beta_1 = 0.2)
		self.DM = Sequential()
		self.DM.add(self.discriminator())
		self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.DM

	def adversarial_model(self):
		if self.AM:
			return self.AM
		optimizer = Adam(lr=0.0001, beta_1 = 0.2)
		self.AM = Sequential()
		self.AM.add(self.generator())
		for layer in self.discriminator().layers:
			layer.trainable = False
		self.AM.add(self.discriminator())
		self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.AM


class CIFAR_DCGAN(object):
	def __init__(self, meme = False):
		self.img_rows = 32
		self.img_cols = 32
		self.channel = 3
		self.meme = meme
		if self.meme:
			self.x_train = self.load_memes()
		else:
			(self.x_train, _), (_, _) = cifar10.load_data()
		self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, self.channel).astype(np.float32)/127.5 - 1
		self.DCGAN = DCGAN()
		self.discriminator =  self.DCGAN.discriminator_model()
		self.adversarial = self.DCGAN.adversarial_model()
		self.generator = self.DCGAN.generator()

	def load_memes(self):
		meme_list = glob.glob('./32x32/*.jpg')
		memes = np.array([np.array(list(Image.open(fname).getdata())) for fname in meme_list])
		return memes

	def train(self, epochs=30, batch_size=64, save_interval=0):
		noise_input = None
		self.batch_size = batch_size
		if save_interval>0:
			noise_input = np.random.normal(0, 1, size=[16, 100])
		for i in range(epochs):
			np.random.shuffle(self.x_train)
			num_samps = int(self.x_train.shape[0]//float(batch_size))
			for j in range(num_samps):
				start = j*self.batch_size
				end = (j+1)*self.batch_size
				samples = self.x_train[start:end]
				self.train_batch(samples, i, j)
			if save_interval>0:
				if (i+1)%save_interval==0:
					self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))
					self.save_model(i+1)

	def train_batch(self, samples, epoch, batch):
		images_train = samples
		noise = np.random.normal(0., 1., size=[len(samples), 100])
		images_fake = self.generator.predict(noise)
		x = np.concatenate((images_train, images_fake))
		y_valid = np.ones([len(samples), 1])
		y_fake = np.zeros([len(samples), 1])
		y = np.concatenate((y_valid, y_fake))
		# train discriminator
		d_loss = self.discriminator.train_on_batch(x, y)
		# train generator
		y = np.ones([len(samples), 1])
		noise = np.random.normal(0., 1., size=[len(samples), 100])
		a_loss = self.adversarial.train_on_batch(noise, y)
		log_mesg = "Epcoh %d: Batch %d: [D loss: %f, acc: %f]" % (epoch+1, batch+1, d_loss[0], d_loss[1])
		log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
		print(log_mesg)


	def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
		if self.meme:
			im_type = 'meme'
		else:
			im_type = 'cifar'

		filename = im_type + '.png'
		if fake:
			if noise is None:
				noise = np.random.normal(0, 1, size=[samples, 100])
			else:
				filename = im_type + "_epoch_%d.png" % step
			images = self.generator.predict(noise)
		else:
			i = np.random.randint(0, self.x_train.shape[0], samples)
			images = self.x_train[i, :, :, :]

		plt.figure(figsize=(10,10))
		images = (images + 1)*127.5
		images = images.astype(np.uint8)
		for i in range(images.shape[0]):
			plt.subplot(4, 4, i+1)
			image = images[i, :, :, :]
			image = np.reshape(image, [self.img_rows, self.img_cols, self.channel])
			plt.imshow(image)
			plt.axis('off')
		plt.tight_layout()
		if save2file:
			plt.savefig(filename)
			plt.close('all')
		else:
			plt.show()

	def save_model(self, epoch):
        
		def save(model, model_name, epoch):
			model_path = "./saved_model/%s_%d.json" % (model_name,epoch)
			weights_path = "./saved_model/%s_weights_%d.hdf5" % (model_name,epoch)
			options = {"file_arch": model_path, 
						"file_weight": weights_path}
			json_string = model.to_json()
			open(options['file_arch'], 'w').write(json_string)
			model.save_weights(options['file_weight'])

		if self.meme:
			m_name = 'dcgan_meme_gen'
		else:
			m_name = 'dcgan_cifar_gen'

		save(self.generator, m_name, epoch)

if __name__ == '__main__':
    cifar_dcgan = CIFAR_DCGAN(meme = False)
    timer = ElapsedTimer()
    cifar_dcgan.train(epochs=40, batch_size=32, save_interval=1)
    timer.elapsed_time()
