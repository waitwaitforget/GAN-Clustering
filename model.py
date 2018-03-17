import torch.nn as nn
import torch.nn.functional as F 


# custom weights initialization called on netG and netD

class Discriminator(nn.Module):

	def __init__(self, k, nf=64):
		super(Discriminator, self).__init__()

		self.k = k

		def add_conv_batch_relu(m, i, nf1, nf2, last=False):
			if not last:
				m.add_module('conv%d'%i, nn.Conv2d(nf1, nf2, 4, 2, 1, bias=False))
			else:
				m.add_module('conv%d'%i, nn.Conv2d(nf1, nf2, 4))
			m.add_module('bn%d'%i, nn.BatchNorm2d(nf2))
			m.add_module('relu%d'%i, nn.LeakyReLU(0.2, inplace=True))
			m.add_module('drop%d'%i, nn.Dropout(0.5))

		feature = nn.Sequential()

		add_conv_batch_relu(feature, 1, 3, nf)
		add_conv_batch_relu(feature, 2, nf, nf*2)
		add_conv_batch_relu(feature, 3, nf*2, nf*4)
		add_conv_batch_relu(feature, 4, nf*4, nf*4, True)
		#feature.add_module('conv5', nn.Conv2d(nf*4, k, 1))

		softmax = nn.Linear(256, k)

		classifier = nn.Linear(256, 1)

		self.feature = feature
		self.softmax = softmax
		self.classifier = classifier


	def forward(self, x, mode):
		feat = self.feature(x)
		
		feat = feat.view(x.size(0), -1)
		if mode == 1:
			return F.softmax(self.softmax(feat))
		else:
			return self.classifier(feat)


class Generator(nn.Module):

	def __init__(self, k, zdim=20, nf=64):
		super(Generator, self).__init__()

		self.k = k

		def add_convt_bn_relu(m, i, nf1, nf2):
			if i == 1:
				m.add_module('transcov%d'%i, nn.ConvTranspose2d(nf1, nf2, 4, 1, 0, bias=False))
			else:
				m.add_module('transcov%d'%i, nn.ConvTranspose2d(nf1, nf2, 4, 2, 1, bias=False))

			m.add_module('bn%d'%i, nn.BatchNorm2d(nf2))
			m.add_module('relu%d'%i,  nn.LeakyReLU(0.2, inplace=True))

		generator = nn.Sequential()
		
		add_convt_bn_relu(generator, 1, k+zdim, nf*4)
		add_convt_bn_relu(generator, 2, nf*4, nf*2)
		add_convt_bn_relu(generator, 3, nf*2, nf)
		generator.add_module('transcov4', nn.ConvTranspose2d(nf, 3, 4,2,1, bias=False))
		generator.add_module('tanh', nn.Tanh())

		self.main = generator

	def forward(self, x):
		return self.main(x)


def unittest():
	D = Discriminator(10)
	G = Generator(10)
	import torch
	from torch.autograd import Variable
	x = Variable(torch.rand(2, 3, 32,32))
	p = D(x,1)
	print p 
	
	print D.feature

if __name__=='__main__':
	unittest()