from __future__ import print_function
import argparse
import os
import csv
import time
import random

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset 
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from model import Discriminator, Generator
from utils.utility import mkdir_p, generate_image
from utils.plot import plot, flush

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=4, help='number of data loader workers')
parser.add_argument('--batchsize', default=64, type=int, help='input batch size')
parser.add_argument('--imagesize', default=32, type=int, help='the height/ width of the input image to network')
parser.add_argument('--nz', default=100, type=int, help='size of the latent z vector')
parser.add_argument('--ngf', default=64, type=int, help='number of feature channel for generator')
parser.add_argument('--ndf', default=64, type=int, help='number of feature channel for discriminator')
parser.add_argument('--max-epoch', default=25, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--exp-name', required=True, help='name of the experiment')
parser.add_argument('--checkpoints-dir', default='./results', help='name of the experiment')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()

dtype = torch.FloatTensor

configure("tensorboard/" + opt.exp_name)

print('Options:', opt)

try:
	os.mkdir(opt.outf)
except OSError:
	pass

if opt.manualSeed is None:
	opt.manualSeed = random.randint(1, 10000)
print('Random Seed: ', opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
	torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# data loader
if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imagesize),
                                   transforms.CenterCrop(opt.imagesize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imagesize),
                            transforms.CenterCrop(opt.imagesize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imagesize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imagesize, opt.imagesize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3 


netG = Generator(0, nz, ngf)
if opt.netG != '':
	negG.load_state_dict(torch.load(opt.netG))

print('Generator:', netG)

netD = Discriminator(10, ndf)
if opt.netD != '':
	negD.load_state_dict(torch.load(opt.netD))
print('Discriminator: ', netD)

criterion = torch.nn.BCELoss()


label = torch.FloatTensor(opt.batchsize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()

one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
	one = one.cuda()
	mone = mone.cuda()
iter_idx = 0

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
#marginalized entropy
def entropy1(y):
    y1 = Variable(torch.randn(y.size(1)).type(dtype), requires_grad=True)
    y2 = Variable(torch.randn(1).type(dtype), requires_grad=True)
    y1 = y.mean(0)
    y2 = -torch.sum(y1*torch.log(y1+1e-6))

    return y2


# entropy
def entropy2(y):
    y1 = Variable(torch.randn(y.size()).type(dtype), requires_grad=True)
    y2 = Variable(torch.randn(1).type(dtype), requires_grad=True)
    y1 = - y * torch.log(y + 1e-6)

    y2 = 1.0/opt.batchsize * y1.sum()
    return y2

if not os.path.exists(os.path.join(opt.outf, opt.exp_name)):
	os.mkdir(os.path.join(opt.outf, opt.exp_name))

with open(os.path.join(opt.outf, opt.exp_name, 'log.csv'), 'wb') as log:
	log_writer = csv.writer(log, delimiter=',')

	for epoch in xrange(opt.max_epoch):
		start_time = time.time()

		for batch_idx, (real_cpu, real_labels) in enumerate(dataloader):
			# update D network
			for p in netD.parameters():
				p.requires_grad = True
			for p in netG.parameters():
				p.requires_grad = False
			netD.zero_grad()

			# train with real
			if opt.cuda:
				real = Variable(real_cpu.cuda())
			else:
				real = Variable(real_cpu)
			#print(real.size())
			D_real = netD(real, 1)
			# minimize entropy to make certain prediction of real sample
			entropy2_real = entropy2(D_real)
			entropy2_real.backward(one, retain_graph=True)

			# maximize marginalized entropy over real 
			entropy1_real = entropy1(D_real)
			entropy1_real.backward(mone)

			# train with fake
			noise = torch.randn(opt.batchsize, opt.nz, 1, 1)
			if opt.cuda:
				noise = Variable(noise.cuda())
			else:
				noise = Variable(noise)
			fake = netG(noise)

			D_fake = netD(fake, 1)

			# minimize entropy to make uncertain predictions of fake sample
			entropy2_fake = entropy2(D_fake)
			entropy2_fake.backward(mone)

			D_cost = entropy1_real + entropy2_real + entropy2_fake
			optimizerD.step()

			# update G network
			for p in netD.parameters():
				p.requires_grad = False

			for p in netG.parameters():
				p.requires_grad = True

			netG.zero_grad()

			noise = torch.randn(opt.batchsize, opt.nz, 1, 1)
			if opt.cuda:
				noise = Variable(noise.cuda())
			else:
				noise = Variable(noise)
			fake = netG(noise)

			D_fake = netD(fake, 1)

			# fool D to make it believe this samples are real
			entropy2_fake = entropy2(D_fake)
			entropy2_fake.backward(one, retain_graph=True)

			# ensure equal usage of fake samples
			entropy1_fake = entropy1(D_fake)
			entropy1_fake.backward(mone)

			G_cost = entropy2_fake + entropy1_fake
			optimizerG.step()

			D_cost = D_cost.cpu().data.numpy()
			G_cost = G_cost.cpu().data.numpy()
			entropy2_fake = entropy2_fake.cpu().data.numpy()
			entropy2_real = entropy2_real.cpu().data.numpy()

			# monitor the loss
			plot('errD', D_cost, iter_idx)
			plot('errG', G_cost, iter_idx)
			plot('errD_real', entropy2_real, iter_idx)
			plot('errD_fake', entropy2_fake, iter_idx)

			log_value('errD', D_cost, iter_idx)
			log_value('errG', G_cost, iter_idx)
			log_value('errD_real', entropy2_real, iter_idx)
			log_value('errD_fake', entropy2_fake, iter_idx)

			log_writer.writerow([D_cost[0], G_cost[0], entropy2_real[0], entropy2_fake[0]])

			print('iter %d[epoch %d]\t %s %.4f \t %s %.4f \t %s %.4f \t %s %.4f' % (iter_idx, epoch, 
						'errD', D_cost,
						'errG', G_cost,
						'errD_real', entropy2_real,
						'errD_fake', entropy2_fake))
			# checkpointing save
			if iter_idx % 500 == 0:
				torch.save(netG.state_dict(), '%s/netG_lastest.pth' % (os.path.join(opt.checkpoints_dir, opt.exp_name)))
				torch.save(netD.state_dict(), '%s/netD_lastest.pth' % (os.path.join(opt.checkpoints_dir, opt.exp_name)))

			iter_idx += 1

		if epoch % 2 == 0:
			generate_image(epoch, netG, netD)

		if epoch % 20 == 0:
			torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join(opt.checkpoints_dir, opt.exp_name)))
			torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join(opt.checkpoints_dir, opt.exp_name)))
