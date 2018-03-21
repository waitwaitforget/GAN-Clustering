import torch
import torch.utils.data
import torchvision.datasets as dset 
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
from sklearn import metrics


def cluster_acc(Y, Y_pred):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in xrange(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


def evaluate(netD, dataloader, cuda):

	real_labels = []
	pred_labels = []
	netD.eval()
	if cuda:
		netD.cuda()

	for data in dataloader:
		imgs, labels = data
		imgs = Variable(imgs)
		if cuda:
			imgs = imgs.cuda()

		real_labels.append(labels)

		pred = netD(imgs, 1)
		pred_labels.append(pred.data.cpu().max(1)[1])

	real_labels = torch.cat(real_labels)
	pred_labels = torch.cat(pred_labels)

	ARI = metrics.adjusted_rand_score(real_labels.numpy(), pred_labels.numpy())
	NMI = metrics.normalized_mutual_info_score(real_labels.numpy(), pred_labels.numpy())
	ACC, _ = cluster_acc(real_labels.numpy(), pred_labels.numpy())

	return NMI, ACC, ARI





def unittest():
	from model import Discriminator
	netD = Discriminator(10, 64)

	state_dict = torch.load('./results/catgan_v1/netD_lastest.pth')
	netD.load_state_dict(state_dict)

	dataset = dset.CIFAR10(root='./data', download=True,
							#train=False,
                           transform=transforms.Compose([
                               transforms.Scale(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=True, num_workers=int(4))
	NMI, ACC, ARI = evaluate(netD, dataloader, True)

	print('clustering on cifar10: NMI {:.4f}, ACC {:.4f}, ARI {:.4f}'.format(NMI, ACC, ARI))

#if __name__=='__main__':
#	unittest()