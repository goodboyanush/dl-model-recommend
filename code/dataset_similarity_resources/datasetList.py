import torch
from torchvision import datasets, transforms
from zipfile import ZipFile
import os


# Use this abstract class to write generic extensible functions - currently still under construction
class AbstractDataset():
    def make_weights_for_balanced_classes(images, nclasses):                        
        count = [0] * nclasses                                                      
        for item in images:                                                         
            count[item[1]] += 1                                                     
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))                                                   
        for i in range(nclasses):                                                   
            weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(images)                                              
        for idx, val in enumerate(images):                                          
            weight[idx] = weight_per_class[val[1]]                                  
        return weight

    def create_sampler():
        dataset_train = datasets.ImageFolder(traindir)                                                                                                                                                   
        # For unbalanced dataset we create a weighted sampler                       
        weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))                                                                
        weights = torch.DoubleTensor(weights)                                       
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True,                              
                                                             sampler = sampler, num_workers=args.workers, pin_memory=True)    


class NewDataset():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt['workers'],
		  'batch_size' : opt['batchSize'],
		  'shuffle' : True,
		  'pin_memory': True}
		dataTransforms = {
			'train': transforms.Compose([
				transforms.Resize(opt['size']),
				#transforms.RandomResizedCrop(opt['inpSize']),
				#transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]),
			'val': transforms.Compose([
				transforms.Resize((opt['size'],opt['size'])),
				#transforms.CenterCrop(opt['inpSize']),
				transforms.ToTensor(),
				#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		}
		dataDir = opt['dataDir']
		self.dataLoader = torch.utils.data.DataLoader(datasets.ImageFolder(dataDir, dataTransforms['train']), **kwargs)


class NewDatasetZip():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt['workers'],
		  'batch_size' : opt['batchSize'],
		  'shuffle' : True,
		  'pin_memory': True}
		dataTransforms = {
			'train': transforms.Compose([
				transforms.Resize(opt['size']),
				#transforms.RandomResizedCrop(opt['inpSize']),
				#transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]),
			'val': transforms.Compose([
				transforms.Resize((opt['size'],opt['size'])),
				#transforms.CenterCrop(opt['inpSize']),
				transforms.ToTensor(),
				#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		}
		dataDir = opt['dataDir']
		basename = os.path.basename(dataDir).split('.')[0]
		self.newdirname = os.path.join(os.path.dirname(dataDir), basename)
		if not os.path.exists(self.newdirname):
			os.makedirs(self.newdirname)
		with ZipFile(dataDir, 'r') as zip_f:
			zip_f.extractall(self.newdirname)
		self.dataLoader = torch.utils.data.DataLoader(datasets.ImageFolder(self.newdirname, dataTransforms['train']), **kwargs)


########### Classes for offline learning and tasks ###############

class MNIST():
    '''
    Downloads and loads the MNIST dataset.
    Preprocessing -> Data is normalized in Transforms.
    '''
    def __init__(self, opt):
        kwargs = {
          'num_workers': opt['workers'],
          'batch_size' : opt['batchSize'],
          'shuffle' : True,
          'pin_memory': True}
        print('=> Loading MNIST...')

        self.trainLoader = torch.utils.data.DataLoader(
            datasets.MNIST(opt['dataDir'] + "mnist/", train=True, download=True,
                    transform=transforms.Compose([
                        #transforms.RandomCrop(28, padding=4),
                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
						transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                        #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
             **kwargs)

        self.valLoader = torch.utils.data.DataLoader(
            datasets.MNIST(opt['dataDir'] + "mnist/", train=False,
              transform=transforms.Compose([
                           transforms.ToTensor(),
						   transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
              **kwargs)

class CIFAR10():
    def __init__(self, opt):
        kwargs = {
          'num_workers': opt['workers'],
          'batch_size' : opt['batchSize'],
          'shuffle' : True,
          'pin_memory': True}

        print('=> Loading CIFAR10...')

        self.trainLoader = torch.utils.data.DataLoader(
            datasets.CIFAR10(opt['dataDir'] + "cifar10/", train=True, download=True,
                    transform=transforms.Compose([
                        #transforms.RandomCrop(32, padding=4),
                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
							#std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                       ])),
             **kwargs)

        self.valLoader = torch.utils.data.DataLoader(
            datasets.CIFAR10(opt['dataDir'] + "cifar10/", train=False,
              transform=transforms.Compose([
                           transforms.ToTensor(),
                        	#transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
								#std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                       ])),
          **kwargs)

class CIFAR100():
    def __init__(self, opt):
        kwargs = {
          'num_workers': opt['workers'],
          'batch_size' : opt['batchSize'],
          'shuffle' : True,
          'pin_memory': True}

        print('=> Loading CIFAR100...')

        self.trainLoader = torch.utils.data.DataLoader(
            datasets.CIFAR100(opt['dataDir'] + "cifar100/", train=True, download=True,
                    transform=transforms.Compose([
                       #transforms.RandomCrop(32, padding=4),
                       #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                       #transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
							#std=[x/255.0 for x in [68.2, 65.4, 70.4]])
                       ])),
             **kwargs)

        self.valLoader = torch.utils.data.DataLoader(
            datasets.CIFAR100(opt['dataDir'] + "cifar100/", train=False,
              transform=transforms.Compose([
                           transforms.ToTensor(),
                          #transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
							#std=[x/255.0 for x in [68.2, 65.4, 70.4]])
                       ])),
          **kwargs)

class STL():
    def __init__(self,opt):
        kwargs = {
          'num_workers': opt['workers'],
          'batch_size' : opt['batchSize'],
          'shuffle' : True,
          'pin_memory': True}

        print('=> Loading STL...')

        self.trainLoader = torch.utils.data.DataLoader(
            datasets.STL10(
                root=opt['dataDir'] + "stl10/", split='train', download=True,
                transform=transforms.Compose([
                    #transforms.Pad(4),
                    #transforms.RandomCrop(96),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                ])),
             **kwargs)
        self.valLoader = torch.utils.data.DataLoader(
            datasets.STL10(
                root=opt['dataDir'] + "stl10/", split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                ])),
             **kwargs)

class SVHN():
    def __init__(self,opt):
        kwargs = {
          'num_workers': opt['workers'],
          'batch_size' : opt['batchSize'],
          'shuffle' : True,
          'pin_memory': True}

        print('=> Loading SVHN...')

        self.trainLoader = torch.utils.data.DataLoader(
            datasets.SVHN(opt['dataDir'] + "svhn/", split='train', download=True,
                transform=transforms.Compose([
                    #transforms.Pad(4),
                    #transforms.RandomCrop(96),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                ])),
                
             **kwargs)

        self.valLoader = torch.utils.data.DataLoader(
            datasets.SVHN(opt['dataDir'] + "svhn/", split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
             **kwargs)
        
class FashionMNIST():
    def __init__(self,opt):
        kwargs = {'num_workers': opt['workers'],
          'batch_size' : opt['batchSize'],
          'shuffle' : True,
          'pin_memory': True}


        print('=> Loading FashionMNIST...')

        self.trainLoader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                root=opt['dataDir'] + "fmnist/", train=True, download=True,
                transform=transforms.Compose([
                    #transforms.Pad(4),
                    #transforms.RandomCrop(96),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
             **kwargs)

        self.valLoader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                root=opt['dataDir'] + "fmnist/", train=False, download=True,
                transform=transforms.Compose([
                    #transforms.Pad(4),
                    #transforms.RandomCrop(96),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
             **kwargs)