import torch
import torchvision
import folders

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True, isFull=False):

        self.batch_size = batch_size
        self.istrain = istrain

        if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'livec'):
            # Train transforms
            if isFull:
                if istrain:
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(p=0.5),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))
                    ])
                # Test transforms
                else:
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))
                    ])
            else:
                if istrain:
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomCrop(size=patch_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))
                    ])
                # Test transforms
                else:
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomCrop(size=patch_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))
                    ])
        elif dataset == 'koniq-10k':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((384, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((384, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        elif dataset == 'live-fb':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop((600, 600)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])

        if dataset == 'live':
            self.data = folders.LIVEFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'livec':
            self.data = folders.LIVEChallengeFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'csiq':
            self.data = folders.CSIQFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'koniq-10k':
            self.data = folders.Koniq_10kFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'live-fb':
            self.data = folders.LiveFBFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)

    def get_data(self, sampler=None):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, sampler=sampler, pin_memory=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, sampler=sampler, pin_memory=True)
        return dataloader
