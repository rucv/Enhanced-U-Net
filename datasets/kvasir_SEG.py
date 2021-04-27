#Adopted from ACSNet
import os
import os.path as osp
from utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms


# KavSir-SEG Dataset
class kvasir_SEG(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(kvasir_SEG, self).__init__()
        data_path = osp.join(root, data2_dir)
        self.imglist = []
        self.gtlist = []

        datalist = os.listdir(osp.join(data_path, 'images'))
        for data in datalist:
            self.imglist.append(osp.join(data_path+'/images', data))
            self.gtlist.append(osp.join(data_path+'/masks', data))

        if transform is None:
            if mode == 'train':
               transform = transforms.Compose([
                   Resize((320,320 )),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   Translation(10),
                   RandomCrop((256, 256)),
                   ToTensor(),

               ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((320, 320)),
                   ToTensor(),
               ])
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imglist[index]
        gt_path = self.gtlist[index]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        data = {'image': img, 'label': gt}
        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.imglist)


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406],
                                 #[0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
