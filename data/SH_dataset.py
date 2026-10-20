
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat
from torch.utils.data import DataLoader
import tifffile
import matplotlib.pyplot as plt
import sys
# sys.path.append('./data')
sys.path.append('./')
from data.base_dataset import BaseDataset, get_params, get_transform
from options.train_options import TrainOptions
from data.pix2pix_dataset import Pix2pixDataset
from torchvision.utils import save_image


torch.set_printoptions(profile="full")
# 使图像均值为0，标准差为1
normalize = transforms.Normalize((0.5,), (0.5,))

train_transform = transforms.Compose([
    transforms.Resize((256, 256), antialias=False),
    # normalize
    ])

# 训练集
class SHDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='scale_width_and_crop')
        parser.set_defaults(no_pairing_check=True)
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(label_nc=1)
        # parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser


    def __init__(self, opt, data_path, transform=train_transform, train=True, split=0.9):
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.physics_cfg = None
        image_dir = getattr(opt, 'sh_image_dir', 'images')
        if getattr(opt, 'lambda_phys', 0.0) > 0:
            from models.physics import load_config
            self.physics_cfg = load_config(opt.physics_config)

        # 读取数据
        self.images = os.listdir(os.path.join(data_path, image_dir))
        self.labels = os.listdir(data_path + '/targets')
        self.ref = os.listdir(data_path + '/ref')
        self.label_ref = os.listdir(data_path + '/label_ref')

        # 取出文件中的数字字符，然后对数字进行排序
        self.images.sort(key=lambda x: int(x.split(".")[0]))
        self.labels.sort(key=lambda x: int(x.split(".")[0]))
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        # 查看图片与标签数量是否一致
        if self.physics_cfg is not None:
            # images_sys can be a subset. Pair by ID, never sorted list position.
            targets = {os.path.splitext(name)[0]: name for name in self.labels}
            stems = [os.path.splitext(name)[0] for name in self.images]
            if len(set(stems)) != len(stems) or any(s not in targets for s in stems):
                raise ValueError('SH image IDs must be unique and have matching phase targets')
            self.labels = np.array([targets[s] for s in stems])
        else:
            assert len(self.images) == len(self.labels), 'Number does not match'

        self.transform = transform  # 定义做何种变换

        # 将数据与标签连接在一起，为了之后可以索引
        self.images_and_labels = []  # 创建一个空列表

        for i in range(len(self.images)):  # 往空列表里装东西，为了之后索引
            self.images_and_labels.append(
                (os.path.join(data_path, image_dir, self.images[i]), data_path + '/targets/' + self.labels[i])
            )

        self.images_and_labels = np.array(self.images_and_labels)
        n=len(self.images_and_labels)
        if self.train:
            self.data=self.images_and_labels[:int(split*n)]
        else:
            self.data=self.images_and_labels[int(split*n):]


    def __getitem__(self, index):  # 读取数据和标签，并返回数据和标签
        # 读取数据
        image_path, label_path = self.data[index]

        # 读取ref(exampler zero_phase)
        ref_path = self.data_path + '/ref/' + str(self.ref)[2:-2]
        ref = tifffile.imread(ref_path)  # 读取tif为array
        ref = torch.from_numpy(ref).unsqueeze(dim=0).float()/255
        # ref = Image.open(ref_path)  # 读取tif为array
        # params = get_params(self.opt, ref.size)
        # transform_image = get_transform(self.opt, params, normalize=False)
        # ref = transform_image(ref)

        # 读取图片(distorted_SH)
        image = tifffile.imread(image_path)  # 读取tif为array
        image = torch.from_numpy(image).unsqueeze(dim=0).float()
        physics_data = {}
        if self.physics_cfg is not None:
            cfg = self.physics_cfg
            if cfg['amplitude']['kind'] == 'per_sample':
                from models.physics import read_map
                if not self.opt.physics_amplitude_dir:
                    raise ValueError('Missing per-sample amplitude directory')
                stem = os.path.splitext(os.path.basename(image_path))[0]
                physics_data['physics_amplitude'] = read_map(os.path.join(
                    self.opt.physics_amplitude_dir, stem + '.npy'))[0].float()
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
        # print(torch.max(image))
        # print(torch.min(image))

        # 读取label_ref(sys SH)
        label_ref_path = self.data_path + '/label_ref/' + str(self.label_ref)[2:-2]
        label_ref = tifffile.imread(label_ref_path)  # 读取tif为array
        label_ref = torch.from_numpy(label_ref).unsqueeze(dim=0).float()
        label_ref = (label_ref - torch.min(label_ref)) / (torch.max(label_ref) - torch.min(label_ref))

        # 读取label(distorted_phase)
        # label = tifffile.imread(label_path)  # 读取tif为array
        # label = torch.from_numpy(label).unsqueeze(dim=0).float()/255
        label = loadmat(label_path)
        key_name = list(label.keys())[3]
        label = torch.from_numpy(label[str(key_name)]).unsqueeze(0).float()
        label=label/50

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
            label_ref = self.transform(label_ref)
            ref = self.transform(ref)

        if torch.isnan(image).any():
            print('nan_img')
        if torch.isnan(label).any():
            print('nan_label')

        self_ref_flag = 1.0

        input_dict = {'label': image,
                      'image': label,
                      'path': label_path,
                      'self_ref': self_ref_flag,
                      'ref': ref,
                      'label_ref': label_ref
                      }
        input_dict.update(physics_data)
        return input_dict

    def __len__(self):  # 必须写，返回数据集的长度
        return len(self.data)

if __name__ == '__main__':
    data_path = '/home/welab/data/welabdata/lc/AOdata/20240127/flat'
    # parse options
    opt = TrainOptions().parse()
    test_dataset = SHDataset(opt, data_path, train=True, split=1)
    print(test_dataset.__len__())
    test_dataset.initialize(opt)
    train_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    for batch, data_i in enumerate(train_dataloader):
        if torch.isnan(data_i['label']).any():
            print(print("batch: {:.1f}  ".format(batch)))
            print('nan_img')
        label = data_i['label']
        ref = data_i['ref']
        image = data_i['image']*50
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
        imgs = torch.cat((label.cpu(), ref.cpu(), \
                          image.cpu()), 0)
        # save_root = '/home/lc/Data/AO-data/aotest/'
        # save_name = os.path.join(save_root, '%05d.tiff' % batch)
        # save_image(imgs, save_name, nrow=6, padding=0, normalize=False)
        # y = y.squeeze(dim=1)
        # tensor数据显示为灰度图像
        # img = x.numpy()
        # img = x[0]
        # img = np.transpose(img, (1, 2, 0))
        # plt.imshow(img, cmap="gray")
        # plt.axis('off')
        # plt.show()
