import os
import torch
import numpy as np
from imageio import imread
from PIL import Image
import glob
from termcolor import colored, cprint
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets
from transformers import BertTokenizer,BertweetTokenizer
import random

from preprocess import clean_text
from base_dataset import BaseDataset, scale_shortside
from paths import dataroot


a=15
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(a)
#numpy.random.seed(1)
torch.manual_seed(a)
torch.cuda.manual_seed(a)


task_dict = {
    'task1': 'informative',
    'task2': 'humanitarian',
    'task2_merged': 'humanitarian',
    'task3' : 'damage'
}

labels_task1 = {
    'informative': 1,
    'not_informative': 0
}

labels_task2 = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 6,
    'missing_or_found_people': 7,
}

labels_task2_merged = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 5,
    'missing_or_found_people': 5,
}

labels_task3={
    'severe_damage': 0,
    'mild_damage': 1,
    'little_or_no_damage': 2,
}



class CrisisMMDataset(BaseDataset):

    def read_data(self, ann_file):
        with open(ann_file, encoding='utf-8') as f:
            self.info = f.readlines()[1:]

        self.data_list = []

        for l in self.info:
            l = l.rstrip('\n')
            if self.task != "task3":
                event_name, tweet_id, image_id, tweet_text,image,	label,	label_text,	label_image, label_text_image = l.split('\t')
                self.data_list.append(
                    {
                        'path_image': '%s/%s' % (self.dataset_root, image),

                        'text': tweet_text,
                        'text_tokens': self.tokenize(tweet_text),

                        'label_str': label,
                        'label': self.label_map[label],

                        'label_image_str': label_image,
                        'label_image': self.label_map[label_image],

                        'label_text_str': label_text,
                        'label_text': self.label_map[label_text]
                    }
                )
            elif self.task == "task3":
                event_name, tweet_id, image_id, tweet_text,image,label = l.split('\t')
                self.data_list.append(
                    {
                        'path_image': '%s/%s' % (self.dataset_root, image),

                        'text': tweet_text,
                        'text_tokens': self.tokenize(tweet_text),

                        'label_str': label,
                        'label': self.label_map[label],

                        'label_image_str': label,
                        'label_image': self.label_map[label],

                        'label_text_str': label,
                        'label_text': self.label_map[label]
                    }
                )		  
		    

    def tokenize(self, sentence):
        ids = self.tokenizer(clean_text(
            sentence), padding='max_length', max_length=40, truncation=True).items()
        return {k: torch.tensor(v) for k, v in ids}

    def initialize(self, opt, phase='train', cat='all', task='task2', shuffle=False):
        self.opt = opt
        self.shuffle = shuffle

        self.dataset_root = f'{dataroot}/CrisisMMD_v2.0_toy' if opt.debug else f'{dataroot}/CrisisMMD_v2.0'
        self.image_root = f'{self.dataset_root}/data_image'
        self.task=task
        self.label_map = None
        if task == 'task1':
            self.label_map = labels_task1
        elif task == 'task2':
            self.label_map = labels_task2
        elif task== 'task2_merged':
            self.label_map = labels_task2_merged
        elif task == 'task3':
            self.label_map = labels_task3

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        ann_file = '%s/crisismmd_datasplit_all/task_%s_text_img_%s.tsv' % (
            self.dataset_root, task_dict[task], phase
        )

        # Append list of data to self.data_list
        self.read_data(ann_file)

        if self.shuffle:
            np.random.default_rng(seed=0).shuffle(self.data_list)
        self.data_list = self.data_list[:self.opt.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.data_list)), 'yellow')

        self.N = len(self.data_list)

        self.to_tensor = transforms.ToTensor()
        #self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

        self.transforms = transforms.Compose([
            # transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, Image.BICUBIC)),
            transforms.Lambda(lambda img: scale_shortside(
                img, opt.load_size, opt.crop_size, Image.BICUBIC)),
            # transforms.Resize((opt.crop_size, opt.crop_size)),
            transforms.RandomCrop(opt.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        data = self.data_list[index]
        if 'image' not in data:
            with Image.open(data['path_image']).convert('RGB') as img:
                image = self.transforms(img)
            data['image'] = image

        return data

    def __len__(self):
        return len(self.data_list)

    def name(self):
        return 'CrisisMMDataset'



if __name__ == '__main__':

    opt = object()

    dset = CrisisMMDataset(opt, 'train')
    import pdb
    pdb.set_trace()
