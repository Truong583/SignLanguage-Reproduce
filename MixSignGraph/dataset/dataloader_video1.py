import os
import cv2
import sys
import glob,gzip
import time
import torch
import random
import pandas
import warnings
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
sys.path.append("..")
global kernel_sizes
from PIL import Image

from utils import clean_phoenix_2014_trans, clean_phoenix_2014
usingpil=False
from utils import video_augmentation

def cut_how2sign(image):
    # print(np.array(image).shape)
    cropped_image = image[150:,400:-300]
    # print(np.array(cropped_image).shape)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([35, 40, 40])
    # upper_green = np.array([85, 255, 255])
    # mask = cv2.inRange(hsv, lower_green, upper_green)
    # mask_inv = cv2.bitwise_not(mask)
    # # foreground = cv2.bitwise_and(image, image, mask=mask_inv)
    # coords = np.column_stack(np.where(mask_inv > 0))
    # x, y, w, h = cv2.boundingRect(coords)
    # cropped_image = image[x-15:x+w+15,y-15:y + h+15]
    # print(x, x+w, y, y+h)
    return cropped_image


def readmp4(path):
    img_list = []
    # print(path, os.path.exists(path))
    videoCap = cv2.VideoCapture(path)
    success, frame = videoCap.read()
    while success:
        success, frame = videoCap.read()
        if success:
            frame = cut_how2sign(frame)
            # frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            # cv2.imshow("1", frame)
            # cv2.waitKey(10)
            img_list.append(frame)
    vid = [cv2.cvtColor(cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4),
                        cv2.COLOR_BGR2RGB) for frame in img_list]
    # for frame in vid:
    #     cv2.imshow("1", frame)
    #     cv2.waitKey(10)
    # if len(vid)<1:
    #     print(path, os.path.exists(path))
    videoCap.release()
    # print("debug==", np.array(vid).shape,  path)
    return vid




class BaseFeeder(data.Dataset):
    def __init__(self, prefix, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train",
                 transform_mode=True,
                 datatype="video", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224,
                 annotation_file=None):
        self.mode = mode
        self.prefix = prefix
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval
        self.image_scale = image_scale
        self.transform_mode = "train" if transform_mode else "test"
        self.load_annotations(annotation_file)
        print(mode, len(self))
        self.data_aug = self.transform()


    def load_annotations(self, annotation_file):
        with gzip.open(annotation_file, 'rb') as f:
            self.inputs_list = pickle.load(f)

    def __getitem__(self, idx):
        if self.data_type == "video":
            input_data, label, fi = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, label['gloss'], label['text'], fi

    def read_video(self, index):
        # load file info
        fi = self.inputs_list[index]
        gloss,text = [], []
        if 'phoenix2014-T' == self.dataset:
            gloss, text = fi['gloss'], fi['text']
            text, gloss = clean_phoenix_2014_trans(text), clean_phoenix_2014_trans(gloss)
            img_folder = os.path.join(self.prefix, "features/fullFrame-210x260px/" + fi['name'])
            img_list = sorted(glob.glob(img_folder + "/*.png"))
        elif self.dataset == 'phoenix2014':
            gloss = clean_phoenix_2014(fi['gloss'])
            img_folder = os.path.join(self.prefix, "features/" + fi['name'])
            img_list = sorted(glob.glob(img_folder + "*.png"))
        elif self.dataset == 'CSL-Daily':
            if fi['name'] == "S000005_P0004_T00":
                fi['name'] = "S000007_P0003_T00"
            img_folder = os.path.join(self.prefix, "sentence/frames_512x512/" + fi['name'])
            img_list = sorted(glob.glob(img_folder + "/*.jpg"))
            gloss, text = fi['gloss'], fi['text']
        elif self.dataset == "how2sign":
            ppre = "val" if self.mode=="dev" else self.mode
            img_folder = os.path.join(self.prefix, ppre+"_rgb_front_clips/raw_videos/" + fi['name']+".mp4")
            gloss, text = fi['gloss'], fi['text']

        if self.dataset != "how2sign":
            if len(img_list)<1:
                print("debu_dataset==", fi, img_list)
            img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]

        if len(text)>0:
            text = text.strip('\n').strip().replace("  ", " ").lower()
        gloss = gloss.strip('\n').strip().replace("  ", " ").lower()
        label = {'gloss': gloss, 'text': text}
        # if self.dataset== 'CSL-Daily':
        #     if usingpil:
        #         vid = [Image.open(img_path).crop((0,80,512,512)).resize((256, 256)) for img_path in img_list]
        #     else:
        #         vid = [cv2.cvtColor(cv2.resize(cv2.imread(img_path)[80:512, 0:512], (256, 256), interpolation=cv2.INTER_LANCZOS4),
        #                             cv2.COLOR_BGR2RGB) for img_path in img_list]
        #     return vid, label, fi

        if usingpil:
            vid = [Image.open(img_path).resize((256, 256)) for img_path in img_list]
        else:
            if self.dataset == "how2sign":
                vid = readmp4(img_folder)
                if len(vid)<1:
                    return self.read_video(index+1)
            else:
                vid =[cv2.cvtColor(cv2.resize(cv2.imread(img_path), (256, 256), interpolation=cv2.INTER_LANCZOS4),
                                 cv2.COLOR_BGR2RGB) for img_path in img_list]
        # max_gloss_len= int((((len(vid)-4)/2)-4)/2)
        max_gloss_len = int((len(vid)-4)/2)
        if len(gloss.split())>=max_gloss_len:
            # print(gloss, len(gloss.split()), max_gloss_len, len(vid),  int((len(vid)-4)/2) )
            gloss = " ".join(gloss.split()[:max_gloss_len])
            label["gloss"] = gloss
        return vid, label, fi

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.input_size),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            if self.dataset=="how2sign":
                return video_augmentation.Compose([
                    video_augmentation.CenterCrop(self.input_size),
                    video_augmentation.Resize(self.input_size),
                    video_augmentation.ToTensor(),
                    video_augmentation.TemporalRescale1(0.2, self.frame_interval),
                ])
            else:
                return video_augmentation.Compose([
                        video_augmentation.CenterCrop(self.input_size),
                        video_augmentation.Resize(self.input_size),
                        video_augmentation.ToTensor(),
                    ])

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, gloss, text, info = list(zip(*batch))

        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes
        for layer_2idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride
                left_pad += int((int(ks[1]) - 1) / 2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor(
                [np.ceil(len(vid) / total_stride) * total_stride + 2 * left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        return padded_video, video_length, gloss, text, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    path = "D:/dataset/how2sign/train_rgb_front_clips/raw_videos/"
    name = "caSH1HZRhZA_28-5-rgb_front.mp4"
    vid=readmp4(path+name)
    print("hh=", vid)

    feeder = BaseFeeder("/data1/gsw/how2sign/", dataset='how2sign', drop_ratio=1, num_gloss=-1, mode="test",
                 transform_mode=True,
                 datatype="video", frame_interval=1, image_scale=1.0, kernel_size= ['K5', "P2"], input_size=224,
                 annotation_file="../../Dataprocessing/how2sign/how2sign_test.test")
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=feeder.collate_fn
    )
    for da in dataloader:
        video, v_l, gloss, text , info=da
        # print(video)
        # print(video.shape, v_l, gloss, text)

'''
torch.Size([2, 120, 3, 224, 224]) tensor([120,  56]) 
('jetzt wetter voraussage morgen freitag vier zwanzig dezember', 'lieb zuschauer gut abend') 
('und nun die wettervorhersage für morgen freitag den vierundzwanzigsten dezember .', 'liebe zuschauer guten abend .')
torch.Size([2, 200, 3, 224, 224]) tensor([200,  52]) ('suedwest deutsch land bisschen feucht kommen ix moeglich schauer gewitter wahrscheinlich', 'suedost meistens trocken') ('in die südwestlichen teile deutschlands sickert etwas feuchterer luft damit steigt dort das schauer und gewitterrisiko leicht .', 'im südosten noch meist trocken .')
torch.Size([2, 156, 3, 224, 224]) tensor([156,  44]) ('suedost schnee region trocken minus acht bis plus eins zwischen', 'region mehr sonne') ('im südosten schneit es etwas sonst ist es trocken bei minus acht bis plus ein grad .', 'sonst neben dichten wolken hier und da etwas sonne .')
^CTraceback (most recent call last):


'''