import glob,gzip
import time, torch, random,pandas
import warnings
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch.utils.data as data
global kernel_sizes


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train",
                 transform_mode=True,
                 datatype="video", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224,
                 annotation_file=None):
        self.mode = mode
        self.prefix = prefix
        self.data_type = datatype
        self.dataset = dataset.lower()
        self.input_size = input_size
        global kernel_sizes
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval
        self.image_scale = image_scale
        self.transform_mode = "train" if transform_mode else "test"
        self.load_annotations(annotation_file)
        print(mode, len(self))
        self.load_pose()


    def load_annotations(self, annotation_file):
        with gzip.open(annotation_file, 'rb') as f:
            self.inputs_list = pickle.load(f)

    def load_pose(self):
        with open(self.prefix + self.mode+'_keypoints.pkl', "rb") as f:
            self.name2keypoints = pickle.load(f)
            f.close()

    def __getitem__(self, idx):
        fi = self.inputs_list[idx]
        fi['name'] = fi['name'].split('/')[-1]
        if  fi['name'] not in self.name2keypoints.keys():
            return self.__getitem__((idx + 1) % len(self.inputs_list))

        input_data = self.name2keypoints[fi['name']]['keypoints']
        input_data = np.array(input_data)
        input_data = torch.from_numpy(input_data).to(torch.float32)


        if self.dataset == 'phoenix2014t' or self.dataset == 'phoenix14t':
            input_data = input_data.view(-1, 116*2)

            pose_up = input_data[:,  96:]  # 保留上半身，后面可能使用
            pose_down = input_data[:,  :96]  # 0-47 48个关节点
            pose_face = input_data[:,  110:112]  # 第55个节点
            pose_add = (input_data[:, :2] + input_data[:,  2:4]) / 2  # 新增的节点 116
            input_data = torch.cat((pose_down, pose_add, pose_face), dim=-1)  # 0-47 是原本的节点， 48是新增的节点， 49是原来face 55的节点
            input_data = input_data.view(-1, 50, 2)
        # print("ggg=", input_data.shape)

        gloss, text = fi['gloss'], fi['text']
        text = text.strip('\n').strip().replace("  ", " ").lower()
        gloss = gloss.strip('\n').strip().replace("  ", " ").lower()

        label = {'gloss': gloss, 'text': text}
        return input_data, label['gloss'], label['text'], fi




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
            max_len = len(video[0])
            video_length = torch.LongTensor(
                [np.ceil(len(vid) / total_stride) * total_stride + 2 * left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
        if len(video[0].shape) > 3:
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                ), dim=0) for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1,-1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1),
                ), dim=0) for vid in video]
            padded_video = torch.stack(padded_video)

        return padded_video, video_length, gloss, text, info

    def __len__(self):
        return len(self.inputs_list)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time





def padd():
    gloss = batch['gloss_id']  # 对标 batch.src  torch.Size([8, 11])
    gloss_mask = batch['gloss_id'] != 1  # gloss的《pad》下标为2，对标 batch.src_mask
    gloss_len = batch['gloss_len']  # 对标 batch.src_lengths
    b, f, x, y = batch['skel_2d'].shape
    pose = batch['skel_2d'].reshape(b, f, -1)  # 对标 batch.trg_input B,F,166,2 -->B,F,232
    pose_lenth = batch['skel_len']
    # 根据 psoe 制作 解码器 的 mask
    pose_mask = (pose != 0).unsqueeze(1)
    pad_amount = pose.shape[1] - pose.shape[2]
    pose_mask = (F.pad(input=pose_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)
    ntokens = (pose != 2).data.sum().item()

    # 使用的是下半身
    pose_up = pose[:, :, 96:]  # 保留上半身，后面可能使用
    pose_down = pose[:, :, :96]  # 0-47 48个关节点
    pose_face = pose[:, :, 110:112]  # 第55个节点
    pose_add = (pose[:, :, :2] + pose[:, :, 2:4]) / 2  # 新增的节点 116
    pose_input = torch.cat((pose_down, pose_add, pose_face), dim=-1)  # 0-47 是原本的节点， 48是新增的节点， 49是原来face 55的节点