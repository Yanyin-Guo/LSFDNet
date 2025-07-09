import glob
from torch.utils.data.dataset import Dataset
from scripts.util import *
from torchvision.transforms import functional as F
from basicsr.utils.registry import DATASET_REGISTRY

def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0,0,0,0]*(target_len - len(some_list))

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.txt"))))
    data.sort()
    filenames.sort()
    return data, filenames

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    
    for i in range(y.shape[0]):
        y[i, 0] = w if y[i, 0] > w else y[i, 0]
        y[i, 2] = w if y[i, 2] > w else y[i, 2]
        
        y[i, 1] = h if y[i, 1] > h else y[i, 1]
        y[i, 3] = h if y[i, 3] > h else y[i, 3]
    return y

def flip_pos(x,w):
    x = w/2 - (x - w/2)
    return x

def xyxy_to_normalized_xywh(xyxy_array, image_shape):  
    image_height, image_width = image_shape  
    normalized_boxes = []  
 
    for box in xyxy_array:  
        x1, y1, x2, y2 = box  
        width = x2 - x1  
        height = y2 - y1  
        x_center = x1 + width / 2  
        y_center = y1 + height / 2  
        x_center_normalized = x_center / image_width  
        y_center_normalized = y_center / image_height  
        width_normalized = width / image_width  
        height_normalized = height / image_height  
        normalized_boxes.append([x_center_normalized, y_center_normalized, width_normalized, height_normalized])  
     
    return np.array(normalized_boxes)

def pre_batch(labels, coordinates, ln):
    ln = ln+1
  
    filled_labels = np.zeros((ln, labels.shape[1]))  
    filled_coordinates = np.zeros((ln, coordinates.shape[1]))  
    filled_batch_idx = torch.zeros(ln)

    n_labels = labels.shape[0]  
    n_coordinates = coordinates.shape[0]  
    filled_batch_idx[:n_labels] = 0
    filled_batch_idx[n_labels:n_labels+1] = 999
    filled_labels[:n_labels] = labels
    filled_labels[n_labels:n_labels+1] = 999
    filled_coordinates[:n_coordinates, :] = coordinates 
    return filled_labels, filled_coordinates, filled_batch_idx
def max_valid_lines(filepath_label): 
    max_lines = 0  
    for file in filepath_label:  
        with open(file, 'r', encoding='utf-8') as file:  
            valid_lines = sum(1 for line in file if line.strip())    
            if valid_lines > max_lines:  
                max_lines = valid_lines  
    return max_lines  
@DATASET_REGISTRY.register()
class OSLSPdet_FusionDataset(Dataset):
    def __init__( self, opt):
        super(OSLSPdet_FusionDataset, self).__init__()
        assert opt["name"] in ['train', 'val', 'test'], 'name must be "train"|"val"|"test"'
        self.opt = opt
        self.split = opt["name"]
        self.is_crop = opt["is_crop"]
        crop_size = opt["crop_size"]
        self.shape = (opt["crop_size"],opt["crop_size"])
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor=1)
        self.dethr_transform = train_dethr_transform(crop_size)
        self.centerhr_transform = train_centerhr_transform(crop_size)
        self.is_size = opt["is_size"]
        self.is_pad = opt["is_pad"]
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.1)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.1)
        self.max_label = 10
        
        self.data_dir_SW = opt["SW_path"]
        self.data_dir_LW = opt["LW_path"]
        self.data_dir_pa = opt["paired_path"] if self.split == 'train' else None
        label_dir = opt["label_path"]
        self.filepath_SW, self.filenames_SW = prepare_data_path(self.data_dir_SW)
        self.filepath_LW, self.filenames_LW = prepare_data_path(self.data_dir_LW)
        self.filepath_label, self.filenames_label = prepare_data_path(label_dir)
        self.max_label = max_valid_lines(self.filepath_label)
        self.length = min(len(self.filenames_SW), len(self.filenames_LW))

    def __getitem__(self, index):
        SW_image = Image.open(self.filepath_SW[index])
        LW_image = Image.open(self.filepath_LW[index])
        with open(self.filepath_label[index],"r") as f:    
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        h, w = SW_image.height, SW_image.width
        lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h, 0, 0)  
        la = np.asarray(lb, dtype=int)
                                        
        if self.split == 'train':
            
            self.filepath_pa, self.filenames_pa = prepare_data_path(self.data_dir_pa) 
            pa_image = Image.open(self.filepath_pa[index])
            if self.is_crop:
                crop_size = self.dethr_transform({'SW':SW_image,'label':la,'path':self.filepath_SW[index]})
                SW_image, LW_image, pa_image = F.crop(SW_image, crop_size[0], crop_size[1], crop_size[2],crop_size[3]), \
                                            F.crop(LW_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3]), \
                                            F.crop(pa_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3])
                object_label = crop_size[4]

            if self.is_size:
                pre_size, resize_size = test_size(SW_image.size)
                SW_image = SW_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality
                LW_image = LW_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality
                pa_image = pa_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality

            # Random horizontal flipping
            if random.random() > 0.5:
                SW_image = self.hflip(SW_image)
                LW_image = self.hflip(LW_image)
                pa_image = self.hflip(pa_image)
                for label in object_label:
                    label[1] = flip_pos(label[1],crop_size[3])
                    label[3] = flip_pos(label[3],crop_size[3])
                    label[1], label[3] = label[3], label[1]

            # Random vertical flipping
            if random.random() > 0.5:
                SW_image = self.vflip(SW_image)
                LW_image = self.vflip(LW_image)
                pa_image = self.vflip(pa_image)
                for label in object_label:
                    label[2] = flip_pos(label[2],crop_size[2])
                    label[4] = flip_pos(label[4],crop_size[2])
                    label[2], label[4] = label[4], label[2]

            SW_image = ToTensor()(SW_image)
            LW_image = ToTensor()(LW_image)
            pa_image = ToTensor()(pa_image)

            object_label_fusion = []
            for i in range(len(object_label)):
                object_label_fusion.append(object_label[i])
            if len(object_label_fusion) > 10 :
                object_label_fusion = object_label_fusion[0:10]
            else:
                for i in range(len(object_label_fusion),10):
                    object_label_fusion.append(np.zeros(5).astype(int) )
            object_label_fusion = torch.IntTensor(np.array(object_label_fusion).astype(int))

            cat_img = torch.cat([SW_image[0:1, :, :], LW_image[0:1, :, :]], axis=0)

            if len(object_label) >0:
                labels_all = np.array(object_label)[:, 0:1]
                coordinates_all = np.array(object_label)[:, 1:]
                cls, bboxes, batch_idx = pre_batch(labels_all, coordinates_all, self.max_label) 
            else:
                cls = np.zeros((self.max_label + 1, 1))  
                bboxes =  np.zeros((self.max_label + 1, 4))  
                batch_idx = torch.zeros(self.max_label + 1)
                batch_idx[0] = 999
                cls[0] = 999
            bboxes_norm = xyxy_to_normalized_xywh(bboxes, self.shape)
            # t = (np.array(object_label)[:, 0:1]).shape[0]
            batch={
                    "im_name": self.filenames_SW[index],
                    "im_file": self.filepath_SW[index],
                    "ori_shape": self.shape,
                    "resized_shape": self.shape,
                    "img": SW_image[0:1, :, :],
                    "cls": cls,  # n, 1
                    "bboxes": bboxes_norm,  # n, 4
                    "batch_idx": batch_idx, 
                    }

            return {'img': cat_img, 'SW': SW_image[0:1, :, :], 'LW': LW_image[0:1, :, :], 'label':object_label_fusion, 'LW_th':pa_image[0:1, :, :]}, batch

        elif self.split == 'val':
            ori_size = []
            ori_size.append(SW_image.width)
            ori_size.append(SW_image.height)
            ori_size = torch.tensor(ori_size)
            pre_size=[160,160]

            if self.is_crop:
                crop_size = self.centerhr_transform({'SW':SW_image,'label':la,'path':self.filepath_SW[index]})
                SW_image, LW_image = F.crop(SW_image, crop_size[0], crop_size[1], crop_size[2],crop_size[3]), \
                                            F.crop(LW_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3])
                object_label = crop_size[4]

            if self.is_size:
                pre_size, resize_size = test_size(SW_image.size)
                SW_image = SW_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality
                LW_image = LW_image.resize((resize_size[0], resize_size[1]),Image.LANCZOS) #resize image with high-quality

            if self.is_pad:
                target_size = max(SW_image.width, SW_image.height)    
                result_S = Image.new('RGB', (target_size, target_size), (0, 0, 0))  
                result_L = Image.new('RGB', (target_size, target_size), (0, 0, 0))  
                result_p = Image.new('RGB', (target_size, target_size), (0, 0, 0))  
                paste_x = (target_size - SW_image.width) // 2  
                paste_y = (target_size - SW_image.height) // 2  
                
                result_S.paste(SW_image, (paste_x, paste_y)) 
                SW_image = result_S
                result_L.paste(LW_image, (paste_x, paste_y)) 
                LW_image = result_L

            #
            SW_image = ToTensor()(SW_image)
            LW_image = ToTensor()(LW_image)

            cat_img = torch.cat([SW_image[0:1, :, :], LW_image[0:1, :, :]], axis=0)

            if len(object_label) >0:
                labels_all = np.array(object_label)[:, 0:1]
                coordinates_all = np.array(object_label)[:, 1:]
                cls, bboxes, batch_idx = pre_batch(labels_all, coordinates_all, self.max_label) 
            else:
                cls = np.zeros((self.max_label + 1, 1))  
                bboxes =  np.zeros((self.max_label + 1, 4))  
                batch_idx = torch.zeros(self.max_label + 1)
                batch_idx[0] = 999
                cls[0] = 999
            bboxes_norm = xyxy_to_normalized_xywh(bboxes, self.shape)
            ratio_pad = np.repeat((np.array([[1, 1], [0, 0]]))[np.newaxis, :, :], self.max_label+1, axis=0)

            batch={
                    "im_name": self.filenames_SW[index],
                    "im_file": self.filepath_SW[index],
                    "ori_shape": self.shape,
                    "resized_shape": self.shape,
                    "img": SW_image[0:1, :, :],
                    "cls": cls,  # n, 1
                    "bboxes": bboxes_norm,  # n, 4
                    "batch_idx": batch_idx, 
                    "ratio_pad": ratio_pad
                    }

            return {'img': cat_img, 'SW': SW_image[0:1, :, :], 'LW': LW_image[0:1, :, :], 'ori_size': ori_size}, batch

    def __len__(self):
        return self.length
