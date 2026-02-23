import glob
import math
import cv2
import psutil
from types import SimpleNamespace
from copy import deepcopy
from typing import Optional
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from scripts.util import *
from basicsr.utils import get_root_logger,tensor2img,imwrite,img2tensor
from ultralytics.utils.ops import xywhn2xyxy
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import get_root_logger
from scripts.util import RGB2YCrCb, YCrCb2RGB
from scripts.YOLO_data import v8_transforms, Format, Compose, Instances, LetterBox, ToTensor
from scripts.YOLO_util import verify_image_label
from ultralytics.utils.ops import resample_segments
from ultralytics.data.utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    load_dataset_cache_file,
    save_dataset_cache_file,
)
from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
DATASET_CACHE_VERSION = "1.0.3"

def prepare_data_path(dataset_path):
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.txt"))))
    filenames = [item.split('/')[-1] for item in data]
    data.sort()
    filenames.sort()
    return data, filenames

def filter_images_by_size(filepath_SW, filepath_LW, filepath_label, filenames_SW, filenames_LW, filenames_label, target_size=(1024, 768)):  
    indices_to_keep = []  
    for idx, SW_path in enumerate(filepath_SW):  
        try:  
            with Image.open(SW_path) as img:  
                if img.size == target_size:  
                    indices_to_keep.append(idx)  
        except Exception as e:  
            continue  
    filepath_SW_new      = [filepath_SW[i] for i in indices_to_keep]  
    filepath_LW_new      = [filepath_LW[i] for i in indices_to_keep]  
    filepath_label_new   = [filepath_label[i] for i in indices_to_keep]  
    filenames_SW_new     = [filenames_SW[i] for i in indices_to_keep]  
    filenames_LW_new     = [filenames_LW[i] for i in indices_to_keep]  
    filenames_label_new  = [filenames_label[i] for i in indices_to_keep]  
    return filepath_SW_new, filepath_LW_new, filepath_label_new, filenames_SW_new, filenames_LW_new, filenames_label_new  

def max_valid_lines(filepath_label): 
    max_lines = 0  
    for file in filepath_label:  
        with open(file, 'r', encoding='utf-8') as file:  
            valid_lines = sum(1 for line in file if line.strip())    
            if valid_lines > max_lines:  
                max_lines = valid_lines  
    return max_lines  

@DATASET_REGISTRY.register()
class Fusion_YOLODataset():
    def __init__(self, opt):
        super(Fusion_YOLODataset, self).__init__()
        assert opt["name"] in ['train', 'val', 'test'], 'name must be "train"|"val"|"test"'
        self.opt = opt
        self.logger = get_root_logger()
        self.split = opt["name"]
        self.target_size = (opt['target_size'][0], opt['target_size'][1])
        self.use_segments = False
        self.use_keypoints = False
        self.use_obb = False
        self.data = self.load_data(opt)
        self.hyp = self.load_hyp_opt(opt)
        self.imgsz = opt["crop_size"]
        self.augment = self.opt['hyp']['augment'] if self.split == "train" else False
        self.single_cls = self.opt['Det_labels']['nc'] == 1
        self.prefix = colorstr(f"{self.split}: ")
        self.fraction = self.opt['hyp']['fraction'] if self.split == "train" else 1.0

        ## Initialize image files and labels
        self.data_dir_SW = opt["SW_path"]
        self.data_dir_LW = opt["LW_path"]
        label_dir = opt["label_path"]
        self.filepath_SW, self.filenames_SW = prepare_data_path(self.data_dir_SW)
        self.filepath_LW, self.filenames_LW = prepare_data_path(self.data_dir_LW)
        self.filepath_label, self.filenames_label = prepare_data_path(label_dir)
        # self.filepath_SW, self.filepath_LW, self.filepath_label, self.filenames_SW, self.filenames_LW, self.filenames_label = filter_images_by_size(  
        # self.filepath_SW, self.filepath_LW, self.filepath_label, self.filenames_SW, self.filenames_LW, self.filenames_label, target_size=self.target_size) 
        self.max_label = max_valid_lines(self.filepath_label) * 4
        assert len(self.filenames_SW) == len(self.filenames_LW) == len(self.filenames_label), "SW and VI images and labels must have the same length."
        self.length = min(len(self.filenames_SW), len(self.filenames_LW))
        self.im_files = self.filepath_SW
        self.an_im_files = self.filepath_LW
        self.labels = self.get_labels()
        self.update_labels(include_class=self.opt['hyp'].get('include_class',None))  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = self.split != "train" 
        self.batch_size = opt['datasets'][self.split]['batch_size_per_gpu']
        self.stride = self.opt['hyp']['stride']
        self.pad = self.opt['hyp']['pad'] if self.split == "train" else 0
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()
        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.an_buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.an_ims, self.an_im_hw0, self.an_im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.an_npy_files = [Path(f).with_suffix(".npy") for f in self.an_im_files]
        cache = self.opt['hyp'].get('cache', None)
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if self.hyp.deterministic:
                # LOGGER.warning(
                #     "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
                #     "Consider cache='disk' as a deterministic alternative if your disk space allows."
                # )
                self.logger.info(
                    "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        self.transforms = self.build_transforms(hyp=self.hyp)

    def __getitem__(self, index):
        label = deepcopy(self.labels[index]) 
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"],  label["an_img"], label["an_ori_shape"], label["an_resized_shape"]= self.load_image(index)
        del label["an_ori_shape"]  
        del label["an_resized_shape"]
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        label = self.transforms(self.update_labels_info(label))
        label["img"] = label["img"]/255
        label["an_img"] = label["an_img"]/255
        label["im_name"] = self.filenames_SW[index]

        return label 

    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img" or k == "an_img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
    
    def __len__(self):
        return len(self.im_files)

    def close_mosaic(self, hyp):
        """
        Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def load_data(self, opt):
        data={}
        if opt['datasets'].get('train', None) is not None:
            data['train'] = opt['datasets']['train']['label_path'].split('label')[0]
        if opt['datasets'].get('val', None) is not None:
            data['val'] = opt['datasets']['val']['label_path'].split('label')[0]
        data['names'] = opt['Det_labels']['names']
        data['nc'] = opt['Det_labels']['nc']
        return data
    
    def load_hyp_opt(self, opt):
        hyp = SimpleNamespace(
            deterministic=opt["hyp"]["deterministic"],
            mosaic=opt["hyp"]["mosaic"],
            mixup=opt["hyp"]["mixup"],
            degrees=opt["hyp"]["degrees"],
            translate=opt["hyp"]["translate"],
            scale=opt["hyp"]["scale"],
            shear=opt["hyp"]["shear"],
            perspective=opt["hyp"]["perspective"],
            copy_paste_mode=opt["hyp"]["copy_paste_mode"],
            copy_paste=opt["hyp"]["copy_paste"],
            flipud=opt["hyp"]["flipud"],
            fliplr=opt["hyp"]["fliplr"],
            mask_ratio=opt["hyp"]["mask_ratio"],
            overlap_mask=opt["hyp"]["overlap_mask"],
            bgr=opt["hyp"]["bgr"],
            hsv_h=opt["hyp"]["hsv_h"],
            hsv_s=opt["hyp"]["hsv_s"],
            hsv_v=opt["hyp"]["hsv_v"],
        )
        return hyp
    
    def check_cache_disk(self, safety_margin=0.5):
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            an_im_file = random.choice(self.an_im_files)
            im = cv2.imread(im_file)
            an_im = cv2.imread(an_im_file)
            if im is None and an_im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                # LOGGER.info(f"{self.prefix}Skipping caching images to disk, directory not writeable ⚠️")
                self.logger.info(f"{self.prefix}Skipping caching images to disk, directory not writeable ⚠️")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            # LOGGER.info(
            #     f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
            #     f"with {int(safety_margin * 100)}% safety margin but only "
            #     f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk ⚠️"
            # )
            self.logger.info(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk ⚠️"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin=0.5):
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            an_im = cv2.imread(random.choice(self.an_im_files))  # sample annotation image
            if im is None and an_im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            an_ratio = self.imgsz / max(an_im.shape[0], an_im.shape[1])  # max(h, w)  # ratio
            assert ratio == an_ratio, "Image and annotation sizes do not match."
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            # LOGGER.info(
            #     f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
            #     f"with {int(safety_margin * 100)}% safety margin but only "
            #     f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            # )
            self.logger.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )
            return False
        return True
  
    def set_rectangle(self):
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.an_im_files = [self.an_im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.filenames_SW = [self.filenames_SW[i] for i in irect]
        self.filenames_LW = [self.filenames_LW[i] for i in irect]
        self.filenames_label = [self.filenames_label[i] for i in irect]
        self.filepath_SW = [self.filepath_SW[i] for i in irect]
        self.filepath_LW = [self.filepath_LW[i] for i in irect]
        self.filepath_label = [self.filepath_label[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        self.logger.info(desc)
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.an_im_files,
                    self.filepath_label,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_files, an_im_files, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_files:
                    x["labels"].append(
                        {
                            "im_file": im_files,
                            "an_im_file": an_im_files,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            # LOGGER.info("\n".join(msgs))
            self.logger.info("\n".join(msgs))
        if nf == 0:
            # LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
            self.logger.info(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.filepath_label + self.im_files + self.an_im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x
    
    def get_labels(self):
        if self.fraction < 1:
            fraction_num = round(self.length * self.fraction)
            self.im_files = self.im_files[:fraction_num]
            self.filenames_SW = self.filenames_SW[:fraction_num]
            self.an_im_files = self.an_im_files[:fraction_num]
            self.filenames_LW = self.filenames_LW[:fraction_num]
            self.filepath_label = self.filepath_label[:fraction_num]
            self.filenames_label = self.filenames_label[:fraction_num]
        # self.label_files = self.filepath_label
        cache_path = Path(self.filepath_label[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.filepath_label + self.im_files + self.an_im_files)  # identical hash
            # assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops
        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            self.logger.info(f"{self.prefix}{d}")
            if cache["msgs"]:
                # LOGGER.info("\n".join(cache["msgs"]))  # display warnings
                self.logger.info("\n".join(cache["msgs"]))  # display warnings
        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            # LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
            self.logger.info(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files
        self.an_im_files = [lb["an_im_file"] for lb in labels]  # update im_files
        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            # LOGGER.warning(
            #     f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
            #     f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
            #     "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            # )
            self.logger.info(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            # LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
            self.logger.info(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels
    
    def update_labels(self, include_class: Optional[list]):
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def cache_images(self):
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        an_b, an_gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                    an_b += self.an_npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i], self.an_ims[i], self.an_im_hw0[i], self.an_im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                    an_b += self.an_ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
                pbar.desc = f"{self.prefix}Caching an_images ({an_b / an_gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        an_f = self.an_npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)
        if not an_f.exists():
            np.save(an_f.as_posix(), cv2.imread(self.an_im_files[i]), allow_pickle=False)
    
    def load_image(self, i, rect_mode=True):
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        an_im, an_f, an_fn = self.an_ims[i], self.an_im_files[i], self.an_npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    # LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    self.logger.info(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            
        if an_im is None:  # not cached in RAM
            if an_fn.exists():  # load npy
                try:
                    an_im = np.load(an_fn)
                except Exception as e:
                    # LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {an_fn} due to: {e}")
                    self.logger.info(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {an_fn} due to: {e}")
                    Path(an_fn).unlink(missing_ok=True)
                    an_im = cv2.imread(an_f)  # BGR
            else:  # read image
                an_im = cv2.imread(an_f)  # BGR
            if an_im is None:
                raise FileNotFoundError(f"Image Not Found {an_f}")

            an_h0, an_w0 = an_im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                an_r = self.imgsz / max(an_h0, an_w0)  # ratio
                if an_r != 1:  # if sizes are not equal
                    an_w, an_h = (min(math.ceil(an_w0 * an_r), self.imgsz), min(math.ceil(an_h0 * an_r), self.imgsz))
                    an_im = cv2.resize(an_im, (an_w, an_h), interpolation=cv2.INTER_LINEAR)
            elif not (an_h0 == an_w0 == self.imgsz):  # resize by stretching image to square imgsz
                an_im = cv2.resize(an_im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.an_ims[i], self.an_im_hw0[i], self.an_im_hw[i] = an_im, (an_h0, an_w0), an_im.shape[:2]  # im, hw_original, hw_resized
                self.an_buffer.append(i)
                if 1 < len(self.an_buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.an_buffer.pop(0)
                    if self.cache != "ram":
                        self.an_ims[j], self.an_im_hw0[j], self.an_im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2], an_im, (an_h0, an_w0), an_im.shape[:2]  

        return self.ims[i], self.im_hw0[i], self.im_hw[i], self.an_ims[i], self.an_im_hw0[i], self.an_im_hw[i] 
    
    def update_labels_info(self, label):
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label
    
    def get_image_and_label(self, index):
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"],  label["an_img"], label["an_ori_shape"], label["an_resized_shape"]= self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms
