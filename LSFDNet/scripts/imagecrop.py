from torchvision.transforms import functional as F
import numbers
import random

def get_overlap(polygon_top_left, polygon_bottom_right, box_top_left, box_bottom_right):

    # 左上角为原点
    Xa1 = polygon_top_left[0]
    Xb1 = polygon_top_left[1]
    Xa2 = polygon_bottom_right[0]
    Xb2 = polygon_bottom_right[1]

    Ya1 = box_top_left[0]
    Yb1 = box_top_left[1]
    Ya2 = box_bottom_right[0]
    Yb2 = box_bottom_right[1]

    if(Xa1>Ya1+(Ya2-Ya1)):
        return []
    if(Xb1>Yb1+(Yb2-Yb1)):
        return []
    if(Xa1+(Xa2-Xa1)<Ya1):
        return []
    if(Xb1+(Xb2-Xb1)<Yb1):
        return []


    Xc1 = max(Xa1, Ya1)
    Yc1 = max(Xb1, Yb1)
    Xc2 = min(Xa2, Ya2)
    Yc2 = min(Xb2, Yb2)
    return [Xc1, Yc1, Xc2, Yc2]



class FusionRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self,img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        i, j, h, w = self.get_params(img, self.size)

        return (i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    
    
class FusionlabelCrop(object):


    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size, label,path):
        w, h = img.size
        th, tw = output_size
        x_choose = []
        ob = []
        ob_label = []
        if w == tw and h == th:
            return 0, 0, h, w, label
        for x in label:
            if((x[4] - x[2]) <= th and (x[3] - x[1]) <= tw):
                x_choose.append(x)

        if random.random() > 0.1 and len(x_choose) > 0:
            ob_num = random.randint(0, len(x_choose)-1)
            ob = x_choose[ob_num]
            i = random.randint(0 if (ob[4]-th) < 0 else ob[4]-th, h-th if (ob[2]+th) > h else ob[2])
            j = random.randint(0 if (ob[3]-tw) < 0 else ob[3]-tw, w-tw if (ob[1]+tw) > w else ob[1])     
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        for x in label:
            cross = get_overlap([x[1],x[2]], [x[3],x[4]], [j,i], [j+tw,i+th])
            if len(cross) > 0:
                if (cross[2]-cross[0] < tw/24) or (cross[3]-cross[1] < th/24):
                    continue
                for t in range(4):
                    x[t+1] = cross[t]
                ob_label.append(x-[0,j,i,j,i])
        
        return i, j, th, tw, ob_label

    def __call__(self,img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img['SW'] = F.pad(img['SW'], self.padding)

        i, j, h, w, ob_label = self.get_params(img['SW'], self.size, img['label'],img['path'])

        return (i, j, h, w, ob_label)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    
class FusionCenterCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size, label,path):
        w, h = img.size
        th, tw = output_size
        x_choose = []
        ob = []
        ob_label = []
        if w == tw and h == th:
            return 0, 0, h, w, label
        i = int((h-th)/2)
        j = int((w-tw)/2)

        for x in label:
            cross = get_overlap([x[1],x[2]], [x[3],x[4]], [j,i], [j+tw,i+th])
            if len(cross) > 0:
                if (cross[2]-cross[0] < tw/24) or (cross[3]-cross[1] < th/24):
                    continue
                for t in range(4):
                    x[t+1] = cross[t]
                ob_label.append(x-[0,j,i,j,i])

        return i, j, th, tw, ob_label

    def __call__(self,img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img['SW'] = F.pad(img['SW'], self.padding)

        i, j, h, w, ob_label = self.get_params(img['SW'], self.size, img['label'],img['path'])

        return (i, j, h, w, ob_label)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)