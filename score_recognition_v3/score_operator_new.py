import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
from glob import glob

def debug(img):
    plt.imshow(img, cmap='binary')
    plt.show()

class Operater(object):
    def __init__(self, h):

        self.h = h
        self.expand = 1
        # 读取图片并放缩图片，并且二值化
        whole_note_1 = self.preprocess_img(r'.\picture\1_1.jpg')

        self.whole_note = whole_note_1

        half_note_1 = self.preprocess_img(r'.\picture\2_1.jpg')
        half_note_2 = self.preprocess_img(r'.\picture\2_2.jpg')

        mh1 = min(half_note_1.shape[1], half_note_2.shape[1])
        self.half_note = half_note_1[:,0:mh1]//2 + half_note_2[:,0:mh1]//2
        _, self.half_note = cv2.threshold(self.half_note, 127, 255, cv2.THRESH_BINARY)


        quarter_note_1 = self.preprocess_img(r'.\picture\4_1.jpg')
        quarter_note_2 = self.preprocess_img(r'.\picture\4_2.jpg')

        mq1 = min(quarter_note_1.shape[1], quarter_note_2.shape[1])

        self.quarter_note = quarter_note_1[:,0:mq1]//2 + quarter_note_2[:,0:mq1]//2
        _, self.quarter_note = cv2.threshold(self.quarter_note, 127, 255, cv2.THRESH_BINARY)

        # 高音谱号、低音谱号、小节号
        self.treble_clef = self.preprocess_img(r'.\picture\treble_clef.jpg')
        self.bass_clef = self.preprocess_img(r'.\picture\bass_clef.jpg')

        # 升号、降号、还原号
        self.sharp = self.preprocess_img(r'.\picture\sharp_1.jpg')

        self.flat = self.preprocess_img(r'.\picture\flat_1.jpg')

        self.natural = self.preprocess_img(r'.\picture\natural_1.jpg')

        # 装饰音
        self.ornament = self.preprocess_img(r'.\picture\ornament.jpg')

        # 遥指符号，删除
        # 读取路径r'.\delete下的文件数量
        delete_path = r'.\delete'
        deletefile = sorted(glob(f'{delete_path}/*'))

        self.deletenum = len(deletefile)

        # 一些新的拓展，以五线为基础
        self.expand = 1
        if self.expand == 1:

            # 小节号
            self.bar = self.preprocess_img(r'.\picture_expand\bar.jpg')
            # 休止符
            self.stop_1 = self.preprocess_img(r'.\picture_expand\stop_1.jpg')
            self.stop_2 = self.preprocess_img(r'.\picture_expand\stop_2.jpg')
            self.stop_4 = self.preprocess_img(r'.\picture_expand\stop_4.jpg')
            self.stop_8 = self.preprocess_img(r'.\picture_expand\stop_8.jpg')
            # 数字
            self.num_1_4 = self.preprocess_img(r'.\picture_expand\number_1_4.jpg')
            self.num_2_4 = self.preprocess_img(r'.\picture_expand\number_2_4.jpg')
            self.num_3_4 = self.preprocess_img(r'.\picture_expand\number_3_4.jpg')
            self.num_4_4 = self.preprocess_img(r'.\picture_expand\number_4_4.jpg')
            self.num_5_4 = self.preprocess_img(r'.\picture_expand\number_5_4.jpg')
            self.num_6_4 = self.preprocess_img(r'.\picture_expand\number_6_4.jpg')
            self.num_2_8 = self.preprocess_img(r'.\picture_expand\number_2_8.jpg')
            self.num_3_8 = self.preprocess_img(r'.\picture_expand\number_3_8.jpg')

            self.allexpand = ['bar','stop_1','stop_2','stop_4','stop_8','num_1_4','num_2_4','num_3_4','num_4_4','num_5_4','num_6_4','num_2_8','num_3_8']


        # 为了稳健起见，只支持删除三个图片，再多怕删错了东西
        for i in range(self.deletenum):
            if i == 0:
                self.delete0 = self.preprocess_img(r'.\delete\delete0.jpg')
            elif i == 1:
                self.delete1 = self.preprocess_img(r'.\delete\delete1.jpg')
            elif i == 2:
                self.delete2 = self.preprocess_img(r'.\delete\delete2.jpg')


    def preprocess_img(self, dir):
        # 以灰度读取图片
        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
        # 将图像二值化

        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 读取图片长度和高度
        h, w = img.shape


        scale = h/self.h
        h_new = self.h
        w_new = round(w/scale)

        img = cv2.resize(img, (w_new,h_new))



        return img


