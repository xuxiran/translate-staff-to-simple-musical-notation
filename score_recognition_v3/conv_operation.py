import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
import scipy
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

treble_clef_range = [2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1]
treble_clef_note = [2, 1, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5]


bass_clef_range = [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -3]
bass_clef_note = [4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 7]

def debug(img):
    plt.imshow(img, cmap='binary')
    plt.show()


def debug2(conv):
    plt.plot(conv)
    plt.show()


def conv_note(Convolutional_kernel, res_paper, all_conv, line, cnt, img_line, result_draw, start_line, end_line):

    if Convolutional_kernel == 'treble':
        conv_q = all_conv.treble_clef
    elif Convolutional_kernel == 'bass':
        conv_q = all_conv.bass_clef
    elif Convolutional_kernel == 'sharp':
        conv_q = all_conv.sharp
    elif Convolutional_kernel == 'flat':
        conv_q = all_conv.flat
    elif Convolutional_kernel == 'natural':
        conv_q = all_conv.natural
    elif Convolutional_kernel == 'quarte':
        conv_q = all_conv.quarter_note
    elif Convolutional_kernel == 'half':
        conv_q = all_conv.half_note
    elif Convolutional_kernel == 'whole':
        conv_q = all_conv.whole_note
    elif Convolutional_kernel == 'ornament':
        conv_q = all_conv.ornament
    elif Convolutional_kernel == 'delete0':
        conv_q = all_conv.delete0
    elif Convolutional_kernel == 'delete1':
        conv_q = all_conv.delete1
    elif Convolutional_kernel == 'delete2':
        conv_q = all_conv.delete2
    elif Convolutional_kernel == 'bar':
        conv_q = all_conv.bar

    # 模板匹配
    # 将img_line和conv_q转化为灰度图片
    img_line = img_line*255

    img_line = np.float32(img_line)
    conv_q = np.float32(conv_q)
    conv_res = cv2.matchTemplate(img_line, conv_q, cv2.TM_CCOEFF_NORMED)[0]


    # 设定参数
    if Convolutional_kernel == 'treble':
        conv_res[conv_res < 0.4] = 0.4
    elif Convolutional_kernel == 'bass':
        conv_res[conv_res < 0.4] = 0.4
    # elif Convolutional_kernel == 'section_1':
    #     conv_res[conv_res < 100] = 100
    # elif Convolutional_kernel == 'section_2':
    #     conv_res[conv_res < 100] = 100
    elif Convolutional_kernel == 'sharp':
        conv_res[conv_res < 0.5] = 0.5
    elif Convolutional_kernel == 'flat':
        conv_res[conv_res < 0.6] = 0.6
    elif Convolutional_kernel == 'natural':
        conv_res[conv_res < 0.7] = 0.7

    elif Convolutional_kernel == 'quarte':
        conv_res[conv_res < 0.5] = 0.5

    elif Convolutional_kernel == 'half':
        conv_res[conv_res < 0.3] = 0.3
    elif Convolutional_kernel == 'whole':
        conv_res[conv_res < 0.3] = 0.3

    elif Convolutional_kernel == 'ornament':
        conv_res[conv_res < 0.7] = 0.7

    elif Convolutional_kernel == 'delete0':
        conv_res[conv_res < 0.4] = 0.4
    elif Convolutional_kernel == 'delete1':
        conv_res[conv_res < 0.4] = 0.4
    elif Convolutional_kernel == 'delete2':
        conv_res[conv_res < 0.4] = 0.4

    elif Convolutional_kernel == 'bar':
        conv_res[conv_res < 0.4] = 0.4

    # 使用np的函数对conv_res进行高斯平滑
    conv_res = scipy.ndimage.gaussian_filter1d(conv_res, 1)

    # 提取conv_res中每个极值点
    peaks, _ = scipy.signal.find_peaks(conv_res, height=0)

    # 打补丁：对half和whole而言，长度和宽度一定很大，从而可以对peaks进行筛选
    if Convolutional_kernel == 'half' or Convolutional_kernel == 'whole':
        is_peak = np.ones(len(peaks))
        for peak in peaks:
            # img_peak和conv_q是一样的大小
            img_peak = 255 - img_line[:, peak:peak+conv_q.shape[1]]
            conv_q_inverse = 255 - conv_q

            # 将img_peak按照列求和，如果和为0，说明这一列全是0，即这一列全是黑色
            width_img = np.sum(img_peak, axis=0)>0
            height_img = np.sum(img_peak, axis=1)>0
            width_conv = np.sum(conv_q_inverse, axis=0)>0
            height_conv = np.sum(conv_q_inverse, axis=1)>0
            # 如果width_img与width_conv的差异达到width_conv的1/3，则说明这个peak是假的
            # 同理，如果height_img与height_conv的差异达到height_conv的1/3，则说明这个peak是假的
            if np.sum(width_img == width_conv) < 4*len(width_conv)/5:
                is_peak[peaks == peak] = 0
                # 再打一个补丁，如果是上三线或者下三线，会出现“线”没有去除，此时判处is_peak[peaks == peak] = 0要更加严格
                # 也就是说，如果比conv的大，并不删除
                if sum(width_img)>sum(width_conv):
                    is_peak[peaks == peak] = 1
            if np.sum(height_img == height_conv) < 4*len(height_conv)/5:
                is_peak[peaks == peak] = 0
        # 删除假的peak
        peaks = peaks[is_peak == 1]

    # 打补丁，四分音符的补丁要慎重打，必须在很确信的时候才能删除
    if Convolutional_kernel == 'quarte':
        is_peak = np.ones(len(peaks))
        for peak in peaks:
            # img_peak和conv_q是一样的大小
            img_peak = (255 - img_line[:, peak:peak + conv_q.shape[1]])/255
            conv_q_inverse = (255 - conv_q)/255
            # 将img_peak按照列求和，如果和为0，说明这一列全是0，即这一列全是黑色
            width_img = np.sum(img_peak, axis=0) >= conv_q_inverse.shape[1]//4
            height_img = np.sum(img_peak, axis=1) >= conv_q_inverse.shape[0]//4
            width_conv = np.sum(conv_q_inverse, axis=0) >= conv_q_inverse.shape[1]//4
            height_conv = np.sum(conv_q_inverse, axis=1) >= conv_q_inverse.shape[0]//4
            # 如果width_img与width_conv的差异达到width_conv的1/3，则说明这个peak是假的
            # 同理，如果height_img与height_conv的差异达到height_conv的1/3，则说明这个peak是假的
            if np.sum(width_img == width_conv) < 4 * len(width_conv) / 5:
                is_peak[peaks == peak] = 0
            if np.sum(height_img == height_conv) < 5 * len(height_conv) / 6:
                is_peak[peaks == peak] = 0
            # 删除假的peak
        peaks = peaks[is_peak == 1]

    # 在res_paper上呈现对应的结果
    if Convolutional_kernel == 'treble':
        res_paper[line, peaks, cnt] = 8
    elif Convolutional_kernel == 'bass':
        res_paper[line, peaks, cnt] = 9

    # elif Convolutional_kernel == 'section_1':
    #     res_paper[line, peaks, cnt] = 111
    # elif Convolutional_kernel == 'section_2':
    #     res_paper[line, peaks, cnt] = 111

    elif Convolutional_kernel == 'sharp':
        res_paper[line, peaks, cnt] = 5
    elif Convolutional_kernel == 'flat':
        res_paper[line, peaks, cnt] = 6
    elif Convolutional_kernel == 'natural':
        res_paper[line, peaks, cnt] = 7

    elif Convolutional_kernel == 'quarte':
        res_paper[line, peaks, cnt] = 4
    elif Convolutional_kernel == 'half':
        res_paper[line, peaks, cnt] = 2
    elif Convolutional_kernel == 'whole':
        res_paper[line, peaks, cnt] = 1

    elif Convolutional_kernel == 'ornament':
        res_paper[line, peaks, cnt] = 3

    height = (end_line-start_line)//2

    # 对音符绘制对应的结果
    if Convolutional_kernel == 'quarte' or Convolutional_kernel == 'half' or Convolutional_kernel == 'whole':
        for note_w in peaks:
            note_h = round((start_line + end_line) / 2)
            note_range = treble_clef_range[cnt]
            note_token = treble_clef_note[cnt]

            note_str = str(note_token)
            font = ImageFont.truetype("arial.ttf", height*8)
            result_draw.text((note_w, note_h), note_str, font=font, fill="black")
    else:
        for note_w in peaks:
            note_h = round((start_line + end_line) / 2)

            note_str = Convolutional_kernel
            font = ImageFont.truetype("arial.ttf", height)
            result_draw.text((note_w, note_h), note_str, font=font, fill="black")
    return peaks, res_paper, result_draw


def conv_note_expand(Convolutional_kernel, res_paper, all_conv, line, cnt, img_line, result_draw, start_line, end_line):

    if Convolutional_kernel == 'bar':
        conv_q = all_conv.bar
    elif Convolutional_kernel == 'stop_1':
        conv_q = all_conv.stop_1
    elif Convolutional_kernel == 'stop_2':
        conv_q = all_conv.stop_2
    elif Convolutional_kernel == 'stop_4':
        conv_q = all_conv.stop_4
    elif Convolutional_kernel == 'stop_8':
        conv_q = all_conv.stop_8
    elif Convolutional_kernel == 'num_1_4':
        conv_q = all_conv.num_1_4
    elif Convolutional_kernel == 'num_2_4':
        conv_q = all_conv.num_2_4
    elif Convolutional_kernel == 'num_3_4':
        conv_q = all_conv.num_3_4
    elif Convolutional_kernel == 'num_4_4':
        conv_q = all_conv.num_4_4
    elif Convolutional_kernel == 'num_5_4':
        conv_q = all_conv.num_5_4
    elif Convolutional_kernel == 'num_6_4':
        conv_q = all_conv.num_6_4
    elif Convolutional_kernel == 'num_2_8':
        conv_q = all_conv.num_2_8
    elif Convolutional_kernel == 'num_3_8':
        conv_q = all_conv.num_3_8


        # 模板匹配
    # 将img_line和conv_q转化为灰度图片
    img_line = img_line*255

    img_line = np.float32(img_line)
    conv_q = np.float32(conv_q)
    conv_res = cv2.matchTemplate(img_line, conv_q, cv2.TM_CCOEFF_NORMED)[0]


    # 设定参数
    if Convolutional_kernel == 'bar':
        conv_res[conv_res < 0.2] = 0.2
    elif Convolutional_kernel == 'stop_1':
        conv_res[conv_res < 0.3] = 0.3
    elif Convolutional_kernel == 'stop_2':
        conv_res[conv_res < 0.3] = 0.3
    elif Convolutional_kernel == 'stop_4':
        conv_res[conv_res < 0.3] = 0.3
    elif Convolutional_kernel == 'stop_8':
        conv_res[conv_res < 0.3] = 0.3
    elif Convolutional_kernel == 'num_1_4':
        conv_res[conv_res < 0.7] = 0.7
    elif Convolutional_kernel == 'num_2_4':
        conv_res[conv_res < 0.7] = 0.7
    elif Convolutional_kernel == 'num_3_4':
        conv_res[conv_res < 0.7] = 0.7
    elif Convolutional_kernel == 'num_4_4':
        conv_res[conv_res < 0.7] = 0.7
    elif Convolutional_kernel == 'num_5_4':
        conv_res[conv_res < 0.7] = 0.7
    elif Convolutional_kernel == 'num_6_4':
        conv_res[conv_res < 0.7] = 0.7
    elif Convolutional_kernel == 'num_2_8':
        conv_res[conv_res < 0.7] = 0.7
    elif Convolutional_kernel == 'num_3_8':
        conv_res[conv_res < 0.7] = 0.7

    # 使用np的函数对conv_res进行高斯平滑
    conv_res = scipy.ndimage.gaussian_filter1d(conv_res, 1)

    # 提取conv_res中每个极值点
    peaks, _ = scipy.signal.find_peaks(conv_res, height=0)

    # 打补丁：对bar而言，高度一定很高，但宽度一定很小，从而可以对peaks进行筛选
    if Convolutional_kernel == 'bar':
        is_peak = np.ones(len(peaks))
        for peak in peaks:
            # img_peak和conv_q是一样的大小
            img_peak = 1-img_line[:, peak:peak+conv_q.shape[1]]/255.0
            conv_q_inverse = 1-conv_q/255.0

            # 将img_peak按照列求和，如果和为0，说明这一列全是0，即这一列全是黑色
            width_img = np.sum(img_peak, axis=0)
            height_img = np.sum(img_peak, axis=1)
            width_conv = np.sum(conv_q_inverse, axis=0)
            height_conv = np.sum(conv_q_inverse, axis=1)


            # 首先观察width_img，我们知道竖线的特点是只有中间很少的值远远大于平均值，而其它值都小于平均值很多
            # 通过比较各个值和平均值之间的关系，可以判断出是否为竖线
            # 如果只有1/4的值超过均值的两倍，那么就是竖线
            if np.sum(width_img > np.mean(width_img)) > len(width_img)/4 or np.sum(width_img>len(height_img)/8)>len(width_img)/4:
                is_peak[peaks == peak] = 0
                continue

            # 另外，如果这一条竖线并不是出现在中部，也删除
            if np.argmax(width_img) < len(width_img)/3 or np.argmax(width_img) > len(width_img)*2/3:
                is_peak[peaks == peak] = 0
                continue

            # 接着观察height_img，我们知道竖线的特点是沿着height维度比较均匀
            # 如果height_img的方差大于height_conv的方差的3倍，那么就不是竖线
            if np.var(height_img) > 3 * np.var(height_conv):
                is_peak[peaks == peak] = 0
                continue

            # 还需要排除掉大白片的情况
            # 大白片的特征是width_img根本不存在黑点
            # 也就是不存在某个点，比均值大height_conv长度的0.8倍
            if np.sum(width_img > len(height_conv) * 0.8) == 0:
                is_peak[peaks == peak] = 0
                continue


        # 删除假的peak
        peaks = peaks[is_peak == 1]

    # 在allconv.allexpand中寻找Convolutional_kernel并返回下标

    res_paper[line, peaks, cnt] = 11 #扩展，之前已经有了10个数字


    for note_w in peaks:
        note_h = round((start_line + end_line) / 2)

        note_str = Convolutional_kernel
        font = ImageFont.truetype("arial.ttf", 25)
        result_draw.text((note_w, note_h), note_str, font=font, fill="black")

    return peaks, res_paper, result_draw