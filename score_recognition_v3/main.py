import numpy as np
from segmenter import Segmenter
from glob import glob
import cv2
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
from score_operator_new import Operater
import scipy
from PIL import Image, ImageDraw, ImageFont
import skimage.io as io
import torch
from model import DenseNet
from conv_operation import conv_note,conv_note_expand
from gramma_process import string_process
from pdf2image import convert_from_path
import PyPDF2
import io as io2
import os

def debug(img):
    plt.imshow(img, cmap='binary')
    plt.show()


def debug2(conv):
    plt.plot(conv)
    plt.show()
# Press the green button in the gutter to run the script.



def main(input_path, output_path,output_path2):
    imgs_path = sorted(glob(f'{input_path}/*'))

    img_cnt = 0

    for img_path in imgs_path:
        img_name = img_path.split('\\')[-1].split('.')[0]

        print(f"处理图片 {img_name}...")
        img = io.imread(img_path)

        original_img = img.copy()

        # 用cv2将img转换为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_shape = img.shape

        # 交换img_shape中的两个值
        img_shape = (img_shape[1], img_shape[0])


        # 以img为大小创建一张空白图片
        result_image = Image.new("RGB", (img_shape), "white")
        # 创建一个可绘制对象
        result_draw = ImageDraw.Draw(result_image)


            # 呈现这张图片
        plt.imshow(img, cmap='gray')

        # 二值化处理
        bin_img = 1 * (img > 127)
        # _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 呈现二值化后的图片
        plt.imshow(1 - bin_img, cmap='binary')
        # plt.show()

        # 二值化后的图片进行分割，获取五线的位置以及去除五线后的图片
        segmenter = Segmenter(bin_img)

        imgs_with_staff = bin_img
        imgs_without_staff = segmenter.no_staff_img

        imgs_without_staff_raw = imgs_without_staff.copy()

        lines = segmenter.line_indices

        if len(lines)%5!= 0:
            print("这不是五线谱！")
            continue
        # 将lines改写为5行一个
        lines_5line = np.array(lines).reshape(-1, 5)

        # 建立数组，表示这一行某个位置的音符，由于我们最多考虑19个音符，因此为数组建立为(19,bin_img.shape[1])
        # 在这个稀疏矩阵中，每一个横着的点都可能有19个音符，如果有就将其值置为对应音符标号，否则就是初始值0

        # 建立大小为(lines_5line.shape[0],bin_img.shape[1], 19)的全零三维数组res_paper
        res_paper = np.zeros((lines_5line.shape[0], bin_img.shape[1], 19))


        # 寻找高音谱号、低音谱号、小节号的位置
        for line in range(lines_5line.shape[0]):

            # 每一行的图片

            print("正在识别第%d行" % (line + 1))

            # ....................img_include_line使用寻找高音和低音谱号，标志着一行的开始..........................................#

            line_5 = lines_5line[line, :]
            height = line_5[4] - line_5[0]
            # 获取某一line的图片，从而寻找高音谱号、低音谱号、小节号
            start_line = line_5[0] - height//2
            end_line = line_5[4] + height//2

            img_line = imgs_with_staff[start_line:end_line, :]

            height = end_line - start_line

            # 处理高音和低音谱号
            cnt = 0
            all_conv = Operater(height)
            peaks_treble, res_paper, result_draw = conv_note('treble', res_paper, all_conv, line, cnt, img_line,
                                                             result_draw,
                                                             start_line, end_line)
            peaks_bass, res_paper, result_draw = conv_note('bass', res_paper, all_conv, line, cnt, img_line,
                                                           result_draw,
                                                           start_line, end_line)

            # 合并peaks_treble和peaks_bass
            both_clef = np.append(peaks_treble, peaks_bass)

            # 如果both_clef 的大小并不是1，说明识别错误
            if both_clef.shape[0] != 1:
                print('识别错误，请为本页人工选择调性')

            # 如果不存在变量all_clef，则将all_clef赋值为peaks_treble
            if 'all_clef' not in vars():
                all_clef = both_clef
            else:
                all_clef = np.append(all_clef, both_clef)


            # 寻找一些特殊符号
            height = (line_5[4] - line_5[0])-2
            # 这里的+1和-1是为了防止处理到黑线
            start_line = line_5[0]+1
            end_line = line_5[4]-1
            img_line = imgs_with_staff[start_line:end_line, :]
            all_conv = Operater(round(height))
            if all_conv.expand == 1:
                peaks, res_paper, result_draw = conv_note_expand('bar', res_paper, all_conv, line, cnt, img_line,
                                                                     result_draw,
                                                                     start_line, end_line)



            height = (line_5[4] - line_5[0]) / 4.0



            # 上下各补3个值，使得line_5变为11个值，
            line_11 = [line_5[0] - 3 * height, line_5[0] - 2 * height, line_5[0] - height,
                       line_5[0], line_5[1], line_5[2], line_5[3], line_5[4],
                       line_5[4] + height, line_5[4] + 2 * height, line_5[4] + 3 * height]

            # 将line_11转为nparray
            line_11 = np.array(line_11)

            # 将line_11每两个值之间插入两个值的中值
            line_21 = np.insert(line_11, np.arange(1, len(line_11)), (line_11[:-1] + line_11[1:]) / 2)

            # 取整
            for i in range(21):
                line_21[i] = round(line_21[i])
            line_21 = line_21.astype(int)

            # 删除可能出现的遥指
            # 这里制定的比较严格(0.4)，有可能会将一些16分音符删除掉
            for cnt in range(17):
                # 获取某一行的图片
                start_line = line_21[cnt]
                end_line = line_21[cnt + 4]
                if start_line<0 or end_line >np.shape(img)[0]:
                    continue

                height = end_line - start_line
                # 获得所有的卷积核
                all_conv = Operater(round(height))
                img_line = imgs_without_staff[start_line:end_line, :]

                # 处理遥指
                for i in range(min(all_conv.deletenum,3)):
                    delconv_str = 'delete' + str(i)

                    peaks, _, _ = conv_note(delconv_str, res_paper, all_conv, line, cnt, img_line,
                                                          result_draw,
                                                          start_line, end_line)
                    # 与其它操作不同，遇到遥指，我们直接在原始页面上删除它
                    for peak in peaks:
                        tmp_img = imgs_without_staff[start_line-height//6:end_line+height//6, peak-height//5:round(peak + height/1.5)]
                        tmp_sum_img = np.sum(tmp_img, axis=0)
                        # 构造大小和tmp_img相同的全1数组,dtype是int32
                        tmp_img2 = np.ones_like(tmp_img, dtype=np.uint8)
                        tmp_img2[:,(tmp_sum_img <= (height//16))] = 0
                        # # 如果tmp_img2是全1数组，则不进行操作(为了更好的鲁棒性，这句被删除了)
                        # if np.sum(tmp_img2) != np.shape(tmp_img2)[0] * np.shape(tmp_img2)[1]:
                        imgs_without_staff[start_line - height // 6:end_line + height // 6,
                        peak - height // 5:round(peak + height / 1.5)] = tmp_img2






            for cnt in range(19):

                # 获取某一行的图片
                start_line = line_21[cnt]
                end_line = line_21[cnt + 2]

                if start_line<0 or end_line >np.shape(img)[0]:
                    continue


                height = end_line - start_line
                # 获得所有的卷积核
                all_conv = Operater(round(height))

                img_line = imgs_without_staff[start_line:end_line, :]

                # 处理升降还原号

                peaks, res_paper, result_draw = conv_note('natural', res_paper, all_conv, line, cnt, img_line,
                                                          result_draw,
                                                          start_line, end_line)
                peaks, res_paper, result_draw = conv_note('flat', res_paper, all_conv, line, cnt, img_line,
                                                          result_draw,
                                                          start_line, end_line)
                peaks, res_paper, result_draw = conv_note('sharp', res_paper, all_conv, line, cnt, img_line,
                                                          result_draw,
                                                          start_line, end_line)

                # 处理装饰音
                peaks, res_paper, result_draw = conv_note('ornament', res_paper, all_conv, line, cnt, img_line, result_draw,
                                                          start_line, end_line)

                # 处理三种音符
                peaks, res_paper, result_draw = conv_note('half', res_paper, all_conv, line, cnt, img_line, result_draw,
                                                          start_line, end_line)

                peaks, res_paper, result_draw = conv_note('whole', res_paper, all_conv, line, cnt, img_line,
                                                          result_draw,
                                                          start_line, end_line)

                peaks, res_paper, result_draw = conv_note('quarte', res_paper, all_conv, line, cnt, img_line,
                                                          result_draw,
                                                          start_line, end_line)




        # 显示图片
        # result_image.show()
        # 将result_image转化为np格式
        result_np = np.asarray(result_image)
        # debug(result_np)
        # print(1)
        # debug(1-imgs_without_staff)

        # 我们已经收获了一张还不错的图片，接下来我们需要处理res_paper，也就是进行字符串操作，从而让数据可视化
        # 1-9分别代表我们关注的一些符号（除了3没用）
        # 1：全音符；2：半音符；4：四分音符；5：升号；6：降号；7：还原号；8：高音谱号；9：低音谱号

        # white_img和original_img大小完全相同，但是全是白色
        white_img = np.ones_like(original_img, dtype=np.uint8) * 255
        # 输入是res_paper，大小大概是(7,1653,19)，分别代表7行，这一行1653个像素，19个待选音
        # 输出是res_paper_str，大小大概是(7,？,19)，分别代表7行，这一行?个音符，每个音符19个待选音
        res_paper_str,res_paper_str2 = string_process(imgs_with_staff,imgs_without_staff_raw,original_img, white_img,res_paper, lines_5line)

        io.imsave(f'{output_path}/'+ str(img_cnt) + '.jpg', res_paper_str)
        io.imsave(f'{output_path2}/' + str(img_cnt) + '.jpg', res_paper_str2)
        img_cnt = img_cnt + 1


def delete_file(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        os.remove(file_path)

if __name__ == "__main__":

    input_path = './input'
    output_path = './output/'
    output_path2 = './output_2/'
    # 删除input_path下的所有文件
    delete_file(input_path)
    delete_file(output_path)

    #删除output_path下的所有文件
    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename)
        os.remove(file_path)


    pages = convert_from_path('..\input.pdf', 500)
    for count, page in enumerate(pages):
        page.save(f'.\input\out{count}.jpg', 'JPEG')

    main(input_path, output_path,output_path2)

    # 读取output_path下的文件
    image_filenames = sorted(glob(f'{output_path}/*'))

    # 读取image_filenames中的第一张图片
    im_1 = Image.open(image_filenames[0])
    im_1 = im_1.convert('RGB')

    # 将其余的图片读取出来，并且也转化为RGB格式，并且保存在image_list中
    image_list = []
    for i in range(1, len(image_filenames)):
        tmp = Image.open(image_filenames[i])
        tmp = tmp.convert('RGB')
        image_list.append(tmp)

    im_1.save(r'../output.pdf', save_all=True, append_images=image_list)

    # 读取output_path下的文件
    image_filenames = sorted(glob(f'{output_path2}/*'))

    # 读取image_filenames中的第一张图片
    im_1 = Image.open(image_filenames[0])
    im_1 = im_1.convert('RGB')

    # 将其余的图片读取出来，并且也转化为RGB格式，并且保存在image_list中
    image_list = []
    for i in range(1, len(image_filenames)):
        tmp = Image.open(image_filenames[i])
        tmp = tmp.convert('RGB')
        image_list.append(tmp)

    im_1.save(r'../output_2.pdf', save_all=True, append_images=image_list)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
