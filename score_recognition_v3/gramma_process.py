import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from model import DenseNet
from model_key import DenseNet_key
from model_clef import DenseNet_clef
import matplotlib.pyplot as plt

# 读取模型
model = DenseNet().to('cpu')
model.load_state_dict(torch.load('.\model_advance.pth',map_location='cpu'))
model.eval()

model_key = DenseNet_key().to('cpu')
model_key.load_state_dict(torch.load('.\model_key.pth',map_location='cpu'))
model_key.eval()

model_clef = DenseNet_clef().to('cpu')
model_clef.load_state_dict(torch.load('.\model_clef.pth',map_location='cpu'))
model_clef.eval()

treble_clef_range = [2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1]
treble_clef_note = [2, 1, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5]

bass_clef_range = [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -3]
bass_clef_note = [4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 7]

def debug(img):
    # 以灰白图绘制结果

    plt.imshow(img)
    plt.show()


def debug2(conv):
    plt.plot(conv)
    plt.show()


def read_type_note(type_note):
    # 遍历type_note的两个维度，查询是否有11，如果有返回1否则返回0
    # type_note的第一维度是行，第二维度是列
    for i in range(len(type_note)):
        for j in range(len(type_note[i])):
            if type_note[i][j] == 11:
                return 1
    return 0

# 用户的交互界面
def User_modification(key):
    return key
    key_old = key
    print("目前每一行识别的调性是：")
    for i in range(len(key)):
        print("第{}行：{}".format(i+1, key[i]))
    print("请问是否需要修改？")
    print("1.Y   2.N")
    choice = input()
    while choice == 'Y':
        print("请输入改正以后的字符，以空格分割每行：")
        str = (input())
        key = str.split(' ')
        while len(key)!=len(key):
            print("输入的字符数目与行数不符，请重新输入：")
            str = (input())
            key = str.split(' ')
        print("修改后的调性为：")
        for i in range(len(key)):
            print("第{}行：{}".format(i + 1, key[i]))
        print("请问是否需要修改？")
        print("1.Y   2.N")
        choice = input()
    return key


def judge_rhythm(img,pixel,notes,line_5):
    # 根据line_5_line的位置生成line_21
    # 上下各补3个值，使得line_5变为11个值，
    img = np.array(img, dtype ='uint8' )
    # 将notes中为-100的值删除
    notes = notes[notes != -100]

    height = (line_5[4] - line_5[0]) / 4.0

    line_11 = [line_5[0] - 3 * height, line_5[0] - 2 * height, line_5[0] - height,
               line_5[0], line_5[1], line_5[2], line_5[3], line_5[4],
               line_5[4] + height, line_5[4] + 2 * height, line_5[4] + 3 * height]

    # 将line_11转为nparray
    line_11 = np.array(line_11)

    # 将line_11每两个值之间插入两个值的中值
    line_21 = np.insert(line_11, np.arange(1, len(line_11)), (line_11[:-1] + line_11[1:]) / 2)


    start_line = line_21[0]
    end_line = line_21[-1]


    linehigh = round(start_line)
    linelow = round(end_line)
    lineleft = round(pixel)
    lineright = round(pixel + 2 * height)

    img_note = img[linehigh:linelow, lineleft:lineright]

    # 将img_note改为(180,36)大小
    img_note = cv2.resize(img_note, (36, 180), interpolation=cv2.INTER_CUBIC)

    img = torch.from_numpy(img_note)

    img = img.unsqueeze(dim=0)
    # 将img 转为torch.float32
    img = img.to(torch.float32)

    img = img.to('cpu')

    return img,img_note


def plot_paper(type_setting,type_note,pixel_num,max_note_num,original_draw,imgs_without_staff,lines_5line,
               height,clef,line,treble_key_range, treble_key_note ,bass_key_range ,bass_key_note):
    if line!=lines_5line.shape[0]-1:
        highline = lines_5line[line][4]
        lowline = lines_5line[line+1][0]

    else:
        highline = lines_5line[line][4]
        lowline = imgs_without_staff.shape[0]

    # 根据max_note_num的数量将highline和lowline分成max_note_num份，bottomnote是其中最low的一份
    if max_note_num != 1:
        bottomnote = round(highline + (lowline - highline) / max_note_num * (max_note_num - 1))
    else:
        bottomnote = round((highline + lowline)/ 2)

    # 考虑到字体大小是height//2，因此bottomnote还要向上移动height//4
    # bottomnote = bottomnote - height//4




    print(f"正在排版第{line+1}行")




    for pixel in range(pixel_num):
        for note in range(max_note_num):
            rhythm = 0  # 如果是0不用画线
            if type_setting[pixel, note] != -100:

                note_str = []
                note_point = 0  # 表示要画几个点
                # 讨论是哪一类note，绘制不同的note_str
                # 1：全音符；2：半音符；4：四分音符；5：升号；6：降号；7：还原号；8：高音谱号；9：低音谱号
                if type_note[pixel, note] >= 8 and type_note[pixel, note] <= 9:
                    continue
                elif type_note[pixel, note] == 7:
                    note_str = "H"  # 还原号找不到字母，只能用H类似了
                elif type_note[pixel, note] == 6:
                    note_str = "b"
                elif type_note[pixel, note] == 5:
                    note_str = "#"
                elif type_note[pixel, note] == 11:
                    note_str = ''
                elif type_note[pixel, note] <= 4:
                    # 这里的四分音符需要进一步判断节奏型
                    if type_note[pixel, note] == 4:
                        # 这里需要判断节奏型
                        img_note, img_note_low = judge_rhythm(255 * imgs_without_staff, pixel, type_setting[pixel, :],
                                                              lines_5line[line])
                        pred = model(img_note)
                        _, rhythm = pred.max(1)
                        # 将rhythm转为np
                        rhythm = rhythm.cpu().numpy()
                        rhythm = rhythm[0]

                    if clef[line] == 8:
                        note_point = treble_key_range[int(type_setting[pixel, note])]
                        note_token = treble_key_note[int(type_setting[pixel, note])]
                        note_str = str(note_token)
                    else:
                        note_point = bass_key_range[int(type_setting[pixel, note])]
                        note_token = bass_key_note[int(type_setting[pixel, note])]
                        note_str = str(note_token)

                # 每个字符给一个height大小，开始绘制，暂时不画点
                # original_draw.point((pixel, lines_5line[line,4]+height*2 + note_num*height), fill="black")
                fontsize = height // 2
                # 如果是装饰音，则翻译变小
                if type_note[pixel, note] == 3:
                    fontsize = height // 4


                font = ImageFont.truetype("arial.ttf", fontsize)
                font_height = bottomnote - note * height * 3 // 4
                original_draw.text((pixel, font_height), note_str, font=font, fill="red")

                # 对全音符以及半音符还需要画线
                if type_note[pixel, note] == 2:
                    original_draw.text((pixel + height // 2, font_height), '  -', font=font, fill="red")
                if type_note[pixel, note] == 1:
                    original_draw.text((pixel + height // 2, font_height), '  -  -  -', font=font, fill="red")

                # 对小节号,在pixel的位置画一条竖线
                if type_note[pixel, note] == 11:
                    original_draw.line((pixel, bottomnote-height*max_note_num//2, pixel, bottomnote+height), fill="red",width=4)



                # 如果rhythm不是-100，则要画线
                # 画线的位置在bottomnote
                # 将rhythm转为int64
                if rhythm > 0 and rhythm < 4:
                    for i in range(rhythm):
                        # 用红色画一条宽度为1的横线
                        original_draw.line((pixel-height // 8, bottomnote + height // 2 + i*height // 16, pixel + height // 2-height // 8,
                                            bottomnote + height // 2 + i*height // 16), fill="red",width=2)

                    # 如果是低音区并且是最后一行，需要在线的基础上画点
                    y_start = bottomnote + height // 2 + i*height // 10
                else:
                    y_start = bottomnote + height // 2

                # 画点
                if note_point > 0:
                    for point in range(note_point):
                        # 点的大小是height//8
                        y = round(font_height - (point * fontsize // 6 - fontsize // 16))
                        x = pixel + fontsize // 4
                        for xx in range(round(fontsize / 9)):
                            for yy in range(round(fontsize / 9)):
                                original_draw.point((x + xx, y + yy), fill="red")

                if note_point < 0:
                    for point in range(-note_point):
                        # 点的大小是height//8
                        y = round(font_height + fontsize + (point * fontsize // 6 + fontsize // 12))

                        if note!=0:
                            y = round(font_height + fontsize + (point * fontsize // 6 + fontsize // 12))
                        else:
                            y = round(y_start + (point * fontsize // 6 + fontsize // 12))
                        x = pixel + fontsize // 4
                        for xx in range(round(fontsize / 9)):
                            for yy in range(round(fontsize / 9)):
                                original_draw.point((x + xx, y + yy), fill="red")
    return original_draw

def string_process(imgs_with_staff,imgs_without_staff,original_img,white_img, res_paper, lines_5line):

    # 将original_img转为Image格式
    original_img_copy = original_img.copy()
    original_img = Image.fromarray(original_img)
    original_draw = ImageDraw.Draw(original_img)

    white_img = Image.fromarray(white_img)
    white_draw = ImageDraw.Draw(white_img)




    process_paper = res_paper.copy()

    line_num = len(lines_5line)
    pixel_num = process_paper.shape[1]

    # clef记录每行是高音谱号还是低音谱号
    clef = np.zeros((line_num, 1))
    # key记录每行的调性：假设每行调性不变
    # key是一个字符串组成的list
    key = []

    # 确定行间距
    line_space = np.zeros((line_num-1, 1))
    for line in range(line_num-1):
        line_space[line] = lines_5line[line+1][0] - lines_5line[line][4]


    for line in range(line_num):
        line_img = process_paper[line, :, :]
        line_5line = lines_5line[line]
        height = line_5line[4] - line_5line[0]

        # 首先，根据高、低音谱号的位置，删除高音谱号附近识别的音符结果，并且判断调性
        # 高、低音谱号永远记在第0行，并且值是“8”以及“9”
        clef_flag = 0
        for i in range(pixel_num):
            if line_img[i, 0] == 8 or line_img[i, 0] == 9:
                clef_flag = i
                clef[line] = line_img[i, 0]
                break

        # 以高音或低音谱号位置为中心向右寻找确定本行调性
        key_img = line_img[int(clef[line]+height*2):int(clef[line]+height*6)]



        input_image = imgs_with_staff[line_5line[0] - height:line_5line[4] + height]
        input_image = input_image * 255
        input_image = input_image.astype("uint8")

        input_image = cv2.resize(input_image, (int(150 * input_image.shape[1] / input_image.shape[0]), 150))

        if input_image.shape[1] > 1500:
            input_image = input_image[:, 0:1500]
        else:
            input_image = cv2.copyMakeBorder(input_image, 0, 0, 0, 1500 - input_image.shape[1], cv2.BORDER_CONSTANT, value=255)

        intput_raw = input_image.copy()


        # 将key_img转为torch
        input_image = torch.tensor(input_image, dtype=torch.float32)
        input_image = input_image.unsqueeze(dim=0)

        pred = model_clef(input_image)
        _, final_clef = pred.max(1)
        if final_clef == 0:
            print("抱歉，我们不处理中音谱号")
        elif final_clef == 1:
            clef[line] = 9
        elif final_clef == 2:
            clef[line] = 8

        pred = model_key(input_image)
        _, final_key = pred.max(1)
        key_list = ["A", "Ab", "Bb" "B", "C", "D", "E", "F", "G"]
        final_keyflag = key_list[final_key]
        key.append(final_keyflag)
        print(1)


        # 这是之前的模板匹配方法，不再使用
        if 0:
            # 对key_img进行统计，确定其调性，原则是不满足某个要求则认为是原始调
            # 对升号统计4、5、6、7、8、10行，对降号统计6、7、8、9、10、11行


            # 只要sharp_stat的每一行中有一个值是5，那么就输出1，否则输出0
            # 这里有一个更恶心的，高音谱号和低音谱号它出现的位置还不一样，所以要分开统计
            sharp_place = np.array([5, 8, 4, 7, 10, 6])
            flat_place = np.array([9, 6, 10, 7, 11, 8])
            if clef[line] == 9:
                sharp_place = sharp_place + 2
                flat_place = flat_place + 2


            # 调性读取的时候，读完即删除
            # 每次读取的音不应该超过height//2
            cnt_pixel = 0
            sharp_keyflag = 'C'
            max_pixel = height

            for pixel in range(clef_flag,key_img.shape[0]):
                cnt_pixel = cnt_pixel + 1
                if cnt_pixel > 2 * max_pixel or (cnt_pixel > max_pixel and sharp_keyflag != 'C'):
                    break
                # sharp的出现顺序是5,8,4,7,10,6；flat的出现顺序是9，6，10，7，11，8
                if sharp_keyflag == 'C' and key_img[pixel,sharp_place[0]] == 5:
                    sharp_keyflag = 'G'
                    key_img[pixel, sharp_place[0]] = 0
                    cnt_pixel = 0
                elif sharp_keyflag == 'G' and key_img[pixel,sharp_place[1]] == 5:
                    sharp_keyflag = 'D'
                    key_img[pixel, sharp_place[1]] = 0
                    cnt_pixel = 0
                elif sharp_keyflag == 'D' and key_img[pixel,sharp_place[2]] == 5:
                    sharp_keyflag = 'A'
                    key_img[pixel, sharp_place[2]] = 0
                    cnt_pixel = 0
                elif sharp_keyflag == 'A' and key_img[pixel,sharp_place[3]] == 5:
                    sharp_keyflag = 'E'
                    key_img[pixel, sharp_place[3]] = 0
                    cnt_pixel = 0
                elif sharp_keyflag == 'E' and key_img[pixel,sharp_place[4]] == 5:
                    sharp_keyflag = 'B'
                    key_img[pixel, sharp_place[4]] = 0
                    cnt_pixel = 0
                elif sharp_keyflag == 'B' and key_img[pixel,sharp_place[5]] == 5:
                    sharp_keyflag = '#F'
                    key_img[pixel, sharp_place[5]] = 0
                    cnt_pixel = 0

            cnt_pixel = 0
            flat_keyflag = 'C'
            for pixel in range(key_img.shape[0]):
                cnt_pixel = cnt_pixel + 1
                if cnt_pixel > 2 * max_pixel or (cnt_pixel > max_pixel and sharp_keyflag != 'C'):
                    break
                # sharp的出现顺序是5,8,4,7,10,6；flat的出现顺序是9，6，10，7，11，8
                if flat_keyflag == 'C' and key_img[pixel,flat_place[0]] == 6:
                    flat_keyflag = 'F'
                    key_img[pixel,flat_place[0]] = 0
                    cnt_pixel = 0
                elif flat_keyflag == 'F' and key_img[pixel,flat_place[1]] == 6:
                    flat_keyflag = 'bB'
                    key_img[pixel, flat_place[1]] = 0
                    cnt_pixel = 0
                elif flat_keyflag == 'bB' and key_img[pixel,flat_place[2]] == 6:
                    flat_keyflag = 'bE'
                    key_img[pixel, flat_place[2]] = 0
                    cnt_pixel = 0
                elif flat_keyflag == 'bE' and key_img[pixel,flat_place[3]] == 6:
                    flat_keyflag = 'bA'
                    key_img[pixel, flat_place[3]] = 0
                    cnt_pixel = 0
                elif flat_keyflag == 'bA' and key_img[pixel,flat_place[4]] == 6:
                    flat_keyflag = 'bD'
                    key_img[pixel, flat_place[4]] = 0
                    cnt_pixel = 0
                elif flat_keyflag == 'bD' and key_img[pixel,flat_place[5]] == 6:
                    flat_keyflag = 'bG'
                    key_img[pixel, flat_place[5]] = 0
                    cnt_pixel = 0

            final_keyflag = 'C'
            if sharp_keyflag != 'C' and flat_keyflag != 'C':
                print('调性不明，暂时确定为C调')
            elif sharp_keyflag != 'C':
                final_keyflag = sharp_keyflag
            elif flat_keyflag != 'C':
                final_keyflag = flat_keyflag

            key.append(final_keyflag)

            # print('已经识别完本页的第',line,'行，调性为被确定为',final_keyflag)

            line_img[int(clef[line] + height * 2):int(clef[line] + height * 6)] = key_img #赋值回去
            # 删除以clef为中心的区域的内容，但是有些升降号可能会被删除，所以只删除<=4的部分
            for i in range(clef_flag + height):
                for j in range(19):
                    if line_img[i, j] <= 4:
                        line_img[i, j] = 0


        # 如果和上一行的调性不同，则在clef左边绘制调性
        if line == 0 or final_keyflag != key[line-1]:
            font = ImageFont.truetype("arial.ttf", height//2)
            note_str = '1 = ' + final_keyflag
            original_draw.text((clef_flag-height*2, lines_5line[line,0]), note_str, font=font, fill="red")


        # 聚类
        # 没必要太复杂，从左边开始遍历，后面的音符如果在height之内，那就是一起的，调整它的位置即可
        cluster = -10000
        for i in range(pixel_num):
            tmp = line_img[i, :]
            if sum(tmp) == 0:
                continue
            else:
                for j in range(19):
                    if tmp[j] == 0:
                        continue
                    else:
                        if i-cluster<height//4 and i != cluster:
                            line_img[cluster, j] = line_img[i, j]
                            line_img[i, j] = 0
                        else:
                            cluster = i

        # 打补丁
        # 同一列上发现连续的三个音，一定是假的，将中间一个音去除
        # 同一列上有四分音符，则不可能出现二分音符和全音符
        # 同一列上有全音符，就不可能出现二分音符
        # 因为特别容易把乱七八糟的玩意识别成二分音符，因此这里是最严格的，语法冲突则删除二分音符

        for i in range(pixel_num):
            tmp = line_img[i, :]
            if sum(tmp) == 0:
                continue
            else:
                for j in range(19-2):
                    if tmp[j] != 0 and tmp[j+1] != 0 and tmp[j+2] != 0:
                        line_img[i, j+1] = 0
                note_4_flag = 0
                for j in range(19):
                    if tmp[j] == 4:
                        note_4_flag = 1
                for j in range(19):
                    if tmp[j] !=4:
                        if note_4_flag == 1:
                            line_img[i, j] = 0
                note_1_flag = 0
                for j in range(19):
                    if tmp[j] == 1:
                        note_1_flag = 1
                for j in range(19):
                    if tmp[j] == 2:
                        if note_1_flag == 1:
                            line_img[i, j] = 0
        # 打补丁
        # 非常相邻的时候不可能出现两个连续的二分音符，非常相邻的界限是height//4；如果出现，则删除前一个（为啥删除前一个我也不知道，只能说经验解）
        # 这个补丁必须在聚类后打，否则会把和弦删的只剩一个


        for i in range(pixel_num-height//2):
            tmp = (line_img[i:i+height//2, :] == 2)
            # 对tmp在第1维度求和
            tmp_sum = np.sum(tmp, axis=1)
            half_num = tmp_sum >= 1
            if sum(half_num) >= 2:
                line_img[i, :] = 0

        # 打补丁
        # 非常相邻的时候，不可能出现一个二分音符或全音符在四分音符之间，非常相邻的界限是height//4；如果出现，则删除这个二分音符
        for i in range(pixel_num-2*height):
            iindex = i + height//2
            if line_img[iindex, :].any() ==2 or line_img[iindex, :].any() ==1:
                tmp_left = (line_img[iindex-height//2:iindex, :] == 4)
                tmp_right = (line_img[iindex:iindex + height, :] == 4)
                # 对tmp在第1维度求和
                tmp_sum_left = np.sum(tmp_left, axis=1)>=1
                tmp_sum_right = np.sum(tmp_right, axis=1)>= 1
                if sum(tmp_sum_right) >= 1:
                    for j in range(19):
                        if line_img[iindex, j] == 2 or line_img[iindex, j] == 1:
                            line_img[iindex, j] = 0








        # 最终，返回结果
        process_paper[line, :, :] = line_img


    # 在正式识别之前，将key打印出来给大家看，并且提供一个修改的接口
    key = User_modification(key)
    clef = User_modification(clef)



    # 根据process_paper绘制original_img
    for line in range(line_num):

        treble_key_range = treble_clef_range.copy()
        treble_key_note = treble_clef_note.copy()
        bass_key_range = bass_clef_range.copy()
        bass_key_note = bass_clef_note.copy()

        # 根据调性修改treble_key_range等值
        line_key = key[line]
        # 如果line_key的长度大于等于2，则删除第一个字符
        if len(line_key) >= 2:
            line_key = line_key[1:]

        # gap定义line_key和'C'之间ASSIC码的差值
        gap = int(ord(line_key) - ord('C'))

        for i in range(19):

            treble_key_note[i] = treble_key_note[i] - gap
            bass_key_note[i] = bass_key_note[i] - gap

            if treble_key_note[i] > 7:
                treble_key_note[i] = treble_key_note[i] - 7
                treble_key_range[i] = treble_key_range[i] + 1
            if bass_key_note[i] > 7:
                bass_key_note[i] = bass_key_note[i] - 7
                bass_key_range[i] = bass_key_range[i] + 1
            if treble_key_note[i] <= 0:
                treble_key_note[i] = treble_key_note[i] + 7
                treble_key_range[i] = treble_key_range[i] - 1
            if bass_key_note[i] <= 0:
                bass_key_note[i] = bass_key_note[i] + 7
                bass_key_range[i] = bass_key_range[i] - 1



        # 排版
        # 首先讨论这一行最多有多少个音符
        note_num = np.sum((process_paper[line, :, :] > 0), axis=1)
        max_note_num = max(note_num)


        type_setting = -100*np.ones((pixel_num, max_note_num)) # 表示每个音符对应位置
        type_note = -100*np.ones((pixel_num, max_note_num)) # 表示每个音符对应的音高

        for pixel in range(pixel_num):
            note_cnt = 0
            for note in range(19):
                if process_paper[line, pixel, 18-note] >0:
                    type_setting[pixel, note_cnt] = 18-note
                    type_note[pixel, note_cnt] = process_paper[line, pixel, 18-note]
                    note_cnt = note_cnt + 1

        # 对每个升降还原号，去寻找离它最近的音符所对应的位置，然后将其移动到该位置
        # 在上面的type_setting和type_note中，可能会出现升号“掉下去”了，这是因为我们默认从下往上
        # 对每一个升降还原号，寻找离它最近的音符，如果这个音符中有note的type_setting和升降还原号的type_setting相同，则移动到该位置
        type_setting_new = type_setting.copy()
        type_note_new = type_note.copy()



        for pixel in range(pixel_num):
            # 这个循环也是有必要的，因为可能会有多个升号
            # 这里的note_sharp代表升降还原号
            for note_sharp in range(max_note_num):
                if type_note[pixel, note_sharp] == -100:
                    break
                if type_note[pixel, note_sharp] == 5 or type_note[pixel, note_sharp] == 6 or type_note[pixel, note_sharp] == 7:
                    # 寻找离它最近的音符
                    seek_flag = 0
                    for i in range(pixel_num):
                        seek_index = i + pixel
                        # 如果已经找到，跳出
                        if seek_flag == 1:
                            break
                        # 如果很远还没找到，说明这个升降还原号是假的
                        if i >= height // 2:
                            type_setting_new[pixel, note_sharp] = -100
                            type_note_new[pixel, note_sharp] = -100
                            break
                        for note_note in range(max_note_num):

                            if type_setting[pixel, note_sharp] == type_setting[seek_index, note_note] \
                                     and type_note[seek_index, note_note] <= 4:
                                if note_sharp != note_note:
                                    type_setting_new[pixel, note_note] = type_setting_new[pixel, note_sharp]
                                    type_note_new[pixel, note_note] = type_note_new[pixel, note_sharp]
                                    type_setting_new[pixel, note_sharp] = -100
                                    type_note_new[pixel, note_sharp] = -100
                                seek_flag = 1
                                break

        type_setting = type_setting_new.copy()
        type_note = type_note_new.copy()

        # 上面删掉了一些没人要的升降号，因此这里要重新排版
        # # 从下往上寻找，如果下面的是-100而上面有值
        for pixel in range(pixel_num-1):
            tmp = type_note[pixel, :]
            tmpflag = 1
            while tmpflag == 1:
                tmpflag = 0
                for note in range(max_note_num-1):
                    if tmp[note] == -100 and tmp[note+1] != -100 and tmp[note+1] <= 4:
                        # 交换
                        tmpflag = 1
                        tmp[note] = tmp[note+1]
                        tmp[note+1] = -100
                        type_setting[pixel, note] = type_setting[pixel, note+1]
                        type_setting[pixel, note+1] = -100
                        type_note[pixel, note] = type_note[pixel, note+1]
                        type_note[pixel, note+1] = -100
                        break

        for pixel in range(pixel_num - 1):
            # 如果某一列存在升降号，那么这一列出现音符的值全部被改为-100
            tmp = type_note[pixel, :]
            if np.sum(type_note[pixel, :] == 5) > 0 or np.sum(type_note[pixel, :] == 6) > 0 or np.sum(type_note[pixel, :] == 7) > 0:
                type_note[pixel, tmp<=4] = -100
                type_setting[pixel, tmp<=4] = -100

        # 如果一行全是-100，则删除这一行，并且让max_note_num减去删掉的行数
        delete_flag = np.zeros(max_note_num)
        for i in range(max_note_num):
            if np.sum(type_setting[:,i] == -100) == pixel_num:
                delete_flag[i] = 1
        type_setting = np.delete(type_setting, np.where(delete_flag == 1), axis=1)
        type_note = np.delete(type_note, np.where(delete_flag == 1), axis=1)
        max_note_num = round(max_note_num - np.sum(delete_flag))





        # 使用函数plot_paper绘制每一个音符
        original_draw = plot_paper(type_setting, type_note, pixel_num,
                                  max_note_num,original_draw,imgs_without_staff,lines_5line,height,clef,line,
                                   treble_key_range, treble_key_note ,bass_key_range ,bass_key_note)
        white_draw = plot_paper(type_setting, type_note, pixel_num,
                                  max_note_num,white_draw,imgs_without_staff,lines_5line,height,clef,line,
                                   treble_key_range, treble_key_note ,bass_key_range ,bass_key_note)


    ori = np.asarray(original_img)
    whi = np.asarray(white_img)
    # debug(final_np)
    return ori,whi
