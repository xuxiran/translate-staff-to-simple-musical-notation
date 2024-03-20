import numpy as np
from collections import Counter
import skimage.morphology as morph
from skimage.morphology.gray import opening


def rle_encode(arr):
    if len(arr) == 0:
        return [], [], []

    x = np.copy(arr)
    first_dismatch = np.array(x[1:] != x[:-1])
    distmatch_positions = np.append(np.where(first_dismatch), len(x)-1)
    rle = np.diff(np.append(-1, distmatch_positions))
    values = [x[i] for i in np.cumsum(np.append(0, rle))[:-1]]
    return rle, values

def hv_rle(img, axis=1):
    '''
    img: binary image
    axis: 0 for rows, 1 for cols
    '''
    rle, values = [], []

    if axis == 1:
        for i in range(img.shape[1]):
            col_rle, col_values = rle_encode(img[:, i])
            rle.append(col_rle)
            values.append(col_values)
    else:
        for i in range(img.shape[0]):
            row_rle, row_values = rle_encode(img[i])
            rle.append(row_rle)
            values.append(row_values)

    return rle, values

def calculate_pair_sum(arr):
    if len(arr) == 1:
        return list(arr)
    else:
        res = [arr[i] + arr[i + 1] for i in range(0, len(arr) - 1, 2)]
        if len(arr) % 2 == 1:
            res.append(arr[-2] + arr[-1])
        return res


def get_most_common(rle):
    pair_sum = [calculate_pair_sum(col) for col in rle]

    flattened = []
    for col in pair_sum:
        flattened += col

    most_common = np.argmax(np.bincount(flattened))
    return most_common

def most_common_bw_pattern(arr, most_common):
    if len(arr) == 1:
        # print("Empty")
        return []
    else:
        res = [(arr[i], arr[i + 1]) for i in range(0, len(arr) - 1, 2)
               if arr[i] + arr[i + 1] == most_common]

        if len(arr) % 2 == 1 and arr[-2] + arr[-1] == most_common:
            res.append((arr[-2], arr[-1]))
        # print(res)
        return res


def calculate_thickness_spacing(rle, most_common):
    bw_patterns = [most_common_bw_pattern(col, most_common) for col in rle]
    bw_patterns = [x for x in bw_patterns if x]  # Filter empty patterns

    flattened = []
    for col in bw_patterns:
        flattened += col

    pair, count = Counter(flattened).most_common()[0]

    line_thickness = min(pair)
    line_spacing = max(pair)

    return line_thickness, line_spacing


def histogram(img, thresh):
    hist = (np.ones(img.shape) - img).sum(dtype=np.int32, axis=1)
    _max = np.amax(hist)
    hist[hist[:] < _max * thresh] = 0
    return hist

def get_line_indices(hist):
    indices = []
    prev = 0
    for index, val in enumerate(hist):
        if val > 0 and prev <= 0:
            indices.append(index)
        prev = val
    return indices

def whitene(rle, vals, max_height):
    rlv = []
    for length, value in zip(rle, vals):
        if value == 0 and length < 1.1*max_height:
            value = 1
        rlv.append((length, value))

    n_rle, n_vals = [], []
    count = 0
    for length, value in rlv:
        if value == 1:
            count = count + length
        else:
            if count > 0:
                n_rle.append(count)
                n_vals.append(1)

            count = 0
            n_rle.append(length)
            n_vals.append(0)
    if count > 0:
        n_rle.append(count)
        n_vals.append(1)

    return n_rle, n_vals

def rle_decode(starts, lengths, values):
    starts, lengths, values = map(np.asarray, (starts, lengths, values))
    ends = starts + lengths
    n = ends[-1]

    x = np.full(n, np.nan)
    for lo, hi, val in zip(starts, ends, values):
        x[lo:hi] = val
    return x

def hv_decode(rle, values, output_shape, axis=1):
    starts = [[int(np.sum(arr[:i])) for i in range(len(arr))] for arr in rle]

    decoded = np.zeros(output_shape, dtype=np.int32)
    if axis == 1:
        for i in range(decoded.shape[1]):
            decoded[:, i] = rle_decode(starts[i], rle[i], values[i])
    else:
        for i in range(decoded.shape[0]):
            decoded[i] = rle_decode(starts[i], rle[i], values[i])

    return decoded

def remove_staff_lines(rle, vals, thickness, shape):
    n_rle, n_vals = [], []
    for i in range(len(rle)):
        rl, val = whitene(rle[i], vals[i], thickness)
        n_rle.append(rl)
        n_vals.append(val)

    return hv_decode(n_rle, n_vals, shape)

class Segmenter(object):
    def __init__(self, bin_img):
        self.bin_img = bin_img
        self.rle, self.vals = hv_rle(self.bin_img)
        self.most_common = get_most_common(self.rle)
        self.thickness, self.spacing = calculate_thickness_spacing(
            self.rle, self.most_common)
        self.thick_space = self.thickness + self.spacing

        # 移除线后的图片
        self.no_staff_img = remove_staff_lines(
            self.rle, self.vals, self.thickness, self.bin_img.shape)

        self.segment()

    def open_region(self, region):
        thickness = np.copy(self.thickness)
        # if thickness % 2 == 0:
        #     thickness += 1
        return opening(region, np.ones((thickness, thickness)))



    def segment(self):
        # 获取无线谱中“线”的位置
        self.line_indices = get_line_indices(histogram(self.bin_img, 0.8))

        # 如果只有一行，就不需要进行任何分割了，五线谱，所以会有五行
        if len(self.line_indices) < 10:
            self.regions_without_staff = [
                np.copy(self.open_region(self.no_staff_img))]
            self.regions_with_staff = [np.copy(self.bin_img)]
            return

        generated_lines_img = np.copy(self.no_staff_img)
        lines = []

        # 获取每条线的开始和结束位置（这里直接使用了从图片左端点到右端点）
        for index in self.line_indices:
            line = ((0, index), (self.bin_img.shape[1]-1, index))
            lines.append(line)

        # end_of_staff：找到五线谱那五根线的开始结束
        end_of_staff = []
        for index, line in enumerate(lines):
            # 虽然会有1个像素的误差，但是两根线之间不可能超过4*self.spacing，如果超过一定是两行乐句了
            if index > 0 and (line[0][1] - end_of_staff[-1][1] < 4*self.spacing):
                pass
            else:
                p1, p2 = line
                x0, y0 = p1
                x1, y1 = p2
                end_of_staff.append((x0, y0, x1, y1))

        # 五根线的中心位置，其实也就是大约第三根线的位置
        box_centers = []
        spacing_between_staff_blocks = []
        for i in range(len(end_of_staff)-1):
            spacing_between_staff_blocks.append(
                end_of_staff[i+1][1] - end_of_staff[i][1])
            if i % 2 == 0:
                offset = (end_of_staff[i+1][1] - end_of_staff[i][1])//2
                center = end_of_staff[i][1] + offset
                box_centers.append((center, offset))

        max_staff_dist = np.max(spacing_between_staff_blocks)
        # 这里的参数可能还得再调一下（指2和10）
        max_margin = max_staff_dist // 2
        margin = max_staff_dist // 10

        end_points = []
        # 这里的参数可能还得再调一下，实现的任务就是将每一行切割出来，以便进行下面的任务
        regions_without_staff = []
        regions_with_staff = []
        for index, (center, offset) in enumerate(box_centers):
            y0 = int(center) - max_margin - offset + margin
            y1 = int(center) + max_margin + offset - margin
            end_points.append((y0, y1))

            region = self.bin_img[y0:y1, 0:self.bin_img.shape[1]]
            regions_with_staff.append(region)
            staff_block = self.no_staff_img[y0:y1,
                                            0:self.no_staff_img.shape[1]]

            regions_without_staff.append(self.open_region(staff_block))

        self.regions_without_staff = regions_without_staff
        self.regions_with_staff = regions_with_staff
