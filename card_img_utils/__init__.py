import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import rotate

# 身份证大小：长85.6mm*宽54mm；长度：240像素，高度：151像素。
WIDTH_HEIGHT_RATIO = 240 / 151


# 获取随机颜色
# 获取随机颜色
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
def get_random_color():
    return COLORS[np.random.choice(len(COLORS)) % len(COLORS)]

# 计算线段的角度
def calc_line_degrees(line):
    orientation = math.atan2((line[1] - line[3]), (line[0] - line[2]))
    degress = abs(math.degrees(orientation))
    return degress

# 线段于水平线之间的夹角大小
def calc_line_angle(line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    if x1 == x2:
        return 90.0

    if y1 == y2:
        return 0.0
    return abs(math.atan((y1 - y2) / (x1 - x2)) * 180.0 / math.pi)

def calc_line_length(line):
    return math.sqrt((line[0][0] - line[1][0]) ** 2 + (line[0][1] - line[1][1]) ** 2)

# 点到线的垂直距离
def point2line_distance(p0, p1, p2):
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0]*p1[1] - p1[0]*p2[1]
    denominator = math.sqrt(a*a + b*b)
    return abs((a*p0[0] + b*p0[1]+c)/denominator)

# 两条线的交点
def two_line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (int(x), int(y))

# 获取线段上所有的点
def line_all_points(line):
    (x1, y1), (x2, y2) = line
    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points

# 判断线段与区域是否有交集
def is_line_in_or_cross_contour(contour, line, shape_height, shape_width):
    contour_mask = np.zeros(shape=(shape_height, shape_width), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], 0, 255, cv2.FILLED)
    line_mask = np.zeros(shape=(shape_height, shape_width), dtype=np.uint8)
    cv2.line(line_mask, line[0], line[1], 255)
    common_mask = cv2.bitwise_and(contour_mask, line_mask)
    return np.count_nonzero(common_mask) > 0

# 剔除所有于区域有交集的线段
def filter_lines_in_or_cross_contour(contour, line_list, shape_height, shape_width):
    new_line_list = []
    contour_mask = np.zeros(shape=(shape_height, shape_width), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], 0, 1, cv2.FILLED)

    for line in line_list:
        line_mask = np.zeros(shape=(shape_height, shape_width), dtype=np.uint8)
        cv2.line(line_mask, tuple(line[0]), tuple(line[1]), 1)
        common_mask = cv2.bitwise_and(contour_mask, line_mask)
        if np.count_nonzero(common_mask) == 0:
            new_line_list.append(line)
    return new_line_list

# 判断线段与区域是否有交集
def is_line_in_or_cross_contour2(contour, line):
    for p in line_all_points(line):
        if cv2.pointPolygonTest(contour, p, False) >= 0:
            return True
    return False


# 判断线段是否被包含在联通区域中
def is_line_in_contour(contour, line):
    return cv2.pointPolygonTest(contour, line[0], False) >= 0 and cv2.pointPolygonTest(contour, line[1], False) >= 0

# 图片旋转
def rotate_image(image, point_list, angle):
    im_rot = rotate(image,angle)
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.

    new_point_list = []
    for xy in point_list:
        xy = np.array(xy)
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a), -org[0]*np.sin(a) + org[1]*np.cos(a) ])
        new_point_list.append((new+rot_center).tolist())

    return im_rot, new_point_list


# 定义旋转rotate函数
def rotate_image2(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


# 判断图片是否是黑白复印件
def is_gray_copy(origin_image):
    if isinstance(origin_image, str):
        origin_image = cv2.imread(origin_image)

    diff_value = 10
    ratio_threshold = 90

    b,g,r = cv2.split(origin_image)

    diff1 = cv2.absdiff(b,g)
    diff2 = cv2.absdiff(g,r)
    diff3 = cv2.absdiff(b,r)

    total_area_size = origin_image.shape[0] * origin_image.shape[1]

    ratio1 = (diff1 < diff_value).sum()  / total_area_size
    ratio2 = (diff2 < diff_value).sum()  / total_area_size
    ratio3 = (diff3 < diff_value).sum()  / total_area_size

    ratio1 = int(100*ratio1)
    ratio2 = int(100*ratio2)
    ratio3 = int(100*ratio3)

    ratio_list = [ratio1, ratio2, ratio3]
    return sum(1 for y in ratio_list if y >= ratio_threshold) >= 2 or max(ratio_list) >= 96

# 计算文件md5值
def file_md5(file_path):
    with open(file_path, 'rb') as f:
        md5obj = hashlib.md5()
        md5obj.update(f.read())
        hash = md5obj.hexdigest()
        return hash

# 图片矫正 身份证像素 长度为240像素 宽度为151像素
def img_straighten(image_origin, src_vertices, dst_xsize = 720, dst_ysize = 456):
    rect_dist = [(0, 0), (0, dst_ysize), (dst_xsize, 0), (dst_xsize, dst_ysize)]
    src = [src_vertices[0], src_vertices[3], src_vertices[1], src_vertices[2]]

    m1 = cv2.getPerspectiveTransform(np.float32(src), np.float32(rect_dist))
    img_new = cv2.warpPerspective(image_origin, m1, (dst_xsize, dst_ysize))
    return img_new

# 等分线段
def intermediates_line(p1, p2, nb_space):
    assert  nb_space >= 1

    x_spacing = (p2[0] - p1[0]) / nb_space
    y_spacing = (p2[1] - p1[1]) / nb_space
    point_list = [(int(p1[0] + i * x_spacing), int(p1[1] +  i * y_spacing)) for i in range(1, nb_space)]
    return [p1, *point_list, p2]

# 图片指定区域颜色高亮
def color_splash(image, mask):
    assert len(image.shape) == 3
    assert image.shape[:2] == mask.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 获取灰度图
    gray = np.tile(gray[:, :, None], [1, 1, 3])  # 灰度图转为三通道

    mask = (mask >= 1)  # 转为bool类型
    mask = np.tile(mask[:, :, None], [1, 1, 3])  #转为三通道

    return np.where(mask, image, gray).astype(np.uint8)

# 对证件四个点进行顺时针排序，起始点为左上
def box_points_sorted(point_list):
    max_contour_rect_box = sorted(point_list, key=lambda x: x[0])
    a1 = sorted([*max_contour_rect_box[:2]], key=lambda x: x[1])
    a2 = sorted([*max_contour_rect_box[2:]], key=lambda x: x[1])
    return np.array([a1[0], a2[0], a2[1], a1[1]], np.float)

# 指定区域用半透明颜色覆盖
def img_mask_color_blending(image, mask, label_list, color_list=None):
    assert len(image.shape) == 3
    assert len(mask.shape) == 2
    assert image.shape[:2] == mask.shape

    mask = mask.astype(np.uint8)

    result_img = image.copy()
    for idx, label in enumerate(label_list):
        label_mask = (mask == label)
        roi = result_img[:, :][label_mask]

        if color_list is None:
            color = get_random_color()
        else:
            color = color_list[idx]

        blended = ((0.4 * color) + (0.6 * roi)).astype('uint8')

        result_img[:, :][label_mask] = blended
    return result_img


# 水平拼接两张图片, 底部缺失部分用黑色填充
def horizontal_concatenate_images(image_list):
    max_height = max([x.shape[0] for x in image_list])

    new_image_list = []
    for image in image_list:
        new_image = cv2.copyMakeBorder(image, 0, max_height-image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        new_image_list.append(new_image)

    return np.concatenate(new_image_list, axis=1)


def get_max_contour_box_straighten_img(origin_img, output_data, straighten_width, straighten_height):
    output_data = output_data.astype(np.uint8)

    x_ratio = float(origin_img.shape[1]) / output_data.shape[1]
    y_ratio = float(origin_img.shape[0]) / output_data.shape[0]

    # 查找最大轮廓找出包含它的最小外接矩形
    contours = cv2.findContours(output_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    if contours is None:
        return None

    new_contours = []
    for c in contours:
        try:
            cv2.contourArea(c)
            new_contours.append(c)
        except:
            pass

    contours = new_contours

    if contours:
        contours_area_size = [cv2.contourArea(x)  for x in contours]
        max_contour = contours[contours_area_size.index(max(contours_area_size))]
        contour_rect = cv2.minAreaRect(max_contour)
        rect_box_points = cv2.boxPoints(contour_rect)
        rect_box_points[:, 0] *= x_ratio
        rect_box_points[:, 1] *= y_ratio
        rect_box_points = box_points_sorted(rect_box_points).astype(np.int)
        return img_straighten(origin_img, rect_box_points, straighten_width, straighten_height)
    else:
        return None


# 获取本目录及其子目录下面的所有图片文件名
def iter_all_img(src_dir):
    def __iter_dir(dir):
        for p1 in dir.iterdir():
            if p1.is_dir():
                for p2 in __iter_dir(p1):
                    yield str(p2)
            else:
                if p1.suffix.upper() in ['.JPG', '.JPEG', '.BMP', '.PNG']:
                    yield str(p1)
        return

    for p1 in Path(src_dir).iterdir():
        if p1.is_dir():
            for p2 in __iter_dir(p1):
                yield str(p2)
        else:
            if p1.suffix.upper() in ['.JPG', '.JPEG', '.BMP', '.PNG']:
                yield str(p1)
    return
