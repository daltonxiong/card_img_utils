import cv2
import numpy as np


def img_perspective_transformation(fg_img, fg_src_contour, rotx_deg=0, roty_deg=0, rotz_deg=0, f = 500, dist = 500):
    """计算目标图片中的点进行透视转换后的新坐标

    :param fg_img: 原图
    :param fg_src_contour: 原图需要进行转换的点列表
    :param rotx_deg: 上下翻转度数, 范围[-90, 90]
    :param roty_deg: 左右翻转度数, 范围[-90, 90]
    :param rotz_deg: 旋转度数, 范围[-90, 90]
    :param f: 距离1 图片放大缩小效果
    :param dist: 距离2 图片放大缩小效果
    :return: src_points进行透视转换后的新坐标
    """

    assert rotx_deg >=-90 and rotx_deg <=90
    assert roty_deg >=-90 and roty_deg <=90
    assert rotz_deg >=-90 and rotz_deg <=90

    h, w = fg_img.shape[:2]

    rotX = rotx_deg * np.pi / 180
    rotY = roty_deg * np.pi / 180
    rotZ = rotz_deg * np.pi / 180

    # Projection 2D -> 3D matrix
    A1 = np.matrix([[1, 0, -w / 2],
                    [0, 1, -h / 2],
                    [0, 0, 0],
                    [0, 0, 1]])

    # Rotation matrices around the X,Y,Z axis
    RX = np.matrix([[1, 0, 0, 0],
                    [0, np.cos(rotX), -np.sin(rotX), 0],
                    [0, np.sin(rotX), np.cos(rotX), 0],
                    [0, 0, 0, 1]])

    RY = np.matrix([[np.cos(rotY), 0, np.sin(rotY), 0],
                    [0, 1, 0, 0],
                    [-np.sin(rotY), 0, np.cos(rotY), 0],
                    [0, 0, 0, 1]])

    RZ = np.matrix([[np.cos(rotZ), -np.sin(rotZ), 0, 0],
                    [np.sin(rotZ), np.cos(rotZ), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Composed rotation matrix with (RX,RY,RZ)
    R = RX * RY * RZ

    # Translation matrix on the Z axis change dist will change the height
    T = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, dist],
                   [0, 0, 0, 1]])

    # Camera Intrisecs matrix 3D -> 2D
    A2 = np.matrix([[f, 0, w / 2, 0],
                    [0, f, h / 2, 0],
                    [0, 0, 1, 0]])

    # Final and overall transformation matrix
    H = A2 * (T * (R * A1))

    result_points = cv2.perspectiveTransform(fg_src_contour[None, :, :].astype(np.float32), H).squeeze().astype(np.int)
    return result_points

def merge_bg_fg_img(bg_img, fg_img, fg_src_contour, fg_dst_contour, fg_boundrect_ratio=0.9):
    """合并两张图片

    :param bg_img: 背景图片
    :param fg_img: 前景图片
    :param fg_src_contour: 前景图片中待抠出区域的点列表
    :param fg_dst_contour: 前景图片中待抠出区域的点列表透视变换后的点列表
    :param fg_boundrect_ratio: 前景图片的boundingRect占新图的面积比例
    :return: 合并后的图片,foregroun_dst在新图中的新坐标
    """

    # 将所有fg_dst_contour的负坐标转为正坐标, 然后转换为比例坐标
    boundingrect = cv2.boundingRect(fg_dst_contour)
    if boundingrect[0] < 0:
        fg_dst_contour[:, 0] += -boundingrect[0]
    if boundingrect[1] < 0:
        fg_dst_contour[:, 1] += -boundingrect[1]

    fg_dst_contour = fg_dst_contour.astype(np.float)
    fg_dst_contour[:, 0] /= boundingrect[2]
    fg_dst_contour[:, 1] /= boundingrect[3]

    # 根据fg_boundrect_ratio计算boundingrect的新的长和宽
    bg_img_h, bg_img_w = bg_img.shape[:2]
    boundingrect_w, boundingrect_h = boundingrect[-2:]

    boundingrect_new_w = bg_img_w * fg_boundrect_ratio
    boundingrect_new_h = (boundingrect_new_w/boundingrect_w) *  boundingrect_h

    if boundingrect_new_h > bg_img_h:
        boundingrect_new_h = bg_img_h * fg_boundrect_ratio
        boundingrect_new_w = (boundingrect_new_h/boundingrect_h) *  boundingrect_w

    # 计算fg_dst_contour在新图的位置
    boundingrect_start_point = ((bg_img_w-boundingrect_new_w)/4, (bg_img_h-boundingrect_new_h)/4)

    fg_dst_contour[:, 0] *= boundingrect_new_w
    fg_dst_contour[:, 1] *= boundingrect_new_h

    fg_dst_contour[:, 0] += boundingrect_start_point[0]
    fg_dst_contour[:, 1] += boundingrect_start_point[1]
    fg_dst_contour = fg_dst_contour.astype(np.int)

    # 实现opencv的copyto功能
    if fg_src_contour.shape[0] != 4 and fg_dst_contour.shape[0] != 4:
        m1 = cv2.getPerspectiveTransform(
            np.float32([fg_src_contour[4], fg_src_contour[5], fg_src_contour[14], fg_src_contour[15]]),
            np.float32([fg_dst_contour[4], fg_dst_contour[5], fg_dst_contour[14], fg_dst_contour[15]])
        )
    else:
        m1 = cv2.getPerspectiveTransform(np.float32(fg_src_contour), np.float32(fg_dst_contour))

    fg_img_new = cv2.warpPerspective(fg_img, m1, (bg_img_w, bg_img_h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

    mask = np.zeros(fg_img_new.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [fg_dst_contour], 0, 1, cv2.FILLED)
    locs = np.where(mask != 0)

    result_img = bg_img.copy()
    result_img[locs[0], locs[1]] = fg_img_new[locs[0], locs[1]]
    return (result_img,fg_dst_contour)

def merge_perspective_bg_fg_img(bg_img, fg_img, fg_src_contour, rotx_deg=0, roty_deg=0, rotz_deg=0, fg_boundrect_ratio=0.9):
    fg_dst_contour = img_perspective_transformation(fg_img, fg_src_contour, rotx_deg, roty_deg, rotz_deg)
    result = merge_bg_fg_img(bg_img, fg_img, fg_src_contour, fg_dst_contour, fg_boundrect_ratio)
    return result

def perspective_img(fg_img, fg_src_contour,rotx_deg=0, roty_deg=0, rotz_deg=0, f = 500, dist = 500, border_size=20):
    fg_dst_contour = img_perspective_transformation(fg_img, fg_src_contour, rotx_deg, roty_deg, rotz_deg, f, dist)

    boundingrect = cv2.boundingRect(fg_dst_contour)
    fg_dst_contour[:, 0] -= boundingrect[0]
    fg_dst_contour[:, 1] -= boundingrect[1]
    fg_dst_contour[:, :] += border_size

    m1 = cv2.getPerspectiveTransform(np.float32(fg_src_contour), np.float32(fg_dst_contour))
    fg_img_new = cv2.warpPerspective(fg_img, m1, (boundingrect[2]+border_size*2, boundingrect[3]+border_size*2), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

    return (fg_img_new, fg_dst_contour)
