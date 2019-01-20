import cv2
import numpy as np
import card_img_utils.timefn
from card_img_utils import rotate_image

if __name__ == '__main__':
    img = np.zeros(shape=(300,200), dtype=np.uint8)
    h,w = img.shape[:2]

    card_img_utils.timefn.clear()
    for i in range(0,1000):
        card_img_utils.timefn.timefn(rotate_image)(img, [(0,0), (w,0), (h,w), (0,h)], 90)
    t = card_img_utils.timefn.avg_used_time()
    print(f'rotate_image avg_used_time={t}')
