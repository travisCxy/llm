import math
import cv2
import numpy as np

# def three_points_to_center_wh_theta(three_points, to_int):
#    x1, y1, x2, y2, x3, y3 = three_points
#    x_c = (x1 + x3) / 2
#    y_c = (y1 + y3) / 2
#
#    if to_int:
#        x_c = int(x_c)
#        y_c = int(y_c)
#
#    w = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
#    h = math.sqrt(math.pow(x2 - x3, 2) + math.pow(y2 - y3, 2))
#
#    arc_tan =


def three_points_to_lefttop_wh_theta(three_points, to_int):
    x1, y1, x2, y2, x3, y3 = three_points
    w = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    h = math.sqrt(math.pow(x2 - x3, 2) + math.pow(y2 - y3, 2))
    theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
    ret = (x1, y1, w, h, theta)
    if not to_int:
        return ret
    else:
        return map(lambda x: int(round(x)), ret)


def three_points_to_lefttop_rightbottom_theta(three_points, to_int):
    x1, y1, w, h, theta = three_points_to_lefttop_wh_theta(three_points, False)

    x2 = x1 + w
    y2 = y1 + h

    ret = (x1, y1, x2, y2, theta)
    if not to_int:
        return ret
    else:
        return map(lambda x: int(round(x)), ret)

# def topleft_wh_theta_to_three_points(region, to_int):
#    x1, y1, w, h, theta = region


def three_points_crop_img(img, three_points):
    region = three_points_to_lefttop_rightbottom_theta(three_points, True)
    return lefttop_rightbottom_theta_crop_img(img, region)


def lefttop_rightbottom_theta_to_4points(region):
    x1, y1, x2, y2, theta = region
    points = []
    points.append((x1, y1, 1))
    points.append((x2, y1, 1))
    points.append((x2, y2, 1))
    points.append((x1, y2, 1))

    M = cv2.getRotationMatrix2D((x1, y1), - theta, 1)
    Mt = np.transpose(M)
    roated_points = np.matmul(points, Mt)
    ret = []
    for i in range(roated_points.shape[0]):
        ret.append(tuple(roated_points[i]))
    return ret


def lefttop_rightbottom_theta_to_center_wh_theta(topleft):
    x_min, y_min, x_max, y_max, theta = topleft
    w = x_max - x_min
    h = y_max - y_min

    _4points = lefttop_rightbottom_theta_to_4points(topleft)
    p1 = _4points[0]
    p3 = _4points[2]
    x_center = (p1[0] + p3[0]) / 2
    y_center = (p1[1] + p3[1]) / 2

    return (x_center, y_center, w, h, theta)


def draw_lefttop_rightbottom_theta(img, region, color, width = 1):
    _4points = lefttop_rightbottom_theta_to_4points(region)
    x1, y1 = map(lambda x: int(x), _4points[0])
    x2, y2 = map(lambda x: int(x), _4points[1])
    x3, y3 = map(lambda x: int(x), _4points[2])
    x4, y4 = map(lambda x: int(x), _4points[3])
    cv2.line(img, (x1, y1), (x2, y2), color, width)
    cv2.line(img, (x2, y2), (x3, y3), color, width)
    cv2.line(img, (x3, y3), (x4, y4), color, width)
    cv2.line(img, (x4, y4), (x1, y1), color, width)

def rotated_rect_contains_ratio(big_box, small_box):
    '''
    big_box: format lefttop_rightbottom_theta
    small_box: format lefttop_rightbottom_theta
    '''
    big_one = lefttop_rightbottom_theta_to_center_wh_theta(big_box)
    small_one = lefttop_rightbottom_theta_to_center_wh_theta(small_box)

    cv_big = ((big_one[0], big_one[1]), (big_one[2], big_one[3]), big_one[4])
    cv_small = ((small_one[0], small_one[1]), (small_one[2], small_one[3]), small_one[4])

    int_pts = cv2.rotatedRectangleIntersection(cv_big, cv_small)[1]
    if int_pts is None:
        return 0

    small_area = cv_small[1][0] * cv_small[1][1]
    order_pts = cv2.convexHull(int_pts, returnPoints=True)
    int_area = cv2.contourArea(order_pts)
    return int_area / small_area 

def rotated_rect_contains(big_box, small_box, thresh=0.8):
    
    return rotated_rect_contains_ratio(big_box, small_box) >= thresh


def lefttop_reightbottom_theta_bound_box(region, to_int):
    _4points = lefttop_rightbottom_theta_to_4points(region)
    x_min = min([x[0] for x in _4points])
    y_min = min([x[1] for x in _4points])
    x_max = max([x[0] for x in _4points])
    y_max = max([x[1] for x in _4points])
    ret = [x_min, y_min, x_max, y_max]
    if not to_int:
        return ret
    else:
        return [int(round(x)) for x in ret]


def lefttop_rightbottom_theta_crop_img(img, region):
    x1, y1, x2, y2, theta = region

    if theta == 0:
        rotated_img = img
        return rotated_img[y1:y2, x1:x2]
    else:
        x_min, y_min, x_max, y_max = lefttop_reightbottom_theta_bound_box(region, False)
        # simple fix bug x_min, y_min become < 0
        x_min = max(0, int(math.floor(x_min)))
        y_min = max(0, int(math.floor(y_min)))
        x_max = int(math.ceil(x_max))
        y_max = int(math.ceil(y_max))

        crop_img = img[y_min:y_max, x_min:x_max]
        ih, iw = crop_img.shape[:2]
        M = cv2.getRotationMatrix2D((x1 - x_min, y1 - y_min), theta, 1)
        rotated_img = cv2.warpAffine(crop_img, M, (max(iw, x2 - x_min),
                                                   max(ih,  y2 - y_min)), borderValue=(255, 255, 255))

        return rotated_img[y1 - y_min:y2 - y_min, x1 - x_min:x2 - x_min]


FIXED_IMAGE_SIZE = 224


def crop_and_rotate_for_search(img, region_with_rotation=None, debug_file=None):
    crop = img
    if region_with_rotation:
        crop = lefttop_rightbottom_theta_crop_img(img, region_with_rotation)
    crop = cv2.resize(crop, (FIXED_IMAGE_SIZE, FIXED_IMAGE_SIZE))
    if debug_file:
        cv2.imwrite(debug_file, crop)
    return crop


def makesure_point_in_wh(x1,y1, iw, ih):
#     ih, iw = img.shape[:2]
    x1 = max(x1, 1)
    y1 = max(y1, 1)
    x1 = int(min(x1, iw-1))
    y1 = int(min(y1, ih-1))
    
    return x1,y1

def makesure_3pointbox_in_wh(three_points, iw, ih):
    x1, y1, x2, y2, x3, y3 = three_points
#     ih, iw = img.shape[:2]
    x1,y1 = makesure_point_in_wh(x1,y1,iw, ih)
    x2,y2 = makesure_point_in_wh(x2,y2,iw, ih)
    x3,y3 = makesure_point_in_wh(x3,y3,iw, ih)
    
    three_points2 = (x1, y1, x2, y2, x3, y3)
    
    return three_points2
if __name__ =='__main__':
    img = np.ones([500, 500, 3], dtype=np.uint8) * 255

    img2 = lefttop_rightbottom_theta_crop_img(img, [0, 380, 19, 408, 1])
    cv2.imshow("debug", img2)
    cv2.waitKey(0)
