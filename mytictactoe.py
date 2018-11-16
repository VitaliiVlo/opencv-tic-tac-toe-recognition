import math
import numpy as np

import cv2

from constants import NOISE_AREA, WIDTH, TESTS, PERCENT


def image_resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(img, dim, interpolation=inter)
    return resized


def get_image_without_shadow(name):
    img = cv2.imread(name, -1)
    img = image_resize(img, width=WIDTH)
    img = cv2.GaussianBlur(img, (5, 5), 2)
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    # result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm


def recognize_img_type(image, board):
    if board:
        h, w = image.shape[:2]
        top_x = int(PERCENT * w)
        top_y = int(PERCENT * h)
        new_w = int(w * (1 - PERCENT * 2))
        new_h = int(h * (1 - PERCENT * 2))
        image = image[top_y:top_y + new_h, top_x:top_x + new_w]
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= NOISE_AREA]
    canny = cv2.Canny(image, 100, 200)
    circles = cv2.HoughCircles(canny, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if len(contours) == 0:
        return "N"
    elif len(contours) == 2:
        return "O"
    elif len(contours) == 1:
        if circles is not None:
            return "O"
        return "X"


def remove_noise(binary_image):
    contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(binary_image.shape[:2], dtype="uint8") * 255
    noise = [c for c in contours if cv2.contourArea(c) < NOISE_AREA]
    cv2.drawContours(mask, noise, -1, 0, -1)
    return cv2.bitwise_and(binary_image, binary_image, mask=mask)


def get_sub_image(binary_image, rectangle):
    return np.asarray(cv2.cv.GetSubRect(cv2.cv.fromarray(binary_image), rectangle))


def process_poly_contour(poly_contour):
    poly_contour_by_x = sorted(poly_contour, key=lambda c: c[0][0])
    x_line_up = poly_contour_by_x[2:]
    x_line_down = poly_contour_by_x[:2]

    x_mean_up = (x_line_up[1][0][0] + x_line_up[0][0][0]) / 2
    x_mean_down = (x_line_down[1][0][0] + x_line_down[0][0][0]) / 2

    poly_contour_by_y = sorted(poly_contour, key=lambda c: c[0][1])
    y_line_up = poly_contour_by_y[2:]
    y_line_down = poly_contour_by_y[:2]

    y_mean_up = (y_line_up[1][0][1] + y_line_up[0][0][1]) / 2
    y_mean_down = (y_line_down[1][0][1] + y_line_down[0][0][1]) / 2

    rectangle = list()
    rectangle.append([x_mean_up, y_mean_up])
    rectangle.append([x_mean_down, y_mean_up])
    rectangle.append([x_mean_down, y_mean_down])
    rectangle.append([x_mean_up, y_mean_down])

    return rectangle


def show_img(img, win_name="test"):
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)


def change_perspective(binary_img):
    contours, hierarchy = cv2.findContours(binary_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    temp_contour_areas = list(map(cv2.contourArea, contours))
    max_contour_idx = temp_contour_areas.index(max(temp_contour_areas))
    child_contour_idx = hierarchy[0][max_contour_idx][2]

    child_contour = contours[child_contour_idx]
    eps = cv2.arcLength(child_contour, True) * 0.02
    poly_contour = cv2.approxPolyDP(child_contour, eps, True)
    # cv2.drawContours(image, [poly_contour], -1, (255, 255, 0), 2)
    new_points = process_poly_contour(poly_contour)
    old_points = list()
    for n in range(len(poly_contour)):
        old_points.append(min(poly_contour, key=lambda c: dist(c[0], new_points[n])))
    old_points = np.array([arr[0] for arr in old_points], np.float32)
    new_points = np.array(new_points, np.float32)
    M = cv2.getPerspectiveTransform(old_points, new_points)
    height, width = binary_img.shape[:2]
    binary_img = cv2.warpPerspective(binary_img, M, (width, height))
    binary_img = remove_noise(binary_img)
    return binary_img


def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def recognize(binary_img, binary_img_without_board, rect, max_x, min_x, max_y, min_y):
    x_r, y_r, w_r, h_r = rect

    x = [x_r, min_x, max_x]
    y = [y_r, min_y, max_y]

    h = [min_y - y_r, max_y - min_y, h_r + y_r - max_y]
    w = [min_x - x_r, max_x - min_x, w_r + x_r - max_x]

    matrix = list()
    for row in xrange(3):
        temp_row = list()
        for col in xrange(3):
            new_img = get_sub_image(binary_img, (x[col], y[row], w[col], h[row]))
            new_img_without_board = get_sub_image(binary_img_without_board, (x[col], y[row], w[col], h[row]))
            # show_img(new_img, "image %s %s" % (row, col))
            # print row, col
            type_new = recognize_img_type(new_img, True)
            type_old = recognize_img_type(new_img_without_board, False)
            if type_old == "N" and type_new != "N":
                temp_row.append(type_new)
            else:
                temp_row.append(type_old)
        matrix.append(temp_row)
    return matrix


def main(name):
    image = get_image_without_shadow(name)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_img = remove_noise(thresh)
    # structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_PARAM)
    # binary_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, structuring_element)
    # binary_img = cv2.dilate(binary_img, structuring_element, iterations=1)

    binary_img = change_perspective(binary_img)

    # find the biggest contour
    contours, hierarchy = cv2.findContours(binary_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    temp_contour_areas = list(map(cv2.contourArea, contours))
    max_contour_idx = temp_contour_areas.index(max(temp_contour_areas))
    max_contour = contours[max_contour_idx]

    child_contour_idx = hierarchy[0][max_contour_idx][2]
    child_contour = contours[child_contour_idx]

    # remove board
    child_child_contour_idx = hierarchy[0][child_contour_idx][2]
    mask = np.ones(binary_img.shape[:2], dtype="uint8") * 255
    cv2.drawContours(mask, [max_contour], -1, 0, -1)
    if child_child_contour_idx != -1:
        child_child_contour = contours[child_child_contour_idx]
        sub_rect = cv2.boundingRect(child_child_contour)
        sub_img = get_sub_image(binary_img, sub_rect)
        mask[sub_rect[1]:sub_rect[1] + sub_img.shape[0], sub_rect[0]:sub_rect[0] + sub_img.shape[1]] = sub_img
    binary_img_without_board = cv2.bitwise_and(binary_img, binary_img, mask=mask)
    # remove board end

    rect = cv2.boundingRect(max_contour)  # x, y, w, h

    # x, y, w, h = rect
    # cv2.rectangle(binary_img, (x, y), (x + w, y + h), 255, 2)
    # show_img(binary_img)

    eps = cv2.arcLength(child_contour, True) * 0.02
    poly = cv2.approxPolyDP(child_contour, eps, True)

    # cv2.drawContours(binary_img, [child_contour], -1, 255, 2)
    # show_img(binary_img)

    poly_contour_by_x = sorted(poly, key=lambda c: c[0][0])
    x_line_up = poly_contour_by_x[2:]
    x_line_down = poly_contour_by_x[:2]
    poly_contour_by_y = sorted(poly, key=lambda c: c[0][1])
    y_line_up = poly_contour_by_y[2:]
    y_line_down = poly_contour_by_y[:2]
    max_x = x_line_up[-1][0][0]
    min_x = x_line_down[0][0][0]
    max_y = y_line_up[-1][0][1]
    min_y = y_line_down[0][0][1]

    return recognize(binary_img, binary_img_without_board, rect, max_x, min_x, max_y, min_y)


if __name__ == '__main__':
    for x in range(0, 16):
        res = main("img/test%s.jpg" % x)
        print res, res == TESTS[x], x
        # res = main("img/test7.jpg")
        # print res
