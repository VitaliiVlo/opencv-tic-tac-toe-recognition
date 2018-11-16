import os

import cv2

from constants import FOLDER_NAME
from mytictactoe import main, show_img, image_resize


def winner(arr):
    row_win = [r[0] for r in arr if len(set(r)) == 1 and r[0] != "N"]
    col_win = [c[0] for c in zip(*arr) if len(set(c)) == 1 and c[0] != "N"]
    diagonals = [(row[idx], row[abs(idx - 2)]) for idx, row in enumerate(arr)]
    temp = tuple(map(set, zip(*diagonals)))
    d1, d2 = tuple(map(list, temp))
    d1_win = [d1[0]] if len(set(d1)) == 1 and d1[0] != "N" else []
    d2_win = [d2[0]] if len(set(d2)) == 1 and d2[0] != "N" else []

    checks = [row_win, col_win, d1_win, d2_win]
    for check in checks:
        if len(check):
            return "The winner is %s" % check[0]
    for row in arr:
        for el in row:
            if el == "N":
                return "The game continues"
    return "Draw"


if __name__ == '__main__':
    for filename in os.listdir(FOLDER_NAME):
        res = main(os.path.join(FOLDER_NAME, filename))
        if isinstance(res, int):
            continue
        img = cv2.imread(os.path.join(FOLDER_NAME, filename))
        img = image_resize(img, width=720)
        print res
        print filename, "\n"
        show_img(img)
