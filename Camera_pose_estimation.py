import cv2 as cv
import numpy as np

video_file = "IMG_4346.mov"

board_pattern = (8,6)
board_cellsize = 0.025

board_criteria = (cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)

K = np.array([[800, 0, 320],
              [0, 800, 240],
              [0, 0, 1]], dtype=np.float32)

dist_coeff = np.zeros(5)

video = cv.VideoCapture(video_file)

box_lower = board_cellsize * np.array([[4, 2, 0],
                                       [5, 2, 0],
                                       [5, 4, 0],
                                       [4, 4, 0]])

box_upper = board_cellsize * np.array([[4, 2, -1],
                                       [5, 2, -1],
                                       [5, 4, -1],
                                       [4, 4, -1]])

obj_points = board_cellsize * np.array([[c, r, 0]
                                        for r in range(board_pattern[1])
                                        for c in range(board_pattern[0])],
                                       dtype=np.float32)

while True:
    valid, img = video.read()
    if not valid:
        break

    img = cv.resize(img, (480,640))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    complete, img_points = cv.findChessboardCorners(gray, board_pattern, board_criteria)

    if complete:
        img_points = cv.cornerSubPix(
            gray,
            img_points,
            (11, 11),
            (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)

        cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)

        for b, t in zip(line_lower, line_upper):
            cv.line(img,
                    tuple(np.int32(b.flatten())),
                    tuple(np.int32(t.flatten())),
                    (0, 255, 0), 2)

        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()

        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25),
                   cv.FONT_HERSHEY_DUPLEX, 0.6,
                   (0, 255, 0), 1)

    cv.imshow('Pose Estimation', img)

    if cv.waitKey(1) & 0xFF == 27:
        break

video.release()
cv.destroyAllWindows()