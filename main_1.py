import cv2
import numpy as np
import glob

CAMERA_SOURCES = [
    "rtsp://127.0.0.1:8554/stream1",
    "rtsp://127.0.0.1:8554/stream2",
    "rtsp://127.0.0.1:8554/stream3",
    "rtsp://127.0.0.1:8554/stream4"
]
NUM_CAMERAS = len(CAMERA_SOURCES)
FRAME_SIZE = (480, 270)  # Размер каждого видео
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Дисторсия — калибровка
def calibrate_camera(chessboard_dir, chessboard_size=(9,6), square_size=25):
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)*square_size
    obj_points, img_points = [], []
    images = glob.glob(chessboard_dir + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
    # Калибровка (если >= 3 изображений)
    if len(obj_points) < 3:
        print("Недостаточно шахматных досок для калибровки.")
        return None, None, None, None
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print("Калибровка завершена! (mtx, dist):", mtx, dist)
    return mtx, dist, rvecs, tvecs

def undistort(img, mtx, dist):
    if mtx is None or dist is None:
        return img
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    return dst[y:y+h, x:x+w]

def main():
    chessboard_dir = 'calib_imgs' # тут нужна папка с изображениями доски
    mtx, dist, _, _ = calibrate_camera(chessboard_dir)

    print("Открытие 4-х видео...")
    caps = [cv2.VideoCapture(src) for src in CAMERA_SOURCES]
    if not all(cap.isOpened() for cap in caps):
        print("Ошибка открытия видео."); return

    print("Запуск цикла панорамы с дисторсией...")
    frame_num = 0
    while True:
        frames = []
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print("Видео закончилось."); return
            # Применяем дисторсию
            frame = undistort(frame, mtx, dist)
            frame = cv2.resize(frame, FRAME_SIZE)
            # Добавляем подпись
            label = f"CAM {idx+1}"
            frame = cv2.putText(
                frame, label, (15, 35), FONT, 0.6, (255,0,0), 1, cv2.LINE_AA
            )
            frames.append(frame)
        panorama = np.hstack(frames)
        cv2.imshow("PANORAMA W/ LABELS & DISTORTION", panorama)
        frame_num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    for cap in caps: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()