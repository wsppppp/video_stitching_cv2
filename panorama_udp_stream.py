import cv2
import numpy as np
import glob
import subprocess
import shlex

CAMERA_SOURCES = [
    "rtsp://admin:kontour1@169.254.43.76:554/Streaming/Channels/101",
    "rtsp://127.0.0.1:8554/stream1",
]
NUM_CAMERAS = len(CAMERA_SOURCES)
FRAME_SIZE = (480, 270)  # (width, height) каждого видео
FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS = 25  # предполагаемая частота кадров

# размер итоговой панорамы (по горизонтали все камеры подряд)
OUT_WIDTH = FRAME_SIZE[0] * NUM_CAMERAS
OUT_HEIGHT = FRAME_SIZE[1]


# ---------- Калибровка ----------
def calibrate_camera(chessboard_dir, chessboard_size=(9, 6), square_size=25):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]]
        .T.reshape(-1, 2)
        * square_size
    )
    obj_points, img_points = [], []
    images = glob.glob(chessboard_dir + "/*.jpg")
    gray = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    if len(obj_points) < 3 or gray is None:
        print("Недостаточно шахматных досок для калибровки.")
        return None, None, None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    print("Калибровка завершена! (mtx, dist):", mtx, dist)
    return mtx, dist, rvecs, tvecs


def undistort(img, mtx, dist):
    if mtx is None or dist is None:
        return img
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    return dst[y : y + h, x : x + w]


# ---------- Захват ----------
def open_captures(sources):
    caps = []
    for src in sources:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"Не удалось открыть источник: {src}")
            caps.append(None)
        else:
            # по возможности уменьшить буфер (не везде работает, но попробовать стоит)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            caps.append(cap)
    return caps


# ffmpeg (udp)
def start_ffmpeg_udp(width, height, fps, host="127.0.0.1", port=5000):
    cmd = (
        f"ffmpeg -re "
        f"-f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - "
        f"-c:v libx264 -pix_fmt yuv420p "
        f"-preset ultrafast -tune zerolatency -g 10 "
        f"-f mpegts udp://{host}:{port}"
    )
    print("Запуск ffmpeg:", cmd)
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE
    )
    return proc


def main():
    chessboard_dir = "calib_imgs"  # папка с фотками шахматной доски
    mtx, dist, _, _ = calibrate_camera(chessboard_dir)

    print("Открытие потоков...")
    caps = open_captures(CAMERA_SOURCES)
    if not any(cap is not None for cap in caps):
        print("Ошибка открытия всех источников.")
        return

    # Запускаем ffmpeg-стример
    ffmpeg_proc = start_ffmpeg_udp(OUT_WIDTH, OUT_HEIGHT, FPS, host="127.0.0.1", port=5000)

    print("Запуск цикла панорамы с дисторсией и UDP-стримом...")
    print("Нажмите 'q' для выхода.")

    try:
        while True:
            frames = []
            for idx, cap in enumerate(caps):
                if cap is None:
                    frame = None
                else:
                    ret, frame = cap.read()
                    if not ret:
                        frame = None

                if frame is None:
                    # чёрный кадр с надписью NO SIGNAL
                    f = np.zeros(
                        (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8
                    )
                    cv2.putText(
                        f,
                        f"NO SIGNAL {idx+1}",
                        (15, 35),
                        FONT,
                        0.6,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    frames.append(f)
                    continue

                # коррекция дисторсии
                frame = undistort(frame, mtx, dist)
                frame = cv2.resize(frame, FRAME_SIZE)
                label = f"CAM {idx+1}"
                cv2.putText(
                    frame,
                    label,
                    (15, 35),
                    FONT,
                    0.6,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                frames.append(frame)

            # если вдруг список пуст (все капы None), выходим
            if not frames:
                print("Нет кадров ни с одного источника, завершаем.")
                break

            # панорама: все кадры по горизонтали
            panorama = np.hstack(frames)

            # Локальное отображение (можно отключить ради меньшей нагрузки)
            cv2.imshow("PANORAMA W/ LABELS & DISTORTION", panorama)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # отправка в ffmpeg по stdin
            if ffmpeg_proc.poll() is not None:
                print("ffmpeg завершился, выходим.")
                break

            try:
                ffmpeg_proc.stdin.write(panorama.tobytes())
            except BrokenPipeError:
                print("ffmpeg stdin закрыт (BrokenPipe), выходим.")
                break

    finally:
        for cap in caps:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()
        if ffmpeg_proc and ffmpeg_proc.poll() is None:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait()


if __name__ == "__main__":
    main()