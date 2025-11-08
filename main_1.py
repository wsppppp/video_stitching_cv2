import cv2
import numpy as np
import time

CAMERA_SOURCES = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]
NUM_CAMERAS = len(CAMERA_SOURCES)
FRAME_SIZE = (960, 540)
MAX_WIDTH = 2500
MAX_HEIGHT = 1200
CALIB_FRAMES = 10

# находим матрицы преобразования между видео
def compute_homographies(frames):
    # Центр — второй кадр
    center_idx = 1
    homographies = [None]*NUM_CAMERAS
    homographies[center_idx] = np.eye(3, dtype=np.float32)
    match_counts = []
    detector = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # Вправо от центра
    for i in range(center_idx, NUM_CAMERAS-1):
        kpsA, descA = detector.detectAndCompute(frames[i], None)
        kpsB, descB = detector.detectAndCompute(frames[i+1], None)
        matches = matcher.match(descA, descB)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:max(int(0.20 * len(matches)), 40)]
        match_counts.append(len(good))
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)
        homographies[i+1] = homographies[i] @ H
    # Влево от центра
    for i in range(center_idx, 0, -1):
        kpsA, descA = detector.detectAndCompute(frames[i], None)
        kpsB, descB = detector.detectAndCompute(frames[i-1], None)
        matches = matcher.match(descA, descB)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:max(int(0.20 * len(matches)), 40)]
        match_counts.append(len(good))
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)
        homographies[i-1] = homographies[i] @ H
    return homographies, match_counts

# считаем размер панорамы
def calc_pano_size(frames, homographies):
    h, w = FRAME_SIZE
    c_list = []
    for i in range(NUM_CAMERAS):
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        c_list.append(cv2.perspectiveTransform(corners, homographies[i]))
    all_c = np.concatenate(c_list, axis=0)
    x_min, y_min = all_c.min(axis=0).ravel()
    x_max, y_max = all_c.max(axis=0).ravel()
    pano_w, pano_h = int(x_max-x_min), int(y_max-y_min)
    scale = min(MAX_WIDTH/pano_w, MAX_HEIGHT/pano_h, 1.0)
    translation = np.array([[scale,0,-x_min*scale],[0,scale,-y_min*scale],[0,0,1]], dtype=np.float32)
    return [translation @ h for h in homographies], (int(pano_w*scale), int(pano_h*scale))

# эта штука должна обрезать только нужную область без черных пикселей, но я ее пока не победил...
def crop_panorama_to_content(panorama, threshold=20):
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask)
    if coords is None:
        return panorama
    x, y, w, h = cv2.boundingRect(coords)
    return panorama[y:y+h, x:x+w]


def main():
    print("Открытие 4-х видосиков...")
    caps = [cv2.VideoCapture(src) for src in CAMERA_SOURCES]
    if not all(cap.isOpened() for cap in caps):
        print("Ошибка открытия камер."); return

    print("Предкалибровка...")
    calib_frames = []
    while len(calib_frames) < CALIB_FRAMES:
        f = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка чтения с камеры."); return
            f.append(cv2.resize(frame, FRAME_SIZE))
        calib_frames.append(f)
    all_homographies = []
    all_match_counts = []
    for frame_set in calib_frames:
        homographies, match_counts = compute_homographies(frame_set)
        all_homographies.append(homographies)
        all_match_counts.append(match_counts)
    mean_homographies = [np.mean([homos[i] for homos in all_homographies], axis=0) for i in range(NUM_CAMERAS)]
    mean_matches = np.mean(all_match_counts, axis=0)
    tr_homographies, pano_size = calc_pano_size(calib_frames[-1], mean_homographies)
    print(f"Калибровка завершена. Холст: {pano_size} | среднее совпад: {mean_matches.astype(int)}")

    frame_num, t_total = 0, 0.
    RECALIB_FREQ = 120
    while True:
        t0 = time.time()
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret: print("Видео закончилось."); return
            frames.append(cv2.resize(frame, FRAME_SIZE))

        panorama = np.zeros((pano_size[1], pano_size[0], 3), np.uint8)
        for i in range(NUM_CAMERAS):
            warped = cv2.warpPerspective(frames[i], tr_homographies[i], pano_size)
            mask = cv2.threshold(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
            panorama = cv2.bitwise_and(panorama, panorama, mask=cv2.bitwise_not(mask))
            panorama = cv2.add(panorama, warped)
        panorama = crop_panorama_to_content(panorama)

        t1 = time.time()
        frame_num += 1
        t_total += (t1-t0)
        avg_fps = frame_num/t_total if t_total>0 else 0

        if frame_num % RECALIB_FREQ == 0:
            print("Перекалибровка гомографий...")
            calib_frames = []
            for _ in range(CALIB_FRAMES):
                f = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret: print("Видео закончилось."); break
                    f.append(cv2.resize(frame, FRAME_SIZE))
                calib_frames.append(f)
            if len(calib_frames) == CALIB_FRAMES:
                all_homographies = []
                for frame_set in calib_frames:
                    homographies, _ = compute_homographies(frame_set)
                    all_homographies.append(homographies)
                mean_homographies = [np.mean([homos[i] for homos in all_homographies], axis=0) for i in range(NUM_CAMERAS)]
                tr_homographies, pano_size = calc_pano_size(calib_frames[-1], mean_homographies)
                print(f"Панорама перекалибрована. Новый размер: {pano_size}")
        print(f"Кадр {frame_num}: время {t1-t0:.4f} сек | AVG FPS: {avg_fps:.2f} | совпад: {mean_matches.astype(int)}", end="\r")
        cv2.imshow("War Helmet Panorama PROD-CROP", panorama)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    for cap in caps: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()