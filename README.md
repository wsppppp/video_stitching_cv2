# Команды для запуска демо RTSP потоков через Docker и ffmpeg

---

### 1. Запуск RTSP-сервера **mediamtx** в Docker
```sh
docker run --rm -it -p 8554:8554 bluenviron/mediamtx
```

---

### 2. Запуск каждого mp4-файла как отдельного RTSP-потока через ffmpeg

В 4 отдельных терминалах по одной команде
```sh
ffmpeg -re -stream_loop -1 -fflags +genpts -i 1.mp4 -rtsp_transport tcp -c copy -f rtsp rtsp://127.0.0.1:8554/stream1
ffmpeg -re -stream_loop -1 -fflags +genpts -i 2.mp4 -rtsp_transport tcp -c copy -f rtsp rtsp://127.0.0.1:8554/stream2
ffmpeg -re -stream_loop -1 -fflags +genpts -i 3.mp4 -rtsp_transport tcp -c copy -f rtsp rtsp://127.0.0.1:8554/stream3
ffmpeg -re -stream_loop -1 -fflags +genpts -i 4.mp4 -rtsp_transport tcp -c copy -f rtsp rtsp://127.0.0.1:8554/stream4

ffmpeg -re -stream_loop -1 -fflags +genpts -i "rtsp://admin:kontour1@169.254.102.84:554/Streaming/Channels/101" -rtsp_transport tcp -c copy -f rtsp rtsp://127.0.0.1:8554/stream1
```

---

### 3. Адреса RTSP-потоков для подключения из кода

```python
CAMERA_SOURCES = [
    "rtsp://127.0.0.1:8554/stream1",
    "rtsp://127.0.0.1:8554/stream2",
    "rtsp://127.0.0.1:8554/stream3",
    "rtsp://127.0.0.1:8554/stream4"
    
    "rtsp://admin:kontour1@169.254.102.84:554/Streaming/Channels/101"
]
```

---

# Запуск скрипта панорамы и ретрансляция по UDP

После того как RTSP-потоки запущены (см. пункты 1–3), запускаем скрипт, который:

- подключается к RTSP-источникам из `CAMERA_SOURCES`,
- сшивает кадры в панораму,
- кодирует её в H.264 и ретранслирует по UDP.

```sh
# в активированном venv (если используете его)
python panorama_udp_stream.py
```

По умолчанию скрипт отправляет поток на:

```text
udp://127.0.0.1:5000
```

Его можно просмотреть, например, через ffplay:

```sh
ffplay -fflags nobuffer -flags low_delay \
       -i "udp://127.0.0.1:5000?fifo_size=1000000&overrun_nonfatal=1"
```

или подключить этот UDP-поток из другого приложения (например, Godot или VLC плеере).