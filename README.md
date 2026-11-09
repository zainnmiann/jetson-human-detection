# Jetson human detection

Real-time person detection on an NVIDIA Jetson AGX Orin, using an Android phone running
[IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) as the camera.

- YOLO11s (Ultralytics) on the GPU, person class only
- Records annotated MP4s (new segment every 10 min) to `/root/recordings`
- Logs every detection to `/root/recordings/detections.csv`
  (`time, video, frame, persons, x1, y1, x2, y2, conf`)
- Live view in any browser at `http://<jetson-ip>:8090` (auto-reconnecting);
  raw MJPEG for VLC at `/stream`
- Reconnects automatically if the phone stream drops; starts on boot via systemd

## Tested setup

| | |
|---|---|
| Board | Jetson AGX Orin, 30 W mode |
| OS | Yocto (Poky 6.0.1), L4T r36.5 |
| Stack | CUDA 12.6, cuDNN 9.3, TensorRT 10.3, Python 3.14, PyTorch 2.10 (CUDA), OpenCV 4.13 (CUDA) |
| Throughput | ~10-16 fps at 1080x1920 (limited by MJPEG decode of the phone stream) |

## Install

Keep the system's CUDA builds of torch/torchvision/OpenCV — install Ultralytics without deps:

```sh
pip3 install --no-deps ultralytics polars ultralytics-thop
mkdir -p /root/human_detect && cp detect.py /root/human_detect/
cp systemd/human-detect.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now human-detect
```

The model (`yolo11s.pt`) downloads on first run.

## Phone

Install IP Webcam, put the phone on the same Wi-Fi, tap **Start server**.
Set its address with `--src` (default `http://192.168.0.102:8080/video`).

## Options

```
--src URL          camera stream            (default http://192.168.0.102:8080/video)
--model PATH       YOLO weights             (default yolo11s.pt)
--conf F           confidence threshold     (default 0.4)
--rotate MODE      none|cw|ccw|180          (default cw, phone held upright)
--out DIR          recordings directory     (default /root/recordings)
--segment-min N    minutes per MP4          (default 10)
--port N           live view port           (default 8090)
```

## Control

```sh
systemctl status|start|stop human-detect
systemctl disable human-detect     # stop auto-start on boot
journalctl -u human-detect -f      # logs
```

Stop with SIGTERM (`systemctl stop`), not `kill -9`, or the current MP4 is left unplayable.
