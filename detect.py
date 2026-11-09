#!/usr/bin/env python3
"""Human detection on an IP Webcam stream: records annotated MP4 + CSV log, serves live MJPEG view."""
import argparse, csv, os, signal, threading, time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="http://192.168.0.102:8080/video")
ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--out", default="/root/recordings")
ap.add_argument("--segment-min", type=float, default=10, help="start a new MP4 every N minutes")
ap.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="cw", help="fix phone orientation")
ap.add_argument("--port", type=int, default=8090, help="live view at http://<jetson>:PORT/")
args = ap.parse_args()
signal.signal(signal.SIGTERM, signal.default_int_handler)  # kill -> clean close of MP4/CSV
signal.signal(signal.SIGINT, signal.default_int_handler)
os.makedirs(args.out, exist_ok=True)


class Grabber:
    """Reads the stream in a thread and keeps only the newest frame (no lag build-up)."""
    def __init__(self, src):
        self.src, self.frame, self.ts, self.lock = src, None, 0.0, threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("stream not available, retrying in 2s", flush=True); time.sleep(2); continue
            while True:
                ok, f = cap.read()
                if not ok:
                    print("stream dropped, reconnecting", flush=True); break
                with self.lock:
                    self.frame, self.ts = f, time.time()
            cap.release(); time.sleep(1)

    def get(self, after):
        with self.lock:
            return (self.frame, self.ts) if self.ts > after else (None, after)


latest_jpg, frame_id, frame_cv = None, 0, threading.Condition()

def publish(jpg):
    global latest_jpg, frame_id
    with frame_cv:
        latest_jpg, frame_id = jpg, frame_id + 1
        frame_cv.notify_all()

def wait_frame(after, timeout=5):
    with frame_cv:
        frame_cv.wait_for(lambda: frame_id > after, timeout=timeout)
        return latest_jpg, frame_id

# Page pulls single JPEGs in a loop: each request is independent, so a hiccup
# never freezes the view, and a watchdog restarts the loop if it stalls.
PAGE = b"""<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>Jetson live view</title><style>body{margin:0;background:#111;color:#ccc;font:14px sans-serif;text-align:center}
img{max-width:100vw;max-height:94vh}#s{padding:4px}</style></head><body><img id=v><div id=s>connecting...</div><script>
const v=document.getElementById('v'),s=document.getElementById('s');let last=Date.now(),n=0,t0=Date.now(),gen=0,fid=0;
async function loop(g){while(g===gen){try{const c=new AbortController(),to=setTimeout(()=>c.abort(),3000);
const r=await fetch('/frame.jpg?f='+fid+'&t='+Date.now(),{signal:c.signal,cache:'no-store'});const b=await r.blob();clearTimeout(to);fid=+(r.headers.get('X-Frame')||0);
if(g!==gen)return;const u=URL.createObjectURL(b);v.onload=()=>URL.revokeObjectURL(u);v.src=u;last=Date.now();n++;
if(Date.now()-t0>2000){s.textContent='live - '+(n*1000/(Date.now()-t0)).toFixed(1)+' fps';n=0;t0=Date.now();}}
catch(e){s.textContent='reconnecting...';await new Promise(r=>setTimeout(r,500));}}}
setInterval(()=>{if(Date.now()-last>4000){gen++;last=Date.now();loop(gen);}},1000);loop(gen);
</script></body></html>"""

class Live(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _send(self, ctype, body, fid=None):
        self.send_response(200)
        if fid is not None: self.send_header("X-Frame", str(fid))
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        try:
            path = self.path.split("?")[0]
            if path == "/":
                self._send("text/html", PAGE)
            elif path == "/frame.jpg":
                try: after = int(self.path.split("f=")[1].split("&")[0])
                except (IndexError, ValueError): after = 0
                jpg, fid = wait_frame(after, timeout=2)  # block until a frame newer than the client's
                if jpg is None: self.send_error(503); return
                self._send("image/jpeg", jpg, fid)
            elif path == "/stream":  # plain MJPEG, e.g. for VLC
                self.send_response(200); self.protocol_version = "HTTP/1.0"
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.send_header("Cache-Control", "no-store"); self.end_headers()
                fid = 0
                while True:
                    jpg, fid = wait_frame(fid)
                    if jpg is None: continue
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n")
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

srv = ThreadingHTTPServer(("0.0.0.0", args.port), Live)
srv.daemon_threads = True
threading.Thread(target=srv.serve_forever, daemon=True).start()

model = YOLO(args.model)
grab = Grabber(args.src)
log = open(os.path.join(args.out, "detections.csv"), "a", newline="")
logw = csv.writer(log)
if log.tell() == 0:
    logw.writerow(["time", "video", "frame", "persons", "x1", "y1", "x2", "y2", "conf"])

writer, vid_name, seg_start, frame_no, last_ts = None, "", 0.0, 0, 0.0
fps_t, fps_n, fps, warm = time.time(), 0, 0.0, 0
print(f"running; live view on port {args.port}, recording to {args.out}", flush=True)
try:
    while True:
        frame, last_ts = grab.get(last_ts)
        if frame is None:
            time.sleep(0.005); continue
        if args.rotate != "none":
            frame = cv2.rotate(frame, {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}[args.rotate])
        res = model.predict(frame, classes=[0], conf=args.conf, verbose=False)[0]
        boxes = res.boxes
        out = res.plot(line_width=2)
        now = datetime.now()
        cv2.putText(out, f"{now:%Y-%m-%d %H:%M:%S}  persons: {len(boxes)}  {fps:.1f} fps",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        fps_n += 1
        if time.time() - fps_t >= 2:
            fps, fps_t, fps_n = fps_n / (time.time() - fps_t), time.time(), 0
            warm += 1  # first window includes CUDA warm-up; ignore it
            log.flush()

        # new segment: needs a measured fps so playback speed matches real time
        if warm >= 2 and (writer is None or time.time() - seg_start > args.segment_min * 60):
            if writer: writer.release()
            vid_name = f"rec_{now:%Y%m%d_%H%M%S}.mp4"
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(os.path.join(args.out, vid_name), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"mp4v"), max(1, round(fps)), (w, h))
            seg_start, frame_no = time.time(), 0
            print(f"recording {vid_name} {w}x{h} @ {max(1, round(fps))} fps", flush=True)
            log.flush()
        if writer:
            writer.write(out); frame_no += 1
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                logw.writerow([now.isoformat(timespec="milliseconds"), vid_name, frame_no, len(boxes), x1, y1, x2, y2, round(float(b.conf), 3)])

        # live view: downscale to 720 px tall so it stays smooth over Wi-Fi
        h = out.shape[0]
        small = cv2.resize(out, None, fx=720 / h, fy=720 / h, interpolation=cv2.INTER_AREA) if h > 720 else out
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok: publish(jpg.tobytes())
except KeyboardInterrupt:
    pass
finally:
    if writer: writer.release()
    log.close()
    print("stopped, files closed", flush=True)
