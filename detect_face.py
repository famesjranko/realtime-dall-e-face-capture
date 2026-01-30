# -*- coding: UTF-8 -*-
"""Realtime face detection + local text-to-image generation."""
from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path
from queue import Queue, Empty
from threading import Event, Thread
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Local modules
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(ROOT)

from config import GenerationConfig, build_generation_config, detect_device
from local_generator import generate_image, init_pipeline
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams, img_formats, letterbox, vid_formats
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.torch_utils import time_synchronized

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

# Shared state
frame_store: Optional[np.ndarray] = None
img_store: Optional[np.ndarray] = None
detections = None
last_generated_image: Optional[np.ndarray] = None
generation_queue: Queue = Queue()
testing_sleep = 0.1
last_face_crop: Optional[Image.Image] = None


def load_model(weights: str, device: torch.device):
    # Allow YOLO model class for safe loading on newer torch defaults
    try:
        from models.yolo import Model as YoloModel
        import torch.serialization

        torch.serialization.add_safe_globals([YoloModel])
    except Exception:
        pass

    # attempt_load signature is legacy (no weights_only), so call directly
    model = attempt_load(weights, map_location=device)
    model.eval()
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]
    coords[:, :10] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])
    coords[:, 8].clamp_(0, img0_shape[1])
    coords[:, 9].clamp_(0, img0_shape[0])
    return coords


def show_results(img, xyxy, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect(model, source: str, device: torch.device, stop_event: Event):
    global detections, frame_store, img_store, last_generated_image, last_face_crop

    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz = (img_size, img_size)

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    logging.info(f"webcam: {webcam}")

    dataset = LoadStreams(source, img_size=imgsz) if webcam else LoadImages(source, img_size=imgsz)

    prev_time = time.time()

    for path, im, im0s, vid_cap in dataset:
        if stop_event.is_set():
            break

        curr_time = time.time()
        frame_rate = 1 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        global testing_sleep
        recommended_sleep_seconds = round((1 / frame_rate) * 0.8, 2)
        testing_sleep = recommended_sleep_seconds

        img_store = im

        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis=0)
        else:
            orgimg = im.transpose(1, 2, 0)

        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]
        r = img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())

        img = letterbox(img0, new_shape=imgsz)[0]
        img = img.transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]

        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy() if webcam else im0s.copy()
            frame_store = im0.copy()
            detections = det

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:15].view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()
                    im0 = show_results(im0, xyxy, conf, landmarks, class_num)
                # store first face crop for IP-Adapter
                x1, y1, x2, y2 = map(int, det[0, :4].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(im0.shape[1], x2), min(im0.shape[0], y2)
                crop = im0[y1:y2, x1:x2]
                if crop.size != 0:
                    last_face_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            cv2.imshow("result", im0)
            if last_generated_image is not None:
                cv2.imshow("generated", last_generated_image)

            key = cv2.waitKey(1)
            if key == ord("q"):
                stop_event.set()
                break

        if stop_event.is_set():
            break

    # For image sources, keep windows open to allow generation thread to display results
    if not webcam:
        # For image sources, keep windows responsive until stop_event is set
        while not stop_event.is_set():
            if frame_store is not None:
                cv2.imshow("result", frame_store)
            if last_generated_image is not None:
                cv2.imshow("generated", last_generated_image)
            key = cv2.waitKey(50)
            if key == ord("q"):
                stop_event.set()
                break

    cv2.destroyAllWindows()


def generator_worker(gen_config: GenerationConfig, stop_event: Event):
    global last_generated_image

    logging.info(
        f"Loading diffusion pipeline: {gen_config.model_id} (preset={gen_config.preset_key}, device={gen_config.device})"
    )
    bundle = init_pipeline(gen_config)
    logging.info("Pipeline ready. Waiting for prompts...")

    while not stop_event.is_set():
        try:
            job = generation_queue.get(timeout=0.1)
        except Empty:
            continue

        prompt = job["prompt"]
        seed = job.get("seed", gen_config.seed)
        face_image = job.get("face_image")
        try:
            start = time_synchronized()
            image = generate_image(
                bundle,
                prompt=prompt,
                config=gen_config,
                seed=seed,
                face_image=face_image,
            )
            duration = time_synchronized() - start
            last_generated_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            Image.fromarray(cv2.cvtColor(last_generated_image, cv2.COLOR_BGR2RGB)).save("generated.png")
            logging.info(f"Generated image in {duration:.2f}s -> generated.png")
        except Exception as exc:
            logging.error(f"Generation failed: {exc}")
        finally:
            generation_queue.task_done()


def prompt_loop(stop_event: Event, default_seed: Optional[int]):
    while not stop_event.is_set():
        try:
            prompt = input("\nEnter prompt (or 'q' to quit): ")
        except EOFError:
            stop_event.set()
            break

        if prompt.strip().lower() == "q":
            stop_event.set()
            break

        generation_queue.put({"prompt": prompt, "seed": default_seed, "face_image": last_face_crop})
        logging.info("Queued prompt for generation")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="weights/yolov5n-0.5.pt", help="model.pt path(s)")
    parser.add_argument("--source", type=str, default="0", help="source")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["sdxl", "sdxl-turbo", "sd15"],
        help="diffusion model preset",
    )
    parser.add_argument(
        "--face-adapter",
        type=str,
        default=None,
        choices=["none", "faceid-sdxl", "faceid-sd15"],
        help="enable IP-Adapter FaceID (auto-select if omitted)",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device string (cuda, cpu, mps)")
    parser.add_argument("--low-vram", action="store_true", help="enable VRAM-saving settings")
    parser.add_argument("--steps", type=int, default=None, help="override num inference steps")
    parser.add_argument("--guidance-scale", type=float, default=None, help="override guidance scale")
    parser.add_argument("--negative-prompt", type=str, default="", help="negative prompt text")
    parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducibility")
    parser.add_argument("--nsfw-check", action="store_true", help="enable safety checker (off by default)")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()

    device_info = detect_device(opt.device)
    gen_config = build_generation_config(
        device_info,
        requested_model=opt.model,
        requested_adapter=opt.face_adapter,
        low_vram=opt.low_vram,
        steps=opt.steps,
        guidance_scale=opt.guidance_scale,
        negative_prompt=opt.negative_prompt,
        safety_checker=opt.nsfw_check,
        seed=opt.seed,
    )

    logging.info(
        f"Selected diffusion preset: {gen_config.preset_key} (device={device_info.device}, vram={device_info.total_vram_gb} GB)"
    )
    if gen_config.face_adapter:
        logging.info(f"IP-Adapter FaceID enabled ({gen_config.face_adapter.key})")
    else:
        logging.info("IP-Adapter FaceID disabled (text-only generation)")
    if device_info.device.type == "cpu":
        logging.warning("CPU fallback active: expect slow generations. Add --device cuda or --device mps when available.")
    if gen_config.low_vram:
        logging.warning("Low-VRAM mode enabled: using reduced resolution/optimizations for stability.")

    device = gen_config.device
    model = load_model(opt.weights, device)

    stop_event = Event()

    detect_thread = Thread(target=detect, args=(model, opt.source, device, stop_event), daemon=True)
    generator_thread = Thread(target=generator_worker, args=(gen_config, stop_event), daemon=True)

    detect_thread.start()
    generator_thread.start()

    try:
        prompt_loop(stop_event, opt.seed)
    except KeyboardInterrupt:
        logging.info("Interrupted, shutting down...")
        stop_event.set()
    finally:
        stop_event.set()
        detect_thread.join(timeout=2)
        generator_thread.join(timeout=2)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
