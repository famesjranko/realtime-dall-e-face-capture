# -*- coding: UTF-8 -*-
# Built-in Modules
import argparse
import time
import os
import sys
from pathlib import Path
from threading import Thread

# Third-Party Libraries
import base64
import requests
import numpy as np
import cv2
import torch
from numpy import random
import copy
from PIL import Image, ImageDraw
import skimage.morphology
from io import BytesIO
import openai

# Defining ROOT directory for local imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Local Modules
from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


# Global Variables
frame_store = None
mask_store = None
img_store = None
x, y, w, h = 0, 0, 0, 0
dall_e_called = False
detections = None
prompt = ""
testing_sleep = .1


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect(
    model,
    source,
    device
):
    global detections
    global frame_store
    global img_store
    
    prev_time = time.time()

    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz=(640, 640)

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    print("webcam: ", webcam)
    
    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1  # batch_size
    else:
        print('loading images', source)
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Initialize variables for storing previous recommended sleep seconds and the relative percentage for comparison
    prev_recommended_sleep_seconds = 0
    relative_percent = 30
    
    for path, im, im0s, vid_cap in dataset:        
        # Calculate frame rate
        curr_time = time.time()
        frame_rate = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # calculate recommended sleep rate for masked loop
        recommended_sleep_seconds = (1 / frame_rate) * 0.8
        recommended_sleep_seconds = round(recommended_sleep_seconds, 2)

        # update masked sleep rate with recommended rate if difference is greater than relative_percent
        if prev_recommended_sleep_seconds == 0:
            prev_recommended_sleep_seconds = recommended_sleep_seconds
        else:
            # update masked sleep rate with recommended rate if difference is greater than relative_percent
            relative_diff = abs((recommended_sleep_seconds - prev_recommended_sleep_seconds) / prev_recommended_sleep_seconds)
            if relative_diff > (relative_percent / 100):
                #print("relative_diff ", relative_diff, " > relative_percent ",relative_percent)
                #print("recommended_sleep_seconds: ", recommended_sleep_seconds)
                global testing_sleep
                testing_sleep = recommended_sleep_seconds
                prev_recommended_sleep_seconds = recommended_sleep_seconds

        # store im in global
        img_store = im
        
        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis= 0)
        else:
            orgimg = im.transpose(1, 2, 0)
        
        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]
        
        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        #print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
            # store frame and det in globals
            frame_store = im0.copy()
            detections = det

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                #for c in det[:, -1].unique():
                #   n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:15].view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()
                    
                    im0 = show_results(im0, xyxy, conf, landmarks, class_num)
            
            if True:
                cv2.imshow('result', im0)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

def masked():
    global detections
    global frame_store
    global img_store
    global dataset
    global device
    global dall_e_called
    global prompt
    
    # Set OpenAI API key
    openai.api_key = ''
    
    # Process detections
    mask_img = None
    dalle_img = None
    
    # percentage_factor is used to shift bounding boxes downward by this percentage of the image height
    percentage_factor = 2.5
    
    while True:
        global testing_sleep
        time.sleep(testing_sleep)
        
        if detections is not None and len(detections) and dall_e_called:
            # Create a copy of the detections array
            detections_copy = detections.clone()
        
            print('starting dall-e execution...')
            # Rescale boxes from img_size to im0 size
            detections_copy[:, :4] = scale_coords(img_store.shape[2:], detections_copy[:, :4], frame_store.shape).round()
        
            # Create a mask image with the same size as the original image
            frame_store1 = frame_store.copy() 
            mask = np.zeros_like(frame_store1, dtype=np.uint8)
            
            # img_height is used to get the height of the image on which the bounding boxes are drawn
            img_height = mask.shape[0]
                
            for *xyxy, conf, cls in detections_copy:
                # x, y, w, and h are used to define the bounding box coordinates
                # x corresponds to the top-left x-coordinate of the bounding box
                # y corresponds to the top-left y-coordinate of the bounding box
                # w corresponds to the width of the bounding box
                # h corresponds to the height of the bounding box
                
                x = int(xyxy[0].item()) 
                y = int(xyxy[1].item() + (img_height * (percentage_factor / 100)))
                w = int(xyxy[2].item()) - int(xyxy[0].item())
                h = int(xyxy[3].item()) - int(xyxy[1].item())
                
                # add detection box to the mask
                cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
                
            # Invert the mask image
            mask = cv2.bitwise_not(mask)
            
            # Create a copy of the original image that is transparent everywhere except for the areas around the detected objects
            mask_img = cv2.addWeighted(frame_store1, 1.0, mask, 1.0, 0)
            
            print('inverse mask done!')
            
            ## ==============================================================
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            mask_img = cv2.resize(mask_img, (1024, 1024))
            
            pil_mask = Image.fromarray(mask_img)
            pil_mask.save('mask1.png')
            
            ## new transparency, better cleaner edges
            # Open the shirt and make a clean copy before we dink with it too much
            orig = pil_mask.copy()

            # Make all background pixels (not including Nike logo) into magenta (255,0,255)
            ImageDraw.floodfill(pil_mask,xy=(0,0),value=(255,0,255),thresh=50)

            # Make into Numpy array
            n = np.array(pil_mask)

            # Mask of magenta background pixels
            bgMask =(n[:, :, 0:3] == [255,0,255]).all(2)

            # Make a disk-shaped structuring element
            strel = skimage.morphology.disk(13)

            # Perform a morphological closing with structuring element to remove blobs
            newalpha = skimage.morphology.binary_closing(bgMask, footprint=strel)

            # Perform a morphological dilation to expand mask right to edges of shirt
            newalpha = skimage.morphology.binary_dilation(newalpha, footprint=strel)

            # Make a PIL representation of newalpha, converting from True/False to 0/255
            newalphaPIL = (newalpha*255).astype(np.uint8)
            newalphaPIL = Image.fromarray(255-newalphaPIL, mode='L')

            # Put new, cleaned up image into alpha layer of original image
            orig.putalpha(newalphaPIL)
            orig.save('mask.png')
            
            frame2 = cv2.cvtColor(frame_store1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.resize(frame2, (1024, 1024))
            
            pil_frame = Image.fromarray(frame2)
            pil_frame.save('image.png')
            
            print('transparency layer done!')
            
            try:
                print('sending to dall-e...')
                # Send the stored image and its mask to OpenAI's DALL-E API
                response = openai.Image.create_edit(
                    image=open("image.png", "rb"),
                    mask=open("mask.png", "rb"),
                    prompt=prompt, #"A metallic futuristic landscape with orbs and space stuff",
                    n=1,
                    size="1024x1024"
                )

                # Get the generated image from the response
                generated_image_url = response["data"][0]["url"]

                # Download the generated image
                generated_image_data = requests.get(generated_image_url).content
                
                print('dall-e image received!')

                # Decode the image data
                mask_img = cv2.imdecode(np.frombuffer(generated_image_data, np.uint8), cv2.IMREAD_COLOR)
                mask_img = cv2.resize(mask_img, (640, 480))
                
                dalle_img_response = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    
                dalle_img_store = Image.fromarray(dalle_img_response)
                dalle_img_store.save('dalle_img.png')
                
                print('dall-e image saved!')
            
            except Exception as e:
                # Handle any exceptions that might occur
                print(e)
            ## -------------------------------------------------------------------
        elif detections is not None and len(detections):
            # Create a copy of the detections array
            detections_copy = detections.clone()
        
            # Rescale boxes from img_size to im0 size
            detections_copy[:, :4] = scale_coords(img_store.shape[2:], detections_copy[:, :4], frame_store.shape).round()
        
            # Create a mask image with the same size as the original image
            frame_store1 = frame_store.copy() 
            mask = np.zeros_like(frame_store1, dtype=np.uint8)
            
            # img_height is used to get the height of the image on which the bounding boxes are drawn
            img_height = mask.shape[0]
            
            for *xyxy, conf, cls in detections_copy:
                # x, y, w, and h are used to define the bounding box coordinates
                # x corresponds to the top-left x-coordinate of the bounding box
                # y corresponds to the top-left y-coordinate of the bounding box
                # w corresponds to the width of the bounding box
                # h corresponds to the height of the bounding box
                
                x = int(xyxy[0].item()) 
                y = int(xyxy[1].item() + (img_height * (percentage_factor / 100)))
                w = int(xyxy[2].item()) - int(xyxy[0].item())
                h = int(xyxy[3].item()) - int(xyxy[1].item())
                
                # add detection box to the mask
                cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
                
            # Invert the mask image
            mask = cv2.bitwise_not(mask)
            
            # Create a copy of the original image that is transparent everywhere except for the areas around the detected objects
            mask_img = cv2.addWeighted(frame_store1, 1.0, mask, 1.0, 0)
        elif frame_store is not None:
            # Either show full frame or full masked
            #frame_store1 = frame_store.copy() 
            #mask_img = np.zeros_like(frame_store1, dtype=np.uint8)
            mask_img = frame_store.copy() 
        else:
            continue
        
        if mask_img is not None and mask_img.any():
            cv2.imshow("masked", mask_img) # change to mask_img to see the masked version

        # Break the loop if 'q' key is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
            
        if dall_e_called:
            dall_e_called = False
            key = input('press ENTER to start again.')
        
    cv2.destroyAllWindows()


def call_dall_e():
    """Prompts the user to press ENTER to call Dall-E.
    """
    global dall_e_called
    global prompt
    
    print("\n================================================================")
    print("Hi and welcome to the dall-e selfulation space machine!")
    print("Just follow the prompts, be patient, and dall-e will provide...")
    print("================================================================")
    
    while True:
        time.sleep(.1)
        
        if dall_e_called == False:
            prompt = input('\ninput desired dall-e prompt or \'q\' to quit:')
            
            if prompt == "q":
                break
            
            key = input('press ENTER to call Dall-E!')

            if key == "":
                dall_e_called = True

def parse_opt():
    parser = argparse.ArgumentParser()
    # WEIGHTS OPTIONS: 'weights/yolov5n-0.5.pt', 'weights/yolov5n-face.pt', 'weights/yolov5s-face.pt'
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5n-0.5.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()
    
    return opt          


if __name__ == '__main__':
    # collect any passed arguments
    opt = parse_opt()
    
    # set device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    
    # build detection thread
    detect_thread = Thread(target=detect, args=(model, opt.source, device,))
    detect_thread.daemon = True
    
    # build masking/dall-e thread
    masked_thread = Thread(target=masked)
    masked_thread.daemon = True
    
    # start threads
    detect_thread.start()
    masked_thread.start()
    
    # run user input and controls
    time.sleep(3)
    call_dall_e()
