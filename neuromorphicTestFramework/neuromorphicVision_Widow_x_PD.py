
import socket
import serial as sr
import time
import pygame
import utilsDVS128
import matplotlib.pyplot as plt
import copy
import numpy as np
import sys
import math
import matplotlib.patches as patches
from ctrl_widow_x import widow_x

from collections import deque
sys.path.append('general/')
from threading import Lock
from threadhandler import ThreadHandler
import argparse
import time
from pathlib import Path
from filterUtils import filterUtils as fu

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

#40 e 0,45
frameTime = 33000
HOST = ''
PORT = 8000
clock = pygame.time.Clock()



def aquisicaoDvs():
    global udp, pol, x, y, ts, filaFrame,mutex, ts_LSB, ts_MSB
    msg, cliente = udp.recvfrom(5000000)
    vet = []
    for a in msg:
        vet.append(a)
    size = int(len(vet)/5)
    pol.extend(vet[ : size])
    x.extend(vet[size : 2 * size])
    y.extend(vet[2 * size : 3 * size])
    ts_LSB.extend(vet[3 * size : 4 * size])
    ts_MSB.extend(vet[4 * size : ])
    ts = list(map(lambda LSB, MSB: LSB + (MSB << 8), ts_LSB, ts_MSB))
    if sum(ts) >= frameTime:
        mutex.acquire()
        filaFrame.append([pol,y,x])
        pol, x, y, ts_LSB, ts_MSB = [], [], [], [], []
        mutex.release()



def main():
    #posição de controle
    wx = widow_x()
    wx.connect()
    pos_atual_x = wx.POSICAO_INICIAL_X #movimento horizontal
    pos_atual_y = wx.POSICAO_INICIAL_Y #movimento pra frente
    pos_atual_z = wx.POSICAO_INICIAL_Z #movimento vertical
    pos_atual_wrist_angle = wx.POSICAO_INICIAL_WRIST_ANGLE #movimento do punho
    step_wrist_angle = 4
    k_perc = 0.015
    k_x = wx.RANGE_MOVIMENTO_X*k_perc
    k_y = wx.RANGE_MOVIMENTO_Y*k_perc
    k_z = wx.RANGE_MOVIMENTO_Z*k_perc
    print("kx: " + str(k_x) + " ky: " + str(k_y) + " kz: " + str(k_z))
    input("Press enter")
    threadAquisicao = ThreadHandler(aquisicaoDvs)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    weights, imgsz = 'models/SITS/best.pt', 128
    device = select_device('') #CUDA
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    t0 = time.time()
    tw0 = time.perf_counter()
    global imageDimensions
    global filaFrame
    global mutex
    mutex = Lock()
    filaFrame = deque()
    font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 8,
    }
    stop = False
    pygame.init()
    displayEvents = utilsDVS128.DisplayDVS128(128, 128)
    global udp
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind((HOST, PORT))
    global pol, x, y, ts_LSB, ts_MSB
    pol, x, y, ts_LSB, ts_MSB = [], [], [], [], []
    rects = []
    texts = []
    detection = []

#parametros da filtragem por distancia
    flagDistanceFilter = True
    mediaMovelDistancia = [0,0,0]
    qtdeMediaMovel = 3
    thresholdDistanceFilter = 0.2
    maxDistanceImage = math.sqrt(128**2 + 128**2)
    thresholdDistanceFilter = thresholdDistanceFilter*maxDistanceImage
    lastCentroid = (0,0)
    distancia = 0
#fim dos parametros
    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            stop = True

    threadAquisicao.start()

    while True:
        if len(filaFrame) > 0:
            count = len(filaFrame)
            mutex.acquire()
            for i in range(count):
                frame = filaFrame.popleft()
                #if i == count -1:
                displayEvents.plotEventsF(frame[0],frame[1],frame[2])
                img = displayEvents.frame
                img_visualize = np.dstack([img,img,img])
                img_visualize = img_visualize.astype(np.uint8).copy()
                s = img.copy()
                s[s == 0] = 255
                s[s==127.5] = 0
                imgO = np.dstack([s,s,s])
                imgO = imgO.astype(np.uint8).copy()
                img0 = fu.median(imgO,7)
                # time when we finish processing for this frame
                new_frame_time = time.time()
                # fps will be number of frame processed in given time frame
                # since their will be most of time error of 0.001 second
                # we will be subtracting it to get more accurate result
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                # converting the fps into integer
                fps = int(fps)

                # converting the fps to string so that we can display it on frame
                # by using putText function
                fps = str(fps)
                # Padded resize
                img = letterbox(imgO, imgsz, stride=stride)[0]

                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0

                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, 0.45, 0.1, classes=opt.classes, agnostic=opt.agnostic_nms)
                #pred = non_max_suppression(pred, 0.25, 0.1, classes=opt.classes, agnostic=opt.agnostic_nms)
                t2 = time_synchronized()
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    s,im0, frame = '', img_visualize, 0
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'

                            centroid = getCentroid(xyxy)
                            dist_center, erro_x, erro_z = getError(centroid)
                            #print(erro_z)
                            k_z = 250
                            #k_z = 150
                            #k_x = 7
                            step_y = 5
                            if erro_x is not None and erro_x < 0:
                                pos_atual_x += (abs(erro_x)/k_x)
                                if pos_atual_x >= wx.LIMITE_SUPERIOR_SEGURANCA_X:
                                    pos_atual_x = wx.LIMITE_SUPERIOR_SEGURANCA_X
                            elif erro_x is not None and erro_x > 0:
                                pos_atual_x -= (abs(erro_x)/k_x)
                                if pos_atual_x <= wx.LIMITE_INFERIOR_SEGURANCA_X:
                                    pos_atual_x = wx.LIMITE_INFERIOR_SEGURANCA_X
                            if erro_z is not None and erro_z < 0:
                                pos_atual_z += (abs(erro_z)/k_z)
                                if pos_atual_z >= wx.LIMITE_SUPERIOR_SEGURANCA_Z:
                                    pos_atual_z = wx.LIMITE_SUPERIOR_SEGURANCA_Z
                            elif erro_z is not None and erro_z > 0:
                                pos_atual_z -= (abs(erro_z)/k_z)
                                if pos_atual_z <= wx.LIMITE_INFERIOR_SEGURANCA_Z:
                                    pos_atual_z = wx.LIMITE_INFERIOR_SEGURANCA_Z
                            if pos_atual_y <= wx.LIMITE_INFERIOR_SEGURANCA_Y:
                                pos_atual_y = wx.LIMITE_INFERIOR_SEGURANCA_Y
                            elif pos_atual_y >= wx.LIMITE_SUPERIOR_SEGURANCA_Y:
                                #while(time.perf_counter() - tw0 < (1/wx.FREQ_MAX)):
                                #    pass
                                #wx.sendValue(int(pos_atual_x),int(pos_atual_y),int(pos_atual_z),gripper=30)
                                #tw0 = time.perf_counter()
                                #while(time.perf_counter() - tw0 < 2):
                                #    pass
                                pos_atual_y = wx.LIMITE_INFERIOR_SEGURANCA_Y


                            if(flagDistanceFilter):

                                if(lastCentroid != (0,0)):
                                    distancia = getDistanciaPontos(centroid, lastCentroid)
                                    mediaMovelDistancia.append(distancia)
                                distanciaMedia = moving_average(mediaMovelDistancia,len(mediaMovelDistancia))
                                if(len(mediaMovelDistancia)>qtdeMediaMovel):
                                    mediaMovelDistancia = mediaMovelDistancia[1:-1]
                                if(distancia < thresholdDistanceFilter):
                                    plot_one_box(xyxy, im0, label="", color=colors[0], line_thickness=3)
                                    cv2.circle(im0, centroid, 1, colors[0], 3)
                                    lastCentroid = centroid
                                    if wx.isConnected:
                                        if(time.perf_counter() - tw0 > (1/wx.FREQ_MAX)):
                                            pos_atual_y += step_y
                                            wx.sendValue(int(pos_atual_x),int(pos_atual_y),int(pos_atual_z),delta=None)
                                            tw0 = time.perf_counter()
                                else:
                                    cv2.circle(im0, lastCentroid, 1, colors[0], 3)
                            else:
                                plot_one_box(xyxy, im0, label="", color=colors[0], line_thickness=3)
                                cv2.circle(im0, centroid, 1, colors[0], 3)
                                if wx.isConnected:
                                    if(time.perf_counter() - tw0 > (1/wx.FREQ_MAX)):
                                        pos_atual_y += step_y
                                        wx.sendValue(int(pos_atual_x),int(pos_atual_y),int(pos_atual_z),delta=None)
                                        tw0 = time.perf_counter()



                    else:
                        ##responsavel por realizar o movimento de geração de eventos
                        if(time.perf_counter() - tw0 > (1/wx.FREQ_MAX)):
                            pos_atual_wrist_angle = pos_atual_wrist_angle + step_wrist_angle
                            if pos_atual_wrist_angle <= wx.LIMITE_INFERIOR_SEGURANCA_WRIST_ANGLE:
                                pos_atual_wrist_angle = wx.LIMITE_INFERIOR_SEGURANCA_WRIST_ANGLE
                                step_wrist_angle = -step_wrist_angle
                            elif pos_atual_wrist_angle >= wx.LIMITE_SUPERIOR_SEGURANCA_WRIST_ANGLE:
                                pos_atual_wrist_angle = wx.LIMITE_SUPERIOR_SEGURANCA_WRIST_ANGLE
                                step_wrist_angle = -step_wrist_angle
                            if pos_atual_y == wx.LIMITE_INFERIOR_SEGURANCA_Y:
                                dt = wx.DELTA
                            else:
                                dt = 40
                            wx.sendValue(int(pos_atual_x),int(pos_atual_y),int(pos_atual_z),wrist=int(pos_atual_wrist_angle),delta=dt)
                            tw0 = time.perf_counter()
                    #print(f'{s}Done. ({t2 - t1:.3f}s) - {fps} FPS')

                    cv2.namedWindow('N-yolo',cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('N-yolo', 400,400)
                    cv2.imshow('N-yolo', im0)

                    cv2.waitKey(1)  # 1 millisecond
            mutex.release()
    udp.close()


def getError(centroid):
    imageDimensions = (128,128)
    distanceToCenter = None
    x_distance = None
    y_distance = None
    if len(centroid) > 0:
        distanceToCenter = math.sqrt(((centroid[0]-imageDimensions[1]/2)**2)+((centroid[1]-imageDimensions[0]/2)**2))
        x_distance = centroid[0]-(imageDimensions[1]/2)
        y_distance = centroid[1]-(imageDimensions[0]/2)
    return distanceToCenter,x_distance,y_distance


def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def getCentroid(x):
    p1, p2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    width = int(x[2]) - int(x[0])
    #print(width)
    height = int(x[3]) - int(x[1])
    c1 = int(x[0]+width/2)
    c2 = int(x[1]+height/2)
    return (c1, c2)
def getDistanciaPontos(p1,p2):
    distancia = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return distancia

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
