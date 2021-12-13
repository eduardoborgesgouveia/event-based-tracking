from threading import Lock
import socket
import serial as sr
import time
import pygame
import utilsDVS128
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
import matplotlib.patches as patches
from ctrl_widow_x import widow_x
from collections import Counter
from collections import deque

from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.plots import plot_one_box
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.datasets import LoadStreams, LoadImages
from models.experimental import attempt_load
import json
from datetime import datetime
from os import path
import os
from numpy import random
import torch.backends.cudnn as cudnn
import torch
import cv2
from filterUtils import filterUtils as fu
from pathlib import Path
import argparse
import sys

sys.path.append('general/')
from threadhandler import ThreadHandler

# 40 e 0,45
frameTime = 30000
HOST = ''
PORT = 8000
clock = pygame.time.Clock()


def aquisicaoDvs():
    global udp, pol, x, y, ts, filaFrame, mutex, ts_LSB, ts_MSB
    msg, cliente = udp.recvfrom(5000000)
    vet = []
    for a in msg:
        vet.append(a)
    size = int(len(vet) / 5)
    pol.extend(vet[:size])
    x.extend(vet[size:2 * size])
    y.extend(vet[2 * size:3 * size])
    ts_LSB.extend(vet[3 * size:4 * size])
    ts_MSB.extend(vet[4 * size:])
    ts = list(map(lambda LSB, MSB: LSB + (MSB << 8), ts_LSB, ts_MSB))
    if sum(ts) >= frameTime:
        mutex.acquire()
        filaFrame.append([pol, y, x])
        pol, x, y, ts_LSB, ts_MSB = [], [], [], [], []
        mutex.release()


def main():
    global wx
    # posição de controle
    wx = widow_x()
    wx.connect()
    pos_atual_x = wx.POSICAO_INICIAL_X  # movimento horizontal
    pos_atual_y = wx.POSICAO_INICIAL_Y  # movimento pra frente
    pos_atual_z = wx.POSICAO_INICIAL_Z  # movimento vertical
    pos_atual_wrist_angle = wx.POSICAO_INICIAL_WRIST_ANGLE  # movimento do punho
    last_pos_atual_x = pos_atual_x
    last_pos_atual_z = pos_atual_z
    last_pos_atual_y = pos_atual_y
    step_wrist_angle = 4
    k_perc = 0.015
    k_x = wx.RANGE_MOVIMENTO_X * k_perc * 4
    k_y = wx.RANGE_MOVIMENTO_Y * k_perc
    k_z = wx.RANGE_MOVIMENTO_Z * k_perc * 8
    #k_z = 250
    print("kx: " + str(k_x) + " ky: " + str(k_y) + " kz: " + str(k_z))
    input("Press enter")
    threadAquisicao = ThreadHandler(aquisicaoDvs)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',nargs='+',type=str,default='yolov5s.pt',help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source',type=str,default='data/images',help='source')
    parser.add_argument('--img-size',type=int, default=640,help='inference size (pixels)')
    # parser.add_argument('--conf-thres',type=float,default=0.25,help='object confidence threshold')
    # parser.add_argument('--iou-thres',type=float,default=0.45,help='IOU threshold for NMS')
    parser.add_argument('--device',default='',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',action='store_true',help='display results')
    parser.add_argument('--save-txt',action='store_true',help='save results to *.txt')
    parser.add_argument('--save-conf',action='store_true',help='save confidences in --save-txt labels')
    parser.add_argument('--classes',nargs='+',type=int,help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',help='class-agnostic NMS')
    parser.add_argument('--augment',action='store_true',help='augmented inference')
    parser.add_argument('--update',action='store_true',help='update all models')
    parser.add_argument('--project',default='runs/detect',help='save results to project/name')
    parser.add_argument('--exist-ok',action='store_true',help='existing project/name ok, do not increment')
    parser.add_argument('--path-to-save',type=str,default="data_experimentos", help='path to save the active tracking data')
    parser.add_argument('--name',type=str,default="experimento",help='name of the file to save information')
    parser.add_argument('--model',type=str,default="models/SITS/last.pt",help='path to weights')
    parser.add_argument('--conf-thresh',type=float, default=0.30,help='confidence of the predictions')
    parser.add_argument('--iou-thresh',type=float, default=0.3,help='confidence of the predictions')
    parser.add_argument('--speed', type=float, default=1, help='widowX speed')
    parser.add_argument('--num-objects',type=int,default=1,help='number of objects in scene')
    parser.add_argument('--id',type=str,default="sem_id",help='id for identification')
    parser.add_argument('--objeto-interesse',type=int,default=None,help='class of the object of interest:::  0: Banana 1: Cup  2: Fork    3: Key    4: Knife    5: Mug    6: Orange')

    opt = parser.parse_args()
    obj_interesse = opt.objeto_interesse
    predicao_classes = []
    if opt.speed == 1:
        wx.DELTA = 60
    elif opt.speed == 2:
        wx.DELTA = 120

    weights, imgsz = opt.model, 128
    device = select_device('')  # CUDA
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
    font = {
        'family': 'serif',
        'color': 'white',
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
    qtde_nao_deteccao = 0

    # parametros da filtragem por distancia
    flagDistanceFilter = True
    mediaMovelDistancia = [0, 0, 0]
    qtdeMediaMovel = 3
    thresholdDistanceFilter = 0.25
    maxDistanceImage = math.sqrt(128**2 + 128**2)
    thresholdDistanceFilter = thresholdDistanceFilter * maxDistanceImage
    lastCentroid = (0, 0)
    distancia = 0
    # fim dos parametros
    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            stop = True

    threadAquisicao.start()

    flagAlcance = True

    qtde_frames = 0
    qtde_deteccoes = 0
    qtde_deteccoes_corretas = 0
    error_prior_z = 0
    integral_prior_z = 0
    error_prior_x = 0
    integral_prior_x = 0
    iteration_time = (1 / wx.FREQ_MAX)
    Kp_x = 0.02 * k_x
    Ki_x = 0.05 * Kp_x / iteration_time
    Kd_x = (Kp_x * iteration_time) / 160
    Kp_z = 0.2 * k_z
    Ki_z = 0.05 * Kp_z / iteration_time
    Kd_z = (Kp_z * iteration_time) / 160
    qtde_predicoes_tensor = []
    qtde_deteccoes_acc_temp_totais = []
    qtde_deteccoes_acc_temp_validas = []
    taxa_acc_temp = []
    tipo_deteccao = []
    tempo_sacada = []
    tsF = 0
    ts0 = 0
    zf = 0
    time_control = 0
    frames_originais = []
    frames_com_deteccao = []
    while flagAlcance:
        if len(filaFrame) > 0:
            if time_control == 0:
                ti_alcance = time.time()
                time_control += 1
            count = len(filaFrame)

            for i in range(count):
            #if zf == count:
              zf = 0
              qtde_frames += 1
              mutex.acquire()
              frame = filaFrame.popleft()
              mutex.release()
              # if i == count -1:
              displayEvents.plotEventsF(frame[0], frame[1], frame[2])
              img = displayEvents.frame
              img_visualize = np.dstack([img, img, img])
              img_visualize = img_visualize.astype(np.uint8).copy()
              img_save = img_visualize.astype(np.uint8).copy()
              frames_originais.append(img_save)
              s = img.copy()
              s[s == 0] = 255
              s[s == 127.5] = 0
              imgO = np.dstack([s, s, s])
              imgO = imgO.astype(np.uint8).copy()
              #img0 = fu.median(imgO,7)
              # time when we finish processing for this frame
              new_frame_time = time.time()
              # fps will be number of frame processed in given time frame
              # since their will be most of time error of 0.001 second
              # we will be subtracting it to get more accurate result
              fps = 1 / (new_frame_time - prev_frame_time)
              prev_frame_time = new_frame_time
              # converting the fps into integer
              fps = int(fps)

              # converting the fps to string so that we can display it on frame
              # by using putText function
              fps = str(fps)
              # Padded resize
              img = letterbox(imgO, imgsz, stride=stride)[0]

              # Convert
              # BGR to RGB, to 3x416x416
              img = img[:, :, ::-1].transpose(2, 0, 1)
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
              pred = non_max_suppression(pred,
                                          opt.conf_thresh,
                                          opt.iou_thresh,
                                          classes=opt.classes,
                                          agnostic=opt.agnostic_nms)
              #pred = non_max_suppression(pred, 0.25, 0.1, classes=opt.classes, agnostic=opt.agnostic_nms)
              # qtde_predicoes_tensor.append(len(pred))
              t2 = time_synchronized()
              # Process detections
              for i, det in enumerate(pred):  # detections per image
                  s, im0, frame = '', img_visualize, 0
                  s += '%gx%g ' % img.shape[2:]  # print string
                  # normalization gain whwh
                  gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                  qtde_predicoes_tensor.append(len(det))
                  if len(det):
                      qtde_deteccoes += 1

                      # Rescale boxes from img_size to im0 size
                      det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                                im0.shape).round()

                      # Print results
                      for c in det[:, -1].unique():
                          n = (det[:, -1] == c).sum()  # detections per class
                          # add to string
                          s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                      index_min, dist_center, centroid, erro_x, erro_z = getCloserToCenter(det, lastCentroid)

                      # Write results
                      *xyxy, conf, cls = reversed(det)[index_min]
                      label = f'{names[int(cls)]} {conf:.2f}'
                      predicao_classes.append(int(cls))

                      if (flagDistanceFilter):
                          if len(tipo_deteccao) > 0 and tipo_deteccao[-1] == "sacada":
                              tempo_sacada.append(time.perf_counter() - ts0)
                          if (lastCentroid != (0, 0)):
                              distancia = getDistanciaPontos(centroid, lastCentroid)
                              mediaMovelDistancia.append(distancia)
                          distanciaMedia = moving_average(mediaMovelDistancia, len(mediaMovelDistancia))
                          if (len(mediaMovelDistancia) > qtdeMediaMovel):
                              mediaMovelDistancia = mediaMovelDistancia[1:-1]
                          if (distancia < thresholdDistanceFilter):
                              plot_one_box(xyxy,
                                            im0,
                                            label="",
                                            color=colors[0],
                                            line_thickness=3)
                              cv2.circle(im0, centroid, 1, colors[0], 3)
                              lastCentroid = centroid
                              qtde_deteccoes_corretas += 1
                              tipo_deteccao.append("valida")

                              step_y = 2
                              if erro_x is not None and erro_x < 0:
                                  pos_atual_x += (abs(erro_x) / k_x)
                                  if pos_atual_x >= wx.LIMITE_SUPERIOR_SEGURANCA_X:
                                      pos_atual_x = wx.LIMITE_SUPERIOR_SEGURANCA_X
                              elif erro_x is not None and erro_x > 0:
                                  pos_atual_x -= (abs(erro_x) / k_x)
                                  if pos_atual_x <= wx.LIMITE_INFERIOR_SEGURANCA_X:
                                      pos_atual_x = wx.LIMITE_INFERIOR_SEGURANCA_X
                              if erro_z is not None and erro_z < 0:
                                  pos_atual_z += (abs(erro_z) / k_z)
                                  if pos_atual_z >= wx.LIMITE_SUPERIOR_SEGURANCA_Z:
                                      pos_atual_z = wx.LIMITE_SUPERIOR_SEGURANCA_Z
                              elif erro_z is not None and erro_z > 0:
                                  pos_atual_z -= (abs(erro_z) / k_z)
                                  if pos_atual_z <= wx.LIMITE_INFERIOR_SEGURANCA_Z:
                                      pos_atual_z = wx.LIMITE_INFERIOR_SEGURANCA_Z
                              if pos_atual_y <= wx.LIMITE_INFERIOR_SEGURANCA_Y:
                                  pos_atual_y = wx.LIMITE_INFERIOR_SEGURANCA_Y
                              elif pos_atual_y >= wx.LIMITE_SUPERIOR_SEGURANCA_Y:
                                  flagAlcance = False
                                  tf_alcance = time.time()
                                  pos_atual_y = wx.LIMITE_INFERIOR_SEGURANCA_Y

                              if wx.isConnected:
                                  while (time.perf_counter() - tw0 <(1 / wx.FREQ_MAX)):
                                      pass
                                  if (time.perf_counter() - tw0 >(1 / wx.FREQ_MAX)):
                                      pos_atual_y += step_y
                                      wx.sendValue(int(pos_atual_x),
                                                  int(pos_atual_y),
                                                  int(pos_atual_z),
                                                  wrist=int(pos_atual_wrist_angle))
                                      tw0 = time.perf_counter()

                          else:
                              tipo_deteccao.append("invalida")
                              cv2.circle(im0, lastCentroid, 1, colors[0], 3)
                              # step_y = 3

                              # if wx.isConnected:
                              #     while (time.perf_counter() - tw0 <(1 / wx.FREQ_MAX)):
                              #         pass
                              #     if (time.perf_counter() - tw0 >(1 / wx.FREQ_MAX)):
                              #         pos_atual_y += step_y
                              #         wx.sendValue(int(pos_atual_x),
                              #                      int(pos_atual_y),
                              #                      int(pos_atual_z),
                              #                      delta=None)
                              #         tw0 = time.perf_counter()

                      else:
                          #else do filtro de filtragem por distância
                          plot_one_box(xyxy,
                                        im0,
                                        label="",
                                        color=colors[0],
                                        line_thickness=3)
                          cv2.circle(im0, centroid, 1, colors[0], 3)
                          if wx.isConnected:
                              while (time.perf_counter() - tw0 <(1 / wx.FREQ_MAX)):
                                  pass
                              if (time.perf_counter() - tw0 > (1 / wx.FREQ_MAX)):
                                  pos_atual_y += step_y
                                  wx.sendValue(int(pos_atual_x),
                                              int(pos_atual_y),
                                              int(pos_atual_z),
                                              wrist=int(pos_atual_wrist_angle))
                                  tw0 = time.perf_counter()

                      if qtde_deteccoes % 5 == 0:

                          if len(qtde_deteccoes_acc_temp_validas) == 0:
                              last_value_val = 0
                          else:
                              last_value_val = qtde_deteccoes_acc_temp_validas[-1]

                          if len(qtde_deteccoes_acc_temp_totais) == 0:
                              last_value_tot = 0
                          else:
                              last_value_tot = qtde_deteccoes_acc_temp_totais[-1]

                          qtde_deteccoes_acc_temp_totais.append(
                              qtde_deteccoes)
                          qtde_deteccoes_acc_temp_validas.append(
                              qtde_deteccoes_corretas)

                          taxa_acc_temp.append(
                              (qtde_deteccoes_corretas - last_value_val) /
                              (qtde_deteccoes - last_value_tot))

                  else:
                      # qtde_nao_deteccao += 1
                      # if qtde_nao_deteccao > 5:
                      #   qtde_nao_deteccao = 0
                      # responsavel por realizar o movimento de geração de eventos
                      if len(tipo_deteccao) == 0 or tipo_deteccao[-1] != "sacada":
                          ts0 = time.perf_counter()

                      tipo_deteccao.append("sacada")
                      while (time.perf_counter() - tw0 < (1 / wx.FREQ_MAX)):
                          pass
                      
                      if (time.perf_counter() - tw0 > (1 / wx.FREQ_MAX)):
                          
                          pos_atual_wrist_angle = pos_atual_wrist_angle + step_wrist_angle
                          # if pos_atual_wrist_angle <= wx.POSICAO_INICIAL_WRIST_ANGLE - 10:
                          #     step_wrist_angle = -step_wrist_angle
                          # elif pos_atual_wrist_angle >= wx.POSICAO_INICIAL_WRIST_ANGLE + 10:
                          #     step_wrist_angle = -step_wrist_angle
                          if pos_atual_wrist_angle <= wx.LIMITE_INFERIOR_SEGURANCA_WRIST_ANGLE:
                              #pos_atual_wrist_angle = wx.LIMITE_INFERIOR_SEGURANCA_WRIST_ANGLE
                              step_wrist_angle = -step_wrist_angle
                          elif pos_atual_wrist_angle >= wx.LIMITE_SUPERIOR_SEGURANCA_WRIST_ANGLE:
                              #pos_atual_wrist_angle = wx.LIMITE_SUPERIOR_SEGURANCA_WRIST_ANGLE
                              step_wrist_angle = -step_wrist_angle
                          # if pos_atual_y == wx.LIMITE_INFERIOR_SEGURANCA_Y:
                          #     dt = wx.DELTA
                          # else:
                          #     dt = 40
                          # wx.sendValue(int(pos_atual_x),
                          #              int(pos_atual_y),
                          #              int(pos_atual_z),
                          #              wrist=int(pos_atual_wrist_angle),
                          #              delta=dt)
                          wx.sendValue(int(pos_atual_x),
                                        int(pos_atual_y),
                                        int(pos_atual_z),
                                        wrist=int(pos_atual_wrist_angle),
                                        delta=25)
                          tw0 = time.perf_counter()
                          
                          # if len(tipo_deteccao) > 2 and tipo_deteccao[-1] != "sacada" and tipo_deteccao[-2] :

                  #print(f'{s}Done. ({t2 - t1:.3f}s) - {fps} FPS')

              cv2.namedWindow('N-yolo', cv2.WINDOW_NORMAL)
              cv2.resizeWindow('N-yolo', 400, 400)
              cv2.imshow('N-yolo', im0)
              frames_com_deteccao.append(im0)

              cv2.waitKey(1)  # 1 millisecond
            #else:
                #zf += 1

    tempo_alcance = tf_alcance - ti_alcance
    taxa_erro_percentual_frames = qtde_deteccoes / qtde_frames
    taxa_erro_percentual_deteccoes = qtde_deteccoes_corretas / qtde_deteccoes

    c_t = Counter(tipo_deteccao)

    # PREPARACAO PARA SALVAR AS INFORMAÇÕES
    if not os.path.exists(opt.path_to_save):
        os.makedirs(opt.path_to_save)

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    fileName = opt.name + "_" + dt_string

    filmaker(frames_com_deteccao, tempo_alcance,opt.path_to_save + "/" + fileName + "_com_deteccao.avi")

    filmaker(frames_originais,tempo_alcance,opt.path_to_save + "/" + fileName + "_sem_deteccao.avi")

    f = open(opt.path_to_save + "/" + fileName + ".json", "w+")
    dataToSave = {
        "id":
        opt.id,
        "Confianca":
        opt.conf_thresh,
        "limiar_IOU":
        opt.iou_thresh,
        "Kpx":
        k_x,
        "Kpy":
        k_y,
        "Kpz":
        k_z,
        "pos_inicial_x":
        wx.POSICAO_INICIAL_X,
        "pos_inicial_y":
        wx.POSICAO_INICIAL_Y,
        "pos_inicial_z":
        wx.POSICAO_INICIAL_Z,
        "frame_time":
        frameTime,
        "freq_widowX":
        wx.FREQ_MAX,
        "velocidade":
        wx.DELTA,
        "filtragemDistancia":
        flagDistanceFilter,
        "limiar_distancia_pixel":
        thresholdDistanceFilter,
        "numero_objetos":
        opt.num_objects,
        "tempo_alcance":
        tempo_alcance,
        "taxa_acerto_percentual_tracking":
        taxa_erro_percentual_frames,
        "taxa_acerto_percentual_deteccoes":
        taxa_erro_percentual_deteccoes,
        "quantidade_deteccoes_tracking":
        qtde_deteccoes,
        "quantidade_frames":
        qtde_frames,
        "objeto_de_interesse":
        opt.objeto_interesse,
        "vetor_classificacao_classes":
        str(predicao_classes),
        "taxa_acerto_deteccao":
        predicao_classes.count(opt.objeto_interesse) / len(predicao_classes),
        "qtde_predicoes_tensor":
        qtde_predicoes_tensor,
        "qtde_deteccoes_acc_temp_totais":
        qtde_deteccoes_acc_temp_totais,
        "qtde_deteccoes_acc_temp_validas":
        qtde_deteccoes_acc_temp_validas,
        "taxa_deteccoes_acc_temp":
        taxa_acc_temp,
        "tipo_deteccao":
        tipo_deteccao,
        "quantidade_tipo_deteccao":
        c_t,
        "percentagem_detec_sacada":
        (c_t["sacada"] if c_t["sacada"] else 0) / qtde_frames,
        "percentagem_detec_valida":
        (c_t["valida"] if c_t["valida"] else 0) / qtde_frames,
        "percentagem_detec_invalida":
        (c_t["invalida"] if c_t["invalida"] else 0) / qtde_frames,
        "tempo_sacadas":
        tempo_sacada,
        "porcentagem_tempo_sacada":
        sum(tempo_sacada) / tempo_alcance,
    }

    json.dump(dataToSave, f)

    # f.write("Parametros iniciais"\n")
    #f.write("Confiança: " + str(opt.conf_thresh) + "\n")
    #f.write("limiar IOU: " + str(opt.iou_thresh) + "\n")
    #f.write("Kpx: " + str(k_x) + " Kpy: " + str(k_y) + " Kpz: " + str(k_z) + "\n")
    #f.write("Posição inicial X: " + str(pos_atual_x) + "\n")
    #f.write("Posição inicial Y: " + str(pos_atual_y) + "\n")
    #f.write("Posição inicial Z: " + str(pos_atual_z) + "\n")

    f.close()

    # --------------------------------------
    input("Go Home -- press enter")
    wx.goSleep()
    
    udp.close()


def getCloserToCenter(det, ref):
    dists = []
    centroid = (0, 0)
    dist_center = 0
    for *coord, conf, cls in reversed(det):
        centroid = getCentroid(coord)
        dist_center, erro_x, erro_z = getError(centroid,ref)
        dists.append(dist_center)
    index_min = np.argmin(dists)
    return index_min, dist_center, centroid, erro_x, erro_z


def filmaker(imageVector,tempo, name="video.avi"):
    # Cria um vídeo no formato .avi juntando todos os frames.
    video_name = name
    images = imageVector
    height, width, layers = (128, 128, 3)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, int(len(images)/tempo), (width, height))
    for image in images:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


def getError(centroid,ref):
    imageDimensions = (128, 128)
    distanceToCenter = None
    x_distance = None
    y_distance = None
    if len(centroid) > 0:
      if ref == (0,0):
        distanceToCenter = math.sqrt(((centroid[0] - imageDimensions[1] / 2)**2) + ((centroid[1] - imageDimensions[0] / 2)**2))
      else:
        distanceToCenter = math.sqrt(((centroid[0] - ref[1])**2) + ((centroid[1] - ref[0])**2))
        
        x_distance = centroid[0] - (imageDimensions[1] / 2)
        y_distance = centroid[1] - (imageDimensions[0] / 2)
    return distanceToCenter, x_distance, y_distance


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def getCentroid(x):
    p1, p2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    width = int(x[2]) - int(x[0])
    # print(width)
    height = int(x[3]) - int(x[1])
    c1 = int(x[0] + width / 2)
    c2 = int(x[1] + height / 2)
    return (c1, c2)


def getDistanciaPontos(p1, p2):
    distancia = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return distancia


def letterbox(img,
              new_shape=(640, 640),
              color=(114, 114, 114),
              auto=True,
              scaleFill=False,
              scaleup=True,
              stride=32):
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
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img,
                             top,
                             bottom,
                             left,
                             right,
                             cv2.BORDER_CONSTANT,
                             value=color)  # add border
    return img, ratio, (dw, dh)


if __name__ == "__main__":
    global wx
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
    except:
        wx.goSleep()
