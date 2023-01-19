import numpy as np
import json
import cv2
from pyniryo import *
import math
import matplotlib.pyplot as plt

import speech_recognition as sr
import pyaudio
import wave
import base64

from new_objects import take_workspace_img, k_means

def threshold_hls(img, list_min_hsv, list_max_hsv):
    frame_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return cv2.inRange(frame_hsl, tuple(list_min_hsv), tuple(list_max_hsv))

def fill_holes(img):
    # fill holes in a mask
    im_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img = img | im_floodfill_inv
    return img

def objs_mask(img):
    # calculate a mask
    color_hls = [[0, 0, 0], [180, 150, 255]]

    mask = threshold_hls(img, *color_hls)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # erode workspace markers
    mask[:15, :] = cv2.erode(mask[:15, :], kernel7, iterations=5)
    mask[-15:, :] = cv2.erode(mask[-15:, :], kernel7, iterations=5)
    mask[:, :15] = cv2.erode(mask[:, :15], kernel7, iterations=5)
    mask[:, -15:] = cv2.erode(mask[:, -15:], kernel7, iterations=5)

    mask = fill_holes(mask)

    mask = cv2.dilate(mask, kernel3, iterations=1)
    mask = cv2.erode(mask, kernel5, iterations=1)
    mask = cv2.dilate(mask, kernel11, iterations=1)

    mask = fill_holes(mask)

    mask = cv2.erode(mask, kernel7, iterations=1)

    return mask

def extract_objs(mask):
    # calculate the coordonate of the detected object on the workspace
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if cnts is not None:
        for cnt in cnts:
            cx, cy = get_contour_barycenter(cnt)
            try:
                angle = (get_contour_angle(cnt)*180)/math.pi
            except NiryoRobotException:
                angle = 0
            
    return cx, cy, angle

def test_color(dict_color, texte, centers, pos):
    # check the link between recorded data from training and data from actual picture for the color
    color_detect = False
    error_color = False
    for i_color in dict_color.keys():
        if i_color in texte:
            color_detect = True
            sum_diff = 0
            for j_color in range(len(centers[pos])):
                sum_diff = sum_diff + abs(dict_color[i_color][j_color] - centers[pos][j_color])
            if abs(sum_diff) < 100: # gap until 100 is accepted
                print("Couleur détectée !")
            else:
                print("Pas d'objet de couleur", i_color, "sur le workplace !")
                error_color = True
    if color_detect == False:
        print("Couleur non-détectée dans la base de données !")
        error_color = True
    return error_color
        
def test_shape(dict_shape, texte, nb_pixels_centroid):
    # check the link between recorded data from training and data from actual picture for the shape
    shape_detect = False
    error_shape = False
    for i_shape in dict_shape.keys():
        if i_shape in texte:
            shape_detect = True
            if abs(nb_pixels_centroid - dict_shape[i_shape]) < 200: # gap until 200 pixels is accepted
                print("Forme détectée !")
            else:
                print("Pas d'objet de forme", i_shape, "sur le workplace !")
                error_shape = True
    if shape_detect == False:
        print("Forme non-détectée dans la base de données !")
        error_shape = True
    return error_shape

def text_read():
    # collect the audio speech of the user and transcrit it into text
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 3200
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "voc.wav"
    MIC_INDEX = 11

    audio = pyaudio.PyAudio()

    # Start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS, input_device_index=MIC_INDEX,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print ("Donner la désignation de l'objet que vous souhaitez attraper !\n")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)

    print ("Enregistrement terminé, analyse en cours...\nPréparez-vous au déplacement de l'objet.")

    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    rec_vocale = sr.Recognizer()
    fichier = "../data/voc.wav", "fr-FR"

    with sr.AudioFile(fichier[0]) as src: # open file
        audio = rec_vocale.record(src)
        texte = rec_vocale.recognize_google(audio, language=fichier[1]) # translate speech into text in French
        print(texte)
    
    return texte

def main_detection(dict_color, dict_shape, client, workspace, drop_pose, observation_pose):
    # main loop for object detection and object pick and place
    image = take_workspace_img(client)
    centers, nb_pixels_centroid, pos = k_means(image)
    texte = text_read()
    #texte = "cercle vert" # in case of test whithout using vocal option
    mask = objs_mask(image)
    plt.imshow(mask) # show the result of the object detection
    plt.show()
    x, y, angle = extract_objs(mask)
    error_color = test_color(dict_color, texte, centers, pos)
    error_shape = test_shape(dict_shape, texte, nb_pixels_centroid)
    if error_color == False and error_shape == False: # object must not be picked if the wrong object is on the workspace
        z_offset = 0.01 # offset for the vacuum pump
        obj_ = client.get_target_pose_from_rel(workspace, z_offset, (x / 200), (y / 200),
                                                      angle)
        print("Position objet : ", obj_)
        client.pick_from_pose(obj_) # take the object
        client.place_from_pose(*drop_pose.to_list()) # place it in the chosen drop zone
        #client.close_gripper() # in case of use of the gripper instead of vacuum pump
        client.move_pose(*observation_pose.to_list()) # robot is again in observation pose for a new round