import json
import cv2
import numpy as np
from pyniryo import *
import matplotlib.pyplot as plt
import math

'''
# to create new blank dictionnaries in test mode, need to be commented in normal use
dict_color = {}
dict_shape = {}

tf = open("../data/Dictionnary database.json", "w")
json.dump((dict_color, dict_shape),tf)
tf.close()
'''

def update_dict(dict_color, dict_shape):
    # update the content of the two dict
    tf = open("../data/Dictionnary database.json", "w")
    json.dump((dict_color, dict_shape), tf)
    tf.close()

def moy_shape(nb_pixels_centroid, dict_shape, shape):
    # calculate the average value of the number of pixels linked to the chosen centroid
    moy_pixels = (nb_pixels_centroid[0] + nb_pixels_centroid[1] + nb_pixels_centroid[2]) / 3
    
    # add the corresponding value in the dict for the new shape
    dict_shape[shape] = moy_pixels

def moy_color(rgb_code, dict_color, color):
    # calculate the average value of RGB value from each image
    r_moy = (rgb_code[0][0] + rgb_code[1][0] + rgb_code[2][0]) / 3
    g_moy = (rgb_code[0][1] + rgb_code[1][1] + rgb_code[2][1]) / 3
    b_moy = (rgb_code[0][2] + rgb_code[1][2] + rgb_code[2][2]) / 3
    
    # add the corresponding value in the dict for the new color
    rgb_code = [r_moy, g_moy, b_moy]
    dict_color[color] = rgb_code
    
def add_shape(total_pixels_centroid, nb_pixels_centroid):
    # add the data of the color in a list
    total_pixels_centroid.append(nb_pixels_centroid)

def add_color(rgb_code, color, centers, pos):   
    # add the data of the shape in a list
    r = int(centers[pos][0])
    g = int(centers[pos][1])
    b = int(centers[pos][2])
    rgb_code.append([r, g, b])

def k_means(image):
    # calculate the color and the shape of the object on an image
    
    # convert image into rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    k = 4 # white, grey, black and the color of the object
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)
    print(centers)
    
    # detect which centroid correspond to the color of the object thanks to standard deviation method
    sigma_max = 0
    for each in range (len(centers)):
        temp = np.std(centers[each])
        if temp > sigma_max:
            sigma_max = temp
            pos = each
    
    # count the number of pixels attached to the centroid of the object color
    nb_pixels_centroid = int((labels == pos).sum())
    
    return centers, nb_pixels_centroid, pos

def take_workspace_img(client):
    # take an image from the detected workspace
    mtx, dist = client.get_camera_intrinsics()
    while 1:
        img_compressed = client.get_img_compressed()
        img_raw = uncompress_image(img_compressed)
        img = undistort_image(img_raw, mtx, dist)
        img_work = extract_img_workspace(img, workspace_ratio=1)
        if img_work is not None:
            break
    return img_work

def shape_choice(dict_shape):
    # ask to the user the shape of the color
    shape = input("Quelle est la forme de l'objet que vous souhaitez ajouter à la base de données ? ")
    new_shape = "Y"
    for key_shape in dict_shape:
        if key_shape == shape:
            print("La forme", shape, "a déjà été apprise avec le nombre de pixels suivants : ", dict_shape[shape])
            new_shape = input("Souhaitez-vous enregistrer un nouveau nombre de pixels associé à cette forme ? [Y/N] ")
            
    return shape, new_shape

def color_choice(dict_color):
    # ask to the user the color of the object
    color = input("De quelle couleur est l'objet que vous souhaitez ajouter à la base de données ? ")
    new_color = "Y"
    for key_color in dict_color:
        if key_color == color:
            print("La couleur", color, "a déjà été apprise avec le code RGB suivant : ", dict_color[color])
            new_color = input("Souhaitez-vous enregistrer un nouveau code RGB associé à cette couleur ? [Y/N] ")
            
    return color, new_color
    
def main_new_objects(dict_color, dict_shape, client):
        # main loop for the creation of a new object
        color, new_color = color_choice(dict_color)
        shape, new_shape = shape_choice(dict_shape)
        rgb_code = []
        total_pixels_centroid = []
        
        for compt_pictures in range (0,3):
            print("Placer l'objet dans le workplace à une position et dans un sens aléatoire")
            print("Encore", 3 - compt_pictures, "restantes")
            state = input("Presser la touche 'Enter' pour lancer l'analyse de l'image : ")
            if state == "":
                if new_color == "Y" or new_shape == "Y":
                    image = take_workspace_img(client)
                    centers, nb_pixels_centroid, pos = k_means(image)
                if new_color == "Y":
                    add_color(rgb_code, color, centers, pos)
                if new_shape == "Y":
                    add_shape(total_pixels_centroid, nb_pixels_centroid)
        if new_color == "Y":
            moy_color(rgb_code, dict_color, color)
            print("Couleur ajoutée !")
        if new_shape == "Y":
            moy_shape(total_pixels_centroid, dict_shape, shape)
            print("Forme ajoutée !")
        update_dict(dict_color, dict_shape)