import numpy as np
import cv2


def im_pyr_decomp(im, N):

    size_vec  = im.shape
    red_faktor = 2**N
    add_size = np.flip([int((np.ceil(el / red_faktor) * red_faktor - el)) for el in im.shape])
    [v, s] = im.shape
    opim = np.zeros((add_size[1] + v, add_size[0] + s))
    opim[0: v, 0: s] = im
    opim[v + 1: v + add_size[1], :] = opim[v:v-add_size[1]+1:-1, :]
    opim[:, s + 1: s + add_size[0]] = opim[:, s:s - add_size[0] + 1:-1]
    im = opim



    # Funkcija pravi Gausovu i Laplasovu piramidu od N nivoa od slike im
    GPyr = []
    LPyr = []
    for i in range(N):
        GPyr.append(im)
        g = cv2.pyrDown(im, borderType=cv2.BORDER_REPLICATE)
        g_up = cv2.pyrUp(g, cv2.BORDER_REPLICATE)


        l = im - g_up
        LPyr.append(l)
        im = g
    Res = im
    return LPyr, GPyr, Res, size_vec


def im_pyr_recon(LPyr, Res, size_vec):
    # Funkcija rekonstruise sliku na osnovu Laplasove piramide i reziduala

    # dubina razlaganja
    N = len(LPyr)
    for i in range(N, 0, -1):

        Res = cv2.pyrUp(Res, cv2.BORDER_REFLECT)+LPyr[i-1]

    Res = Res[0:size_vec[0], 0:size_vec[1]]
    im_rec = Res
    return im_rec