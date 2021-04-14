import numpy as np
import cv2
import pytesseract
import random
import re
from PIL import Image, ImageGrab
from queue import Queue
from threading import Thread
import time as t
from pynput.keyboard import Key, Controller

keyboard = Controller()

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                             # Grayscaling
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)    # Doubling size
    img = cv2.medianBlur(img, 3)
    return img

def preprocess_info_screen(img, method):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Zooming a bit

    # Filtering RED texts only
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([100, 10, 10])
    if method == "html":
        lower_red = np.array([115, 0, 0])
    upper_red = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask=mask)

    res[np.where((res == [0, 0, 0]).all(axis=2))] = [255, 255, 255]     # Flipping fully black pixels back to white to help OCR
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)                         # Grayscaling image
    res = cv2.medianBlur(res, 3)
    #cv2.imshow('image', img)
    #cv2.imshow('hsv', hsv)
    #cv2.imshow('mask', mask)
    #cv2.imshow('res', res)
    return res

def ocr(img, queue):
    try:
        output = pytesseract.image_to_string(Image.fromarray(img)).replace(' ', '').splitlines()
    except:
        output = 0
    #print(output)
    queue.put(output)

plates, auctioners = 0, 0
method = "flash"
method = "html"
socket_sent = 0

if method == "flash":
    standard_height = 18

    y_adjustment = 4
    plates_x_adjustment = 0  # 0 normally, 14 if using simulation

    plates_x = 621 + plates_x_adjustment
    plates_y = 358 + y_adjustment
    plates_width = 40

    auctioners_x = 650
    auctioners_y = 374 + y_adjustment
    auctioners_width = 50

    info_x = 540
    info_y = 450
    info_width = 350
    info_height = 190

if method == "html":
    standard_height = 25
    x_adjustment = 0
    y_adjustment = -25
    plates_x_adjustment = 0  # 0 normally, 14 if using simulation

    plates_x = 546 + plates_x_adjustment + x_adjustment
    plates_y = 508 + y_adjustment
    plates_width = 70

    auctioners_x = 582 + x_adjustment
    auctioners_y = 532 + y_adjustment
    auctioners_width = 70

    info_x = 560 + x_adjustment
    info_y = 675 + y_adjustment
    info_width = 275
    info_height = 100

if __name__ == "__main__":
    while True:
        # Loading image, dynamic

        grabtic = t.time()
        image = ImageGrab.grab()
        grabtoc = t.time()
        grabtime = grabtoc - grabtic

        # Preprocessing image

        processtic = t.time()

        plates_screen = np.array(image.crop((plates_x, plates_y, plates_x + plates_width, plates_y + standard_height)))
        plates_screen = preprocess(plates_screen)

        auctioners_screen = np.array(image.crop((auctioners_x, auctioners_y, auctioners_x + auctioners_width, auctioners_y + standard_height)))
        auctioners_screen = preprocess(auctioners_screen)

        info_screen = np.array(image.crop((info_x, info_y, info_x + info_width, info_y + info_height)))
        info_screen = preprocess_info_screen(info_screen, method)

        processtoc = t.time()
        processtime = processtoc - processtic

        # Setting up preview for debug purposes

        cv2.imshow('Plates processed', plates_screen)
        cv2.imshow('Auctioners processed', auctioners_screen)
        cv2.imshow('Preview: Info box processed', info_screen)

        # Performing OCR

        ocrtic = t.time()

        infoqueue = Queue()
        infothread = Thread(target=ocr, args=(info_screen, infoqueue))
        infothread.start()

        if plates == 0:
            platesqueue = Queue()
            platesthread = Thread(target=ocr, args=(plates_screen, platesqueue))
            platesthread.start()

        if auctioners == 0:
            auctionersqueue = Queue()
            auctionersthread = Thread(target=ocr, args=(auctioners_screen, auctionersqueue))
            auctionersthread.start()

        if plates == 0:
            platesocr = platesthread.join()
            platesocr = platesqueue.get()

        if auctioners == 0:
            auctionersocr = auctionersthread.join()
            auctionersocr = auctionersqueue.get()

        info = infothread.join()
        info = infoqueue.get()

        #print(info)

        ocrtoc = t.time()
        ocrtime = ocrtoc - ocrtic

        # Postprocessing info screen output

        time, price, indirect_price = 0, None, 0

        if info != 0:
            for line in info:
                #print(line)
                line = line.replace("l", "1")
                line = re.sub("\D", "", line)
                #print(line)
                #print("")
                if len(line) == 6:
                    time = line[:2] + ":" + line[2:4] + ":" + line[4:]
                elif len(line) == 5:
                    price = line
                else:
                    if line[:5].isnumeric() == True and line[-5:].isnumeric() == True:
                        low = int(line[:5])
                        high = int(line[-5:])
                        if high - low == 600:
                            price = (high + low) / 2
                        else:
                            continue

            try:
                price = int(price)
            except:
                print("Error setting price!")
                price = 0

        if plates == 0:
            try:
                plates = int(platesocr[0])
            except:
                plates = 0

        if auctioners == 0:
            try:
                auctioners = int(auctionersocr[0])
            except:
                auctioners = 0

        #print('Reading: {}'.format(path))
        print("Plates: {}, Auctioners: {}, Time: {}, Current price: {}".format(plates, auctioners, time, price))
        print('Grab: {}s, Processing: {}s, OCR: {}s, Total: {}s ({}fps)'.format(np.round(grabtime,2), np.round(processtime,2), np.round(ocrtime,2), np.round(grabtime + processtime + ocrtime,2), np.round(1/(grabtime + processtime + ocrtime),1)))
        print('')

        # Testing autosubmit
        if time == "11:23:30" and socket_sent == 0:
            print("Attempting to send socket info...")
            send_string = str(price)
            keyboard.type(send_string)
            socket_sent = 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


        if time == "11:29:59":
            break