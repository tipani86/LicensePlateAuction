import numpy as np
import cv2
import pytesseract
import random
import re
from mss import mss
from PIL import Image
from queue import Queue
from threading import Thread
import time as t
from pynput.keyboard import Key, Controller

keyboard = Controller()

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

plates, auctioners = 0, 0
method = "flash"
method = "html"
browser = "firefox-half"
socket_sent = 0

settings = {
    'x_adjustment': 0,
    'y_adjustment': 0,
}

print("##### Settings #####")
print("Browser: {}".format(browser))
print("Method: {}".format(method))
print("")

if browser == "firefox":
    settings['x_adjustment'] = 0
    settings['y_adjustment'] = 0

if browser == "firefox-half":
    settings['x_adjustment'] = -360
    settings['y_adjustment'] = 0

if method == "flash":
    settings['method'] = method
    settings['standard_height'] = 18

    settings['y_adjustment'] = 4
    settings['plates_x_adjustment'] = 0  # 0 normally, 14 if using simulation

    settings['plates_x'] = 621 + settings['plates_x_adjustment']
    settings['plates_y'] = 358 + settings['y_adjustment']
    settings['plates_width'] = 40

    settings['auctioners_x'] = 650
    settings['auctioners_y'] = 374 + settings['y_adjustment']
    settings['auctioners_width'] = 50

    settings['info_x'] = 540
    settings['info_y'] = 450
    settings['info_width'] = 350
    settings['info_height'] = 190

if method == "html":
    settings['method'] = method
    settings['crop'] = [
        (546 + settings['x_adjustment'], 481 + settings['y_adjustment']),
        (663 + settings['x_adjustment'], 698 + settings['y_adjustment'])
    ]
    settings['standard_height'] = 25
    settings['plates_x_adjustment'] = 0

    settings['plates_x'] = 0 + settings['plates_x_adjustment']
    settings['plates_y'] = 0
    settings['plates_width'] = 70

    settings['auctioners_x'] = 36
    settings['auctioners_y'] = 24
    settings['auctioners_width'] = 70

    settings['time_x'] = 19
    settings['time_y'] = 167
    settings['time_width'] = 75

    settings['price_x'] = 54
    settings['price_y'] = 192
    settings['price_width'] = 60

    settings['info_x'] = 17
    settings['info_y'] = 167
    settings['info_width'] = 100
    settings['info_height'] = 50

def capture_screenshot():
    # Capture entire screen
    with mss() as sct:
        if settings['method'] == "html":
            monitor = {
                "left": settings['crop'][0][0],
                "top": settings['crop'][0][1],
                "width": settings['crop'][1][0] - settings['crop'][0][0],
                "height": settings['crop'][1][1] - settings['crop'][0][1],
            }
        else:
            monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        # Convert to PIL/Pillow Image
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                             # Grayscaling
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)    # Doubling size
    img = cv2.medianBlur(img, 3)
    return img

def preprocess_info_screen(img, method):
    # Filtering RED texts only
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([100, 10, 10])
    if method == "html":
        lower_red = np.array([110, 140, 150])
    upper_red = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask=mask)
    res[np.where((res == [0, 0, 0]).all(axis=2))] = [255, 255, 255]     # Flipping fully black pixels back to white to help OCR

    img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)                         # Grayscaling image
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Zooming a bit
    img = cv2.medianBlur(img, 3)
    #cv2.imshow('image', img)
    #cv2.imshow('hsv', hsv)
    #cv2.imshow('mask', mask)
    return img

def ocr(img, queue):
    try:
        output = pytesseract.image_to_string(Image.fromarray(img)).replace(' ', '').splitlines()
    except:
        output = 0
    #print(output)
    queue.put(output)

if __name__ == "__main__":
    while True:
        # Loading image, dynamic

        grabtic = t.time()
        image = capture_screenshot()
        grabtoc = t.time()
        grabtime = grabtoc - grabtic

        # Preprocessing image

        processtic = t.time()

        plates_screen = np.array(image.crop((
            settings['plates_x'],
            settings['plates_y'],
            settings['plates_x'] + settings['plates_width'],
            settings['plates_y'] + settings['standard_height']
        )))
        plates_screen = preprocess(plates_screen)

        auctioners_screen = np.array(image.crop((
            settings['auctioners_x'],
            settings['auctioners_y'],
            settings['auctioners_x'] + settings['auctioners_width'],
            settings['auctioners_y'] + settings['standard_height']
        )))
        auctioners_screen = preprocess(auctioners_screen)

        if settings['method'] == "html":
            time_screen = np.array(image.crop((
                settings['time_x'],
                settings['time_y'],
                settings['time_x'] + settings['time_width'],
                settings['time_y'] + settings['standard_height']
            )))
            time_screen = preprocess(time_screen)
            price_screen = np.array(image.crop((
                settings['price_x'],
                settings['price_y'],
                settings['price_x'] + settings['price_width'],
                settings['price_y'] + settings['standard_height']
            )))
            price_screen = preprocess(price_screen)
        else:
            info_screen = np.array(image.crop((
                settings['info_x'],
                settings['info_y'],
                settings['info_x'] + settings['info_width'],
                settings['info_y'] + settings['info_height']
            )))
            info_screen = preprocess_info_screen(info_screen, settings['method'])

        processtoc = t.time()
        processtime = processtoc - processtic

        # Setting up preview for debug purposes

        cv2.imshow('Plates processed', plates_screen)
        cv2.imshow('Auctioners processed', auctioners_screen)
        if settings['method'] == "html":
            cv2.imshow('Time processed', time_screen)
            cv2.imshow('Price processed', price_screen)
        else:
            cv2.imshow('Preview: Info box processed', info_screen)

        # Performing OCR
        ocrtic = t.time()

        if settings['method'] == "html":
            timequeue = Queue()
            timethread = Thread(target=ocr, args=(time_screen, timequeue))
            timethread.start()
            pricequeue = Queue()
            pricethread = Thread(target=ocr, args=(price_screen, pricequeue))
            pricethread.start()
        else:
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

        if settings['method'] == "html":
            time = timethread.join()
            time = timequeue.get()
            price = pricethread.join()
            price = pricequeue.get()
        else:
            info = infothread.join()
            info = infoqueue.get()

            time, price, indirect_price = 0, None, 0

            if info != 0:
                for line in info:
                    # print(line)
                    line = line.replace("l", "1").replace("I", "1")
                    line = re.sub("\D", "", line)
                    # print(line)
                    # print("")
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
                                pass

        # print(info)

        ocrtoc = t.time()
        ocrtime = ocrtoc - ocrtic

        # Postprocessing output
        posttic = t.time()

        try:
            price = int(price[0])
        except:
            print("Error setting price!")
            price = 0

        try:
            time = time[0]
        except:
            pass

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

        # # Testing autosubmit
        # if time == "11:23:30" and socket_sent == 0:
        #     print("Attempting to send socket info...")
        #     send_string = str(price)
        #     keyboard.type(send_string)
        #     socket_sent = 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if time == "11:29:59":
            break