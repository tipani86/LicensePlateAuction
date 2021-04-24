import os
import numpy as np
import pandas as pd
import time as t
import re
import cv2
import gzip
import pickle
import pytesseract
from pynput.keyboard import Key, Controller
from datetime import datetime, timedelta
from queue import Queue
from threading import Thread
from sklearn.preprocessing import *
from sklearn.utils import shuffle
from train_sklearn import _prepare_dataset
from ocr import capture_screenshot, preprocess, preprocess_info_screen, ocr, settings

##### SETTINGS #####

predict_second = 47     # What second (11:29:XX) to make the prediction in validation testing

# Years to skip when importing data (to filter out data from different statistical distribution)
skip_years = []
# skip_years = [2014, 2015]

scaler = StandardScaler()

##########

keyboard = Controller()

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

if __name__ == "__main__":
    if os.path.isfile("model.pklz"):
        print("Loading model ...")
        with gzip.open("model.pklz", "rb") as fp:
            model = pickle.load(fp)
    else:
        print("Error: model.mdl not found!")
        quit()

    # Setting up basic info...

    # """
    year = input("Enter year: ")
    try:
        year = int(year)
        if year > int(datetime.now().strftime("%Y")):
            year = int(datetime.now().strftime("%Y"))
            print("Error in input, using current year value: {}".format(year))
    except:
        year = int(datetime.now().strftime("%Y"))
        print("Error in input, using current year value: {}".format(year))
    # """

    month = input("Enter month: ")
    try:
        month = int(month)
        if month < 1 or month > 12:
            month = int(datetime.now().strftime("%m"))
            print("Error in input, using current month value: {}".format(month))
    except:
        month = int(datetime.now().strftime("%m"))
        print("Error in input, using current month value: {}".format(month))

    # Loading historical data in order to tune the normalization algorithms

    print("Loading data ...")

    if len(skip_years) > 0:
        print("Note: Skipping year(s) {} from training data due to custom setting!".format(skip_years))

    data = pd.DataFrame()
    for row in pd.read_csv("data_csv.csv", sep=",", header=0, chunksize=1):

        if row.iloc[0][0] in skip_years:
            continue

        # Ignore month X for manual validation / simulation purposes, uncomment to train with full data
        ignore_year = year
        ignore_month = month
        if row.iloc[0][0] == ignore_year and row.iloc[0][1] == ignore_month:
            print("Ignoring the month of {}/{} in historical data.".format(ignore_year, ignore_month))
            continue

        # """
        for i in range(predict_second + 1, 60):
            row["11:29:{}".format(i)] = 0  # Removing data after 29:XX as to not let model overfit
        # """

        data = data.append(row)
    data = data.fillna(value=0).reset_index(drop=True)
    X_train, y_train = _prepare_dataset(data)
    scaler.fit(X_train)

    startprice = input("Enter minimum starting bid: ")
    try:
        startprice = int(startprice)
    except:
        startprice = 89500
        print("Error in input, using 2021 default minimum bid of 89500 CNY.")

    plates = 0  # Do not change, will be OCR'd automatically
    auctioners = 0  # Do not change, will be OCR'd automatically
    bid = 0  # Do not change, will be updated automatically

    print("")
    print("Ready to predict! Please run the Shanghai license plate auction site in the main window.")

    # Starting main prediction loop

    pred_df = pd.DataFrame(data=np.zeros((1, len(X_train.columns))), columns=list(X_train.columns))

    socket_sent = 0
    backup_sent = 0
    submit_price = 0

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

        # # Setting up preview for debug purposes
        # cv2.imshow('Plates processed', plates_screen)
        # cv2.imshow('Auctioners processed', auctioners_screen)
        # cv2.imshow('Preview: Info box processed', info_screen)

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
                                continue

        # print(info)

        ocrtoc = t.time()
        ocrtime = ocrtoc - ocrtic

        # Postprocessing output
        posttic = t.time()

        try:
            time = time[0]
        except:
            continue

        try:
            price = int(price[0])
        except:
            # print("Error setting price!")
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

        try:
            datetime_object = datetime.strptime(time, "%H:%M:%S")
        except:
            print("Error parsing time string: {}".format(time))
            continue

        # # This is to prevent more than one line being output per second
        # if datetime_object >= datetime.strptime("11:29:00", "%H:%M:%S"):
        #     if pred_df.at[0, time] != 0:
        #         continue

        # # If simulation mode is off, replace OCR'd time with system time because OCR might skip seconds
        # system_time = datetime.now()
        # system_time = system_time.strftime("%H:%M:%S")
        # if settings['plates_x_adjustment'] == 0:
        #     # time = system_time
        #     pass

        # For debugging purposes
        # print("Time: {}, price: {}, plates: {}, auctioners: {}".format(time, price, plates, auctioners))

        # Updating prediction info

        if time == 0 or price == 0 or plates == 0 or auctioners == 0:
            continue

        # """
        # Comment if only time series without additional data
        # pred_df.at[0, "month_ {}".format(month)] = 1
        # pred_df.at[0, "Year"] = year
        pred_df.at[0, "Plates"] = plates
        pred_df.at[0, "Auctioners"] = auctioners
        pred_df.at[0, "Success rate"] = np.round(plates / auctioners, 4)
        pred_df.at[0, "Startprice"] = startprice
        # """

        time_next = (datetime_object + timedelta(seconds=1)).strftime("%H:%M:%S")
        if datetime_object >= datetime.strptime("11:29:00", "%H:%M:%S") and datetime_object <= datetime.strptime("11:29:59", "%H:%M:%S"):
            pred_df.at[0, time] = price - startprice
            if datetime_object + timedelta(seconds=1) <= datetime.strptime("11:29:59", "%H:%M:%S"):
                pred_df.at[0, time_next] = price - startprice       # Autofill the next second just in case we jump a second

        """
        # Autofill previous second with current price in case there is a jump in seconds
        time_prev = (datetime_object - timedelta(seconds=1)).strftime("%H:%M:%S")
        if datetime_object > datetime.strptime("11:29:00", "%H:%M:%S") and price - startprice > 100:
            if pred_df.at[0, time_prev] == 0:
                pred_df.at[0, time_prev] = price-startprice
        """

        posttoc = t.time()
        posttime = posttoc - posttic

        # Predicting!
        predtic = t.time()

        pred = model.predict(scaler.transform(pred_df))
        pred = int(np.round((pred[0] + 150 + startprice) / 100, 0) * 100)  # Aim for the middle of the 300 RMB range and round to closest 100
        # pred = max(pred, pred_df["11:29:{}".format(predict_second)].to_numpy() + startprice + 300)
        pred = max(pred, price + 300)

        """
        if time == "11:29:37" or time == "11:29:38":
            if startprice + predicted - 300 > price + 300:
                bid = price + 300
            else:
                bid = startprice + predicted - 300
            print("Time: {}, current price: {}, current prediction: {}, bid: {}".format(time, price, startprice + predicted, bid))

        """

        predtoc = t.time()
        predtime = predtoc - predtic

        ##### CHANGE HERE IF PREDICTION TIME CHANGES #####

        if time == "11:29:45":  # Display backup strategy if actual prediction fails (45s + 1000)
            if backup_sent == 0:
                print("Time: {}, current price: {}, BACKUP pred: {}".format(time, price, price + 1000))
                backup_sent = 1
        if time == "11:29:{}".format(predict_second):
            bid = pred
            pred_df_debug = pred_df.copy()
            print("Time: {}, current price: {}, prediction: {}, BID!".format(time, price, bid))
            if socket_sent == 0:
                print("Attempting to send socket info...")
                # socketthread.start()
                send_string = str(bid)
                keyboard.type(send_string)
                socket_sent = 1
        elif bid == 0:
            print("Time: {}, current price: {}, prediction: {}".format(time, price, pred))
        else:
            if bid != 0 and bid - 300 <= price:
                print("Time: {}, current price: {}, SUBMIT!".format(time, price))
            else:
                print("Time: {}, current price: {}".format(time, price))
        # print("Current prediction: {}".format(startprice + predicted))

        if time == "11:29:59":
            # """
            # Pretty print final prediction dataframe for debugging and statistical purposes
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(pred_df.T)
            # """

            # """
            # Pretty print entire pandas prediction series for debugging purposes
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(pred_df_debug.T)
            # """

            break  # Stops the script loop when auction ends

        # # Tally of different processing times
        # print('Grab: {}s, Preprocess: {}s, OCR: {}s, Postprocess: {}, Predict: {}, Total: {}s ({}fps)'.format(
        #     np.round(grabtime,2),
        #     np.round(processtime,2),
        #     np.round(ocrtime,2),
        #     np.round(posttime,2),
        #     np.round(predtime,2),
        #     np.round(np.sum([grabtime, processtime, ocrtime, posttime, predtime]),2),
        #     np.round(1/np.sum([grabtime, processtime, ocrtime, posttime, predtime]),1)
        # ))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break