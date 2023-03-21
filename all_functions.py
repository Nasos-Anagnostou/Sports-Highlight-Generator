# author: Nasos Anagnostou
# Diploma Thesis "Semantic event analysis in sports Video using webcast Text"
# pytesseract and easyOCR engine used to parse timetags for each frame detected
# latest update 7/11/22

# ALL THE IMPORTS NEEDED
import filepaths
import pytesseract
import easyocr
import cv2
import numpy as np
import pandas as pd
import glob
import re
import ntpath
import math
import imutils
import requests
from PIL import Image, ImageEnhance
from pandasgui import show
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
from statistics import mean
from Obj_Det_AI import detect_custom_object


##################################################  OCR FUNCTIONS  ####################################################
# creating method to sort image files in folder based in frame number low to high
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# tesseract ocr function
def tess_dir(ocr_path):
    # tesseract allocation
    pytesseract.pytesseract.tesseract_cmd = r"E:\programs\tessaract\tesseract.exe"
    # tesseract configure
    configure = r'--oem 0 --psm 6'

    # initialise vars
    counter_1 = 0
    counter_2 = 0
    alltimetags = []
    timetags = []
    failrec = []

    # loop for every frame in the dir
    for filename in sorted(glob.glob(ocr_path + '/*.png'), key=numericalSort):

        # get the filename not the whole path
        ftail = ntpath.split(filename)[1]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ALL THIS IS THE PREPROCESSING PIPELINE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # input frame in opencv and convert color
        im_cv2 = cv2.imread(filename)
        im_cv2 = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB)

        # import frame to PIM
        pim = Image.fromarray(im_cv2)
        # pim = Image.open(filename)             #show image

        # 1. enhance image sharpness given a specific factor
        enhancer = ImageEnhance.Sharpness(pim)
        factor_sharp = 2
        pim_en = enhancer.enhance(factor_sharp)

        # 2. enhance image contrast given a specific factor
        enhancer_2 = ImageEnhance.Contrast(pim_en)
        factor_contr = 2  # 2.5 the best value for oem -0
        pim_en2 = enhancer.enhance(factor_contr)

        # CV input from PIL and Convert RGB to BGR
        img_pim = np.array(pim_en)
        img_pim = img_pim[:, :, ::-1].copy()

        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Convert to Gray
        grimg = cv2.cvtColor(img_pim, cv2.COLOR_BGR2GRAY)

        # 3. thresholding image chose binary thresholding since it gives the best results( analusi kata to grapsimo )
        ret, thr_img = cv2.threshold(grimg, 120, 255, cv2.THRESH_BINARY)

        # 4. resize image x1.5 its original size
        (origW, origH) = pim.size
        big_img = cv2.resize(thr_img, (int(1.5 * origW), int(1.5 * origH)), interpolation=cv2.INTER_LINEAR)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ALL THIS IS THE PREPROCESSING PIPELINE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # use pytesseract for OCR
        g = pytesseract.image_to_string(big_img, config=configure, lang="eng")
        # create a list of ocred strings
        gs = g.split()
        # print OCR results for every frame
        print("The frame has this elements: ", ftail, gs)

        # loop through every string parsed from ocr list
        counter_1 += 1
        success_flag = False
        for i, item in enumerate(gs):

            if not success_flag:

                if re.fullmatch(filepaths.time_pat2, item):
                    alltimetags.append([item, ftail])

                    # timetag parsing success inform
                    print("This is a match!", item)

                    # replace commas with dots to increase ocr accuracy
                    gs[i] = item.replace(',', '.')

                    # replace time tags when in under a minute to match play by play format
                    if re.fullmatch(filepaths.under_minute_format, gs[i]):
                        third = gs[i].split('.')[0]

                        if len(third) == 2:
                            gs[i] = "0:" + third
                        else:
                            gs[i] = "0:0" + third

                    else:
                        gs[i] = item.replace(';', ':')

                    # getting frame_id
                    z = re.findall('([0-9]+)', ftail)[0]
                    # creating a list with timetag ocred + frame that was found on
                    timetags.append([gs[i], z])

                    # update if found timetag in specific frame
                    success_flag = True
                    counter_2 += 1

        if not success_flag:
            # creating a list with frames that failed to give a timetag
            failrec.append(ftail)

    # calculating the succes_rate for all frames OCRed
    success_rate = (counter_2 / counter_1) * 100

    # Printing stuff related to ocr success
    print("\n Timetags exported are: ", timetags)
    print("\nYou found {} out of {} images successfully.".format(counter_2, counter_1))
    print("\n Success rate of:", success_rate, "%")
    #print("\n These images failed :", failrec)


    return timetags, alltimetags

# easyOcr ocr function
def easyOcr_dir(ocr_path):
    # EasyOcr Reader initialisation
    reader = easyocr.Reader(['en'], gpu=False)

    alltimetags = []
    ttags = []
    failrec = []
    counter_1 = 0
    counter_2 = 0

    # loop for every frame in the dir
    for filename in sorted(glob.glob(ocr_path + '/*.png'),key=numericalSort):
        counter_1 += 1
        ftail = ntpath.split(filename)[1]
        # get the results

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ALL THIS IS THE PREPROCESSING PIPELINE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        im_cv2 = cv2.imread(filename)
        pim = Image.fromarray(im_cv2)

        # 1. enhance image sharpness given a specific factor
        enhancer = ImageEnhance.Sharpness(pim)
        factor_sharp = 2
        pim_en = enhancer.enhance(factor_sharp)
        # CV input from PIL and Convert RGB to BGR
        img_pim = np.array(pim_en)
        img_pim = img_pim[:, :, ::-1].copy()

        # 2. Convert to Gray
        grimg = cv2.cvtColor(img_pim, cv2.COLOR_BGR2GRAY)

        # 3. thresholding image chose binary thresholding since it gives the best results( analusi kata to grapsimo )
        ret, thr_img = cv2.threshold(grimg, 120, 255, cv2.THRESH_BINARY)

        # 4. resize image x1.5 its original size
        (origW, origH) = pim.size
        big_img = cv2.resize(thr_img, (int(1.5 * origW), int(1.5 * origH)), interpolation=cv2.INTER_LINEAR)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ALL THIS IS THE PREPROCESSING PIPELINE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # read the ocr items and check if the list is empty
        result = reader.readtext(big_img)
        if not result:
            continue

        quarter = result[0][1]
        result = result[1][1]
        alltimetags.append([result, quarter, ftail])
        print("The frame has these elements: ", ftail, result, quarter)

        if re.fullmatch(filepaths.time_pat2, result):

            if re.fullmatch('(\S*((1)|(1s|S|5))\S*)', quarter):
                quarter = "1st Quarter"

            elif re.fullmatch('(\S*(2|Z)\S*)|(\S*(ND)\S*)', quarter):
                quarter = "2nd Quarter"

            elif re.fullmatch('(\S*((J|J|3)|(RD|rd))\S*)', quarter):
                quarter = "3rd Quarter"

            elif re.fullmatch('(\S*((4TH|4|TH)|(AT|At))\S*)', quarter):
                quarter = "4th Quarter"

            # replace commas with dots to increase ocr accuracy
            result = result.replace(',', '.')

            # dirty fix
            if not re.fullmatch(filepaths.under_minute_format, result):
                result = result.replace('.', ':').replace(';', ':')

            else:
                # replace time tags when in under a minute to match play by play format
                third = result.split('.')[0]
                if len(third) == 2:
                    result = "0:" + third
                else:
                    result = "0:0" + third

            print("Its a match: ", result, quarter)
            # getting frame_id
            z = re.findall('([0-9]+)', ftail)[0]
            # creating a list with timetag ocred + frame that was found on
            ttags.append([result, quarter, z])
            counter_2 += 1

        else:
            failrec.append(ftail)

    # calculating the success_rate for all frames OCRed
    success_rate = (counter_2 / counter_1) * 100

    # Printing stuff related to ocr success
    print("\n These images failed :", failrec)
    print("\n Timetags exported are: ", ttags)
    print("\n You found {} out of {} images successfully.".format(counter_2, counter_1))
    print("\n Success rate of:", success_rate, "%")

    return ttags, alltimetags


#############################################  OBJECT DETECTION FUNCTIONS  ############################################
# Template matching
def match_scl(vin_file, ocr_path, tmp_img, start_minute, end_minute):

    # remove the old temp images
    for f in os.listdir(ocr_path):
        os.remove(os.path.join(ocr_path, f))

    if start_minute == 'start' and end_minute == 'end':
        print("Get the whole video of '%s' game" % vin_file[68: -4])
        trim_vid = vin_file
    else:
        print("\nCreating a clipped video of the '%s' match game video" % vin_file[68: -4])
        trim_vid = filepaths.trim_vid_eu + vin_file[68: -4] + '_trimmed.mp4'
        ffmpeg_extract_subclip(vin_file, 60 * start_minute, 60 * end_minute, targetname=trim_vid)


    # read first frame from input video
    cap = cv2.VideoCapture(trim_vid)
    # check if video stream is open
    if not cap.isOpened():
        print("Error opening video  file")

    # get framerate of the video and total frames
    myfps = cap.get(5)
    print("\nFrame rate of this video is: ", myfps)

    # obj detect with nn model
    #tmp_img_nn = detect_custom_object(frame)

    # Template image , edw mporw na valw to output tou automation
    template = cv2.imread(tmp_img)  # load the template image             AN THELW  NA XRHSIMOPOIHSW OBJ DET vazw temp_img_nn
    gray_tem = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # convert it to grayscale

    # loop through every frame read by input
    findings = []  # bookkeeping variable to keep track of the matched region
    ret = True
    first_time = True
    while ret:

        # read frame id
        frameid = cap.get(1)
        ret, frame = cap.read()
        # print('read a new frame:', ret)


        # take 2 frames per every second, one each 500msec
        if ret & ((frameid % math.floor(myfps) == 0) | (frameid % math.floor(myfps) == math.ceil(myfps / 2))):

            # cv2.imwrite(pathOut + 'frame%d.jpg' % frameid, frame)

            # convert frame image to grayscale and then apply canny edge transformation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = cv2.Canny(gray, 50, 200)
            (iH, iW) = image.shape[:2]

            found = None
            # The mechanism to identify the scorebox in game -check if this is the first frame we detect the template image
            if first_time:

                # threshold n1 for the first_time
                threshold = 0.5
                for scale in np.linspace(0.4, 1, 20)[::-1]:         #image pyramid psaksou gia documentation

                    # resize the template according to the scale, and keep track of the ratio of the resizing
                    resized = imutils.resize(gray_tem, width=int(template.shape[1] * scale))
                    # if the resized template is bigger than the image, then break from the loop
                    if resized.shape[0] > iH or resized.shape[1] > iW:
                        break

                    # detect edges in the resized, grayscale template
                    # and apply template matching to find the template in the image
                    edged = cv2.Canny(resized, 50, 200)
                    # for full boxscore template use: ccoef normed / timebox use: ccoor normed
                    result = cv2.matchTemplate(image, edged, cv2.TM_CCORR_NORMED)
                    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                    # if we find a new maximum correlation value, then update the bookkeeping variable
                    if found is None or maxVal > found[0]:
                        found = (maxVal, maxLoc)
                        tH, tW = resized.shape[:2]

            else:
                # The generic template matching, repeat the same process as above
                edged = cv2.Canny(new_tem, 50, 200)
                result = cv2.matchTemplate(image, edged, cv2.TM_CCORR_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                found = (maxVal, maxLoc)
                tH, tW = new_tem.shape[:2]

            # unpack the bookkeeping variable and compute the (x, y) coordinates of the bounding box based on the resized ratio
            (maxVal, maxLoc) = found

            if maxVal >= threshold:
                # we just found the first mathcing image so we use it as the template from now on!
                first_time = False
                # threshold n2 is higher because we use the boxscore of the same game now
                threshold = 0.65

                (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
                (endX, endY) = (int(maxLoc[0] + tW), int(maxLoc[1] + tH))

                # cropping the part we want to ocr
                crop_img = frame[startY:endY, startX:endX]
                new_tem = crop_img
                cv2.imwrite(ocr_path + "frame_%d.png" % frameid, crop_img)
                # cv2.imshow("cropped", crop_img)
                # cv2.waitKey(0)
                #print(found, frameid)
                findings.append(found[0])

            # else:
            #     print("Didnt make it")

    # close capture
    cap.release()

    # quality control for max false positive - min true negative detected so we adjust threshold
    mean_findings = sum(findings) / len(findings)

    print("\nQuality Control:")
    print("Mean score", mean_findings)
    print("Max value", max(findings))
    print("Min value", min(findings))

    return myfps

# get the template image with deep learning object detect
def template_finder(tmpl_path, trim_vid):
    # read first frame from input video
    cap = cv2.VideoCapture(trim_vid)
    # check if video stream is open
    if not cap.isOpened():
        print("Error opening video  file")

    myfps = cap.get(5)
    ret = True
    while ret:
        # obj detect with nn model
        frameid = cap.get(1)
        ret, frame = cap.read()
        if ret & (frameid % math.floor(myfps) == 0):

            template_nn, flag = detect_custom_object(frame)
            if flag:
                break

    template = cv2.imread(template_nn)  # load the template image
    print("Found in frame: ", frameid)

    template_path = os.path.join(tmpl_path, "template_eu.jpg")
    cv2.imwrite(template_path, template)


#############################################  VIDEO EDITING FUNCTIONS  ###############################################
# create the Highlight video clip
def clip_creator(trim_vid, myttag, ttaglist, myfps):
    # delete the old clip
    if os.path.exists(filepaths.clip_1):
        os.remove(filepaths.clip_1)
        print("Deleting the old file")

        # videolcip init
    videoclip = 0
    # flag to check if the video created
    vflag = False
    for item in ttaglist:

        if (myttag[0] in item[0]) and (myttag[1] in item[1]):
            fr_id = float(item[2])
            print("\nFound timestamp: {0} in frame_id: {1}".format(item[0], fr_id))

            # Clip creation creating subclip with duration [mysec-4, mysec+2]  #vrisko to sec thelo [mysec-6, mysec+2] h [fr_id -(fps* 6), fr_id +(fps* 2)]
            mysec = fr_id / myfps
            ffmpeg_extract_subclip(trim_vid, mysec - 6, mysec + 1, targetname=filepaths.clip_1)
            videoclip = filepaths.clip_1
            vflag = True
            break

        elif myttag[0] in item[0] and myttag[1] not in item[2]:
            print("\nWe dont have the quarter")

        else:
            print("we dont have this highlight yet")


    # # Play the video clip created
    # cap = cv2.VideoCapture(videoclip)
    # fps = int(cap.get(cv2.CAP_PROP_FPS))   # or cap.get(5)
    #
    # if not cap.isOpened():
    #     print("Error File Not Found")
    #
    # # setting playback for video clip created
    # while cap.isOpened():
    #     ret,frame= cap.read()
    #
    #     if ret:
    #         time.sleep(1/fps)
    #         cv2.imshow('Highlight!', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
    #
    # # close capture
    # cap.release()
    # cv2.destroyAllWindows()
    return vflag, videoclip


#############################################  WEB SCRAPPER FUNCTION  #################################################
# web scrapper to get the webcast text files of the game you want
def eur_scrapper(game_name):
    ######################### seasoncode=? kai gamecode=? apo to site ths euroleague
    #game_name = 'game1.csv'
    url = "https://live.euroleague.net/api/PlaybyPlay?gamecode=263&seasoncode=E2020"
    r = requests.get(url)
    print(r.status_code)  # if all goes well this must return 200
    data = r.json()

    ############### AUTO GIA OLA TA QUARTERS
    # 1ST QUARTER
    df1Q = pd.DataFrame.from_dict(pd.json_normalize(data['FirstQuarter']), orient='columns')
    df1Q = df1Q[["MARKERTIME", "PLAYER", "PLAYINFO"]]
    df1Q.insert(loc=0, column='Quarter', value='1st Quarter')

    # 2ND QUARTER
    df2Q = pd.DataFrame.from_dict(pd.json_normalize(data['SecondQuarter']), orient='columns')
    df2Q = df1Q[["MARKERTIME", "PLAYER", "PLAYINFO"]]
    df2Q.insert(loc=0, column='Quarter', value='2nd Quarter')

    # 3RD QUARTER
    df3Q = pd.DataFrame.from_dict(pd.json_normalize(data['ThirdQuarter']), orient='columns')
    df3Q = df1Q[["MARKERTIME", "PLAYER", "PLAYINFO"]]
    df3Q.insert(loc=0, column='Quarter', value='3nd Quarter')

    # 4TH QUARTER
    df4Q = pd.DataFrame.from_dict(pd.json_normalize(data['ForthQuarter']), orient='columns')
    df4Q = df1Q[["MARKERTIME", "PLAYER", "PLAYINFO"]]
    df4Q.insert(loc=0, column='Quarter', value='4th Quarter')

    frames = (df1Q, df2Q, df3Q, df4Q)
    result = pd.concat(frames)
    result.to_csv(os.path.join(r"E:\Career files\Degree Thesis\2. Dataset\competition_paths\csv_paths\csv_eur", game_name))


#############################################  CSV EDITING FUNCTIONS  #################################################
# csv file editor obsolete since the frontend is developed
def csv_editor (filename):

	# root path of csv files
	mypath = r"E:/Career files/Degree Thesis/2. Dataset/play by play text"

	# read csv file and create dataframe
	df = pd.read_csv(filename) #,index_col ="event_id")
	rows,cols  = df.shape

	#rename columns
	df.columns = ['Quarter', 'Clock time', 'Score', 'Event']

	#Create event_ids and place them as first column
	event_ids = list(range(1,rows+1))
	df.insert(loc=0, column='Event_Id', value=event_ids)
	print(df.columns)

	# display to user the events of play by play text to choose which event to watch
	show(df)

	# ask user to choose which event_id he wants to watch
	myevid = int(input("Give me the event_id you want to watch:"))  # EDW THELEI ENAN ELEGXO TIMIS
	# future use of the quarter for now not in use
	myquart = "2nd Quarter"

	# create a filter for the specific event id, quarter(not used currently)
	filt_1 = (df['Event_Id'] == myevid)
	filt_2 = (df['Quarter'] == myquart)

	# apply filter to get timetag, quarter(not used
	myttag = df.loc[filt_1, 'Clock time']
	myevent = df.loc[filt_2]  # , 'Clock time']

	# convert timetag,quarter to string
	myevent = myevent.to_string(index=False).strip()
	myttag = myttag.to_string(index=False).strip()

	df.to_csv(mypath+"/sample1.csv", index=False)

	return myttag