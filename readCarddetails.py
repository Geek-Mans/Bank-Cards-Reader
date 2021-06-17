from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pytesseract as tes
import re
from PIL import Image
import os
import imquality.brisque as brisque
from skimage import io, img_as_float


#Using brisque scores to find quality of image
#but due to no proper availability of bank card images we can't create a relationship
def quality_check(loc):
    try:
        img= img_as_float(io.imread(loc, as_gray=True))
        score=brisque.score(img)
        print("Image Score = ", score)
    except:
        print("Exception in finding scores")
        
    


#Many Cards use OCR-A language (Matching with its template)
def template_match(image,req_str):
    try:
        # load the reference OCR-A image from disk, convert it to grayscale,
        # and threshold it, such that the digits appear as *white* on a
        # *black* background
        # and invert it, such that the digits appear as *white* on a *black*
        ref = cv2.imread("C:\\Users\\Man$\\Desktop\\Cards\\OCR.png")
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

        # find contours in the OCR-A image (i.e,. the outlines of the digits)
        # sort them from left to right, and initialize a dictionary to map
        # digit name to the ROI
        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        refCnts = imutils.grab_contours(refCnts)
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
        digits = {}

        # loop over the OCR-A reference contours
        for (i, c) in enumerate(refCnts):
            # compute the bounding box for the digit, extract it, and resize
            # it to a fixed size
            (x, y, w, h) = cv2.boundingRect(c)
            roi = ref[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))

            # update the digits dictionary, mapping the digit name to the ROI
            digits[i] = roi

        # initialize a rectangular (wider than it is tall) and square
        # structuring kernel
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # load the input image, resize it, and convert it to grayscale

        
        image = imutils.resize(image, width=300)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply a tophat (whitehat) morphological operator to find light
        # regions against a dark background (i.e., the credit card numbers)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

        # compute the Scharr gradient of the tophat image, then scale
        # the rest back into the range [0, 255]
        gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
            ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")

        # apply a closing operation using the rectangular kernel to help
        # cloes gaps in between credit card number digits, then apply
        # Otsu's thresholding method to binarize the image
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # apply a second closing operation to the binary image, again
        # to help close gaps between credit card number regions
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

        # find contours in the thresholded image, then initialize the
        # list of digit locations
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        locs = []

        # loop over the contours
        for (i, c) in enumerate(cnts):
            # compute the bounding box of the contour, then use the
            # bounding box coordinates to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            # since credit cards used a fixed size fonts with 4 groups
            # of 4 digits, we can prune potential contours based on the
            # aspect ratio
            if ar > 2.5 and ar < 4.0:
                # contours can further be pruned on minimum/maximum width
                # and height
                if (w > 40 and w < 55) and (h > 10 and h < 20):
                    # append the bounding box region of the digits group
                    # to our locations list
                    locs.append((x, y, w, h))

        # sort the digit locations from left-to-right, then initialize the
        # list of classified digits
        locs = sorted(locs, key=lambda x:x[0])
        output = []

        # loop over the 4 groupings of 4 digits
        for (i, (gX, gY, gW, gH)) in enumerate(locs):
            # initialize the list of group digits
            groupOutput = []

            # extract the group ROI of 4 digits from the grayscale image,
            # then apply thresholding to segment the digits from the
            # background of the credit card
            group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
            group = cv2.threshold(group, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # detect the contours of each individual digit in the group,
            # then sort the digit contours from left to right
            digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            digitCnts = imutils.grab_contours(digitCnts)
            digitCnts = contours.sort_contours(digitCnts,
                method="left-to-right")[0]

            # loop over the digit contours
            for c in digitCnts:
                # compute the bounding box of the individual digit, extract
                # the digit, and resize it to have the same fixed size as
                # the reference OCR-A images
                (x, y, w, h) = cv2.boundingRect(c)
                roi = group[y:y + h, x:x + w]
                roi = cv2.resize(roi, (57, 88))

                # initialize a list of template matching scores 
                scores = []

                # loop over the reference digit name and digit ROI
                for (digit, digitROI) in digits.items():
                    # apply correlation-based template matching, take the
                    # score, and update the scores list
                    result = cv2.matchTemplate(roi, digitROI,
                        cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)

                # the classification for the digit ROI will be the reference
                # digit name with the *largest* template matching score
                groupOutput.append(str(np.argmax(scores)))

            # draw the digit classifications around the group
            cv2.rectangle(image, (gX - 5, gY - 5),
                (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
            cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            # update the output digits list
            output.extend(groupOutput)

        # display the output credit card information to the screen
        #print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
        #tes.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'
        text = tes.image_to_string(image)
        card_number="".join(output)
        print("c = ",card_number)
        card_expiry = re.findall('\d{2}/\d{2}',text)
        if(req_str=="card_expiry")and len(card_expiry)>0:
            return (card_expiry[0])
        elif(req_str=="card_number")and len(card_number)>0:
            return(card_number)
        else:
            return None
    except:
        return None

#card_number
def find_c_no(text,image):
    card_number = re.findall('\d{4}\s\d{4}\s\d{4}\s\d{4}',text)
    #if not sending to template match
    if not card_number:
        card_number=template_match(image,"card_number")
        return card_number
    return card_number[0]

#card_expiry
def find_c_exp(text,image):
    card_expiry = re.findall('\d{2}/\d{2}',text)
    #if not sending to template match
    if not card_expiry:
        card_expiry=template_match(image,"card_expiry")
        return card_expiry
    return card_expiry[0]

#card_name
def find_c_name(text):
    after_expiry=re.split('\d{2}/\d{2}',text) #name exists after expiry
    card_name=[]
    if len(after_expiry)>1:
        index=len(after_expiry)-1
        possible_names=re.split('\n',after_expiry[index]) #cleaning after_expiry
        if possible_names:
            for i in possible_names:
                # if name is in format [fname midname lastname]
                if(re.search('[A-Za-z]+[.]*\s[A-Za-z]+[.]*\s[A-Za-z]+',i)):
                    card_name=re.findall('[A-Za-z]+[.]*\s[A-Za-z]+[.]*\s[A-Za-z]+',i)
                # if name is in format [fname lname]
                elif(re.search('[A-Za-z]+\s[A-Za-z]+',i)):
                    card_name=re.findall('[A-Za-z]+\s[A-Za-z]+',i)
        if card_name:
            return card_name[0]
    return None
    
    
            


#thresholding and removing noise with median blur along with normal OCR
def find_details(loc):
    image = cv2.imread(loc)
    #image = cv2.fastNlMeansDenoisingColored(image,None,20,10,7,21)
    cv2.imshow("Image",image)
    #tes.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'
    text = tes.image_to_string(Image.open(loc))
    
    #thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    filename_thresh = "C:\\Users\\Man$\\Desktop\\{}.png".format(os.getpid())
    cv2.imwrite(filename_thresh, gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text_thresh = tes.image_to_string(gray_thresh)
    

    gray_blur = cv2.medianBlur(gray, 3)
    #filename_blur = "C:\\Users\\Man$\\Desktop\\a.png".format(os.getpid())
    #cv2.imwrite(filename_blur, gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text_blur = tes.image_to_string(gray_blur)
    
    print("extract 1. ",text)
    print("extract 2. ",text_thresh)
    print("extract 3. ",text_blur)

    card_number= find_c_no(text,image)
    if not card_number:
        card_number= find_c_no(text_thresh,gray_thresh)
        if not card_number:
            card_number= find_c_no(text_blur,gray_blur)
    print("Card_Number: ",card_number)
    
    
    card_expiry= find_c_exp(text,image)
    if not card_expiry:
        card_expiry= find_c_exp(text_thresh,gray_thresh)
        if not card_expiry:
            card_expiry= find_c_exp(text_blur,gray_blur)
    print("Card_Expiry: ",card_expiry)
    
    card_name= find_c_name(text)
    if not card_name:
        card_name= find_c_name(text_thresh)
        if not card_expiry:
            card_name= find_c_name(text_blur)
    print("Card_Name: ",card_name)

    #os.remove(filename_thresh)
    #os.remove(filename_blur)




# driver
if __name__ == '__main__':
    loc="C:\\Users\\Man$\\Desktop\\Cards\\card2.png"
    quality_check(loc)
    find_details(loc)
