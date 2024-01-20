import numpy as np 
import cv2 

# Capturing video through webcam 
webcam = cv2.VideoCapture(0) 

def process_contours(frame, mask, color, text):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if area > 300: 
            x, y, w, h = cv2.boundingRect(contour) 
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) 
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color) 

while(1): 
    _, imageFrame = webcam.read() 
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 

    # Set range for red color and define mask 
    red_lower = np.array([136, 87, 111], np.uint8) 
    red_upper = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

    # Set range for green color and define mask 
    green_lower = np.array([25, 52, 72], np.uint8) 
    green_upper = np.array([102, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

    # Set range for blue color and define mask 
    blue_lower = np.array([94, 80, 2], np.uint8) 
    blue_upper = np.array([120, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

    # Set range for black color and define mask 
    black_lower = np.array([0, 0, 0], np.uint8) 
    black_upper = np.array([180, 255, 30], np.uint8) 
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper) 

    # Set range for white color and define mask 
    white_lower = np.array([0, 0, 200], np.uint8) 
    white_upper = np.array([180, 30, 255], np.uint8) 
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 

    # Morphological Transform, Dilation for each color and bitwise_and operator 
    # between imageFrame and mask determines to detect only that particular color 
    kernel = np.ones((5, 5), "uint8") 

    # For red color 
    red_mask = cv2.dilate(red_mask, kernel) 
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask) 

    # For green color 
    green_mask = cv2.dilate(green_mask, kernel) 
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask) 

    # For blue color 
    blue_mask = cv2.dilate(blue_mask, kernel) 
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask) 

    # For black color 
    black_mask = cv2.dilate(black_mask, kernel) 
    res_black = cv2.bitwise_and(imageFrame, imageFrame, mask=black_mask) 

    # For white color 
    white_mask = cv2.dilate(white_mask, kernel) 
    res_white = cv2.bitwise_and(imageFrame, imageFrame, mask=white_mask) 

    # Process contours for each color
    process_contours(imageFrame, red_mask, (0, 0, 255), "Red Colour")
    process_contours(imageFrame, green_mask, (0, 255, 0), "Green Colour")
    process_contours(imageFrame, blue_mask, (255, 0, 0), "Blue Colour")
    process_contours(imageFrame, black_mask, (0, 0, 0), "Black Colour")
    process_contours(imageFrame, white_mask, (255, 255, 255), "White Colour")

    # Display the result 
    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        webcam.release() 
        cv2.destroyAllWindows() 
        break
