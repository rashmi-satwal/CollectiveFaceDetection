import cv2

# loading a webcam image or video capture device

cap = cv2.VideoCapture(0)

#loading HAAR CASCADE Classfiers

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#looping

while True:
    ret, frame = cap.read()  # getting a frame, frame stores the img/numpy array, and ret checks if it working properly

    # converting the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # returns all the location in terms of positions -scalefactor
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # region of interest-location of our face

        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # look in to grayscale image for eye
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        # draw the obtained eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):  # ordinal returns the ascii value,
        break

# release the camera resources for some other device to use
cap.release()
cv2.destroyAllWindows()
