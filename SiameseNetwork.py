import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import numpy as np
from math import ceil
from keras_preprocessing.image import img_to_array
DOWNSCALE_FACTOR = 3
img = cv2.imread("./image.png")
img = img_to_array(img)
img = np.reshape(img, (-1, 720, 1280, 3))
# Model to get encodings of the image
model = Sequential()
model.add(Conv2D(64, (3, 3), strides=2))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(.2))
model.add(Conv2D(128, (2, 2)))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="sigmoid"))
pred = model.predict(img)
cap = cv2.VideoCapture(0)

def calculate_similarity_score(arr1, arr2):
    if len(arr1) != len(arr2):
        raise Exception(f"You passed two arrays that do not have the same length. Length of first array: {len(arr1)}. Length of second array: {len(arr2)}")
    else:
        loss = 0.0
        for id in range(len(arr1[0])):
            loss += (arr2[0][id] - arr1[0][id])**2
        return ceil(loss)




while True:
    ret, frame = cap.read()
    f = np.reshape(frame, (-1, 720, 1280, 3))
    frame_pred = model.predict(f)
    score = calculate_similarity_score(pred, frame_pred) - DOWNSCALE_FACTOR
    s = "No Face Detected..."
    if score < 20:
        s = "Welcome, Joshua!"
    frame = cv2.putText(frame, s, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 1)
    cv2.imshow("jgiug",frame)

    print(score, s)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()
