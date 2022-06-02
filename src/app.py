import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

filename = '../model/mnist_ML_model.sav'
canvas = np.ones((600, 600), dtype='uint8') * 255
canvas[100:500, 100:500] = 0

start_point = None
end_point = None
is_drawing = False
canvas_name = 'Test Canvas'


def draw_line(img, start_at, end_at):
    cv2.line(img, start_at, end_at, 255, 30)


def on_mouse_events(event, x, y, flags, params):
    global start_point
    global end_point
    global canvas
    global is_drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        if is_drawing:
            start_point = (x, y)

    # elif event == cv2.EVENT_

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            end_point = (x, y)
            draw_line(canvas, start_point, end_point)
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False


def clear_canvas():
    canvas[100:500, 100:500] = 0


cv2.namedWindow(canvas_name)
cv2.setMouseCallback(canvas_name, on_mouse_events)

# q: Quit
# p: make Prediction
# c: Clear canvas
print('Press  \'q\' to quit, \'p\' to make prediction...')

while True:
    cv2.imshow(canvas_name, canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # elif key == ord('s'):
    #     is_drawing = True
    # elif key == ord('t'):
    #     is_drawing = False
    elif key == ord('c'):
        clear_canvas()
    elif key == ord('p'):
        image = canvas[100:500, 100:500]

        # display digit that is sent to model, scaled
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        image = image / 255.0
        plt.imshow(image)
        plt.show()

        # resize image into array of 28*28
        image = np.reshape(image, (1, 784))

        # send to prediction model
        load_model = pickle.load(open(filename, 'rb'))
        result = load_model.predict(image)
        print('prediction: ', result[0])
        clear_canvas()

cv2.destroyAllWindows()
