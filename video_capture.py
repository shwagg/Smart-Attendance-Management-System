import cv2

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        print("Capturing video failed")
        break


    cv2.imshow("Video", cap.read()[1])

    #para magstop yung program
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()