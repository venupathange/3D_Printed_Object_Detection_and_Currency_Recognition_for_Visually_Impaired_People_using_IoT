#on camera configuration
    # define a video capture object
import cv2
vid = cv2.VideoCapture(0)

ret, frame = vid.read()

cv2.imwrite('frame_test.jpg', frame)

# After the loop release the cap object
vid.release()
