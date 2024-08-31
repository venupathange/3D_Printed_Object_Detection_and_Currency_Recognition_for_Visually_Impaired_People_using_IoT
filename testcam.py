# import the opencv library
import cv2
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
ret, frame = vid.read()

cv2.imwrite('frame.jpg', frame)
  
# After the loop release the cap object
vid.release()
