import cv2
import sys

import numpy as np

img = np.random.rand(500, 500, 3)

#img[:,:,0] = np.ones([5,5])*64/255.0
#img[:,:,1] = np.ones([5,5])*128/255.0
#img[:,:,2] = np.ones([5,5])*192/255.0

cv2.imshow("image", img)

img = np.random.rand(500, 500, 3)

cv2.imshow("image", img)

while True:
    key_id = cv2.waitKey(100)

    if key_id == ord('z'):
        img = np.random.rand(500, 500, 3)
        cv2.imshow("image", img)

    if key_id == ord('q'):
        img = np.random.rand(500, 500, 3)
        cv2.imshow("image", img)

    if key_id == ord('s'):
        img = np.random.rand(500, 500, 3)
        cv2.imshow("image", img)

    if key_id == ord('d'):
        img = np.random.rand(500, 500, 3)
        cv2.imshow("image", img)

    if key_id == 27:
        cv2.destroyAllWindows()
        sys.exit(0)
