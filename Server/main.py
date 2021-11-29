import cv2, math
import numpy as np
from PIL import ImageFile

from Config import *
from Socket import ImageSocket, DataSocket
from ImageProcessing import *
from Logger import get_logger

logger = get_logger(__name__)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm
from scipy.spatial.transform import Rotation

def neither_none(*args):
    ret = True
    for obj in args:
        ret = ret and (not obj is None)
    return ret

if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    img_sock = ImageSocket(SOCKET_PORT)
    data_sock = DataSocket(SOCKET_PORT+1)

    img_sock.start()
    data_sock.start()

    save_img = False
    save_img_idx = 0
    os.makedirs(f'{BASE_DIR}/save-img', exist_ok=True)
    while os.path.isfile(f'{BASE_DIR}/save-img/{save_img_idx:05d}.png'):
        save_img_idx += 1

    last_pivot = (None, None, None)
    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        if (zipped := data_sock.recv()):
            if zipped[0] == DataSocket.TEXT:
                logger.info(('TEXT', zipped))
                if zipped[1] == 'save':
                    save_img = True
                elif zipped[1] == 'exit':
                    img_sock.send('q')
                    data_sock.send('q')
            elif zipped[0] == DataSocket.SENSOR:
                _, x,y,z, a,b,c, n,m,k, u,v,w = zipped
                logger.info(('SENSOR', x,y,z, a,b,c, n,m,k, u,v,w))
            elif zipped[0] == DataSocket.DEST:
                dest = zipped[1]
                logger.info(('DEST', dest))

        jpg_file = img_sock.recv()
        if not jpg_file:
            continue
    
        img, kp, des = compute_image(jpg_file)
        h,w,_ = img.shape
        if DEBUG: cv2.imshow('img', img)
        if neither_none(*last_pivot, kp, des):
            t_img, t_kp, t_des = last_pivot
            matches = match_images(des, t_des)

            H, R, T = estimate_perspective(matches, kp, t_kp, (w,h), np.array([0, 0, 1]))
            vec = np.dot(R, np.array([0, 0, 1])) + T
            nvec = normalize(vec)
            data_sock.send(nvec.tolist())

        if save_img:
            last_pivot = (img, kp, des)
            save_img = False

            cv2.imwrite(f'{BASE_DIR}/save-img/{save_img_idx:05d}.png', img)
            save_img_idx += 1
    
    img_sock.send('q')
    data_sock.send('q')
    