from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import numpy as np
from skimage.morphology import skeletonize
import requests
import math

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

def measure_line_length_px(skel):
    sk = skel.astype(np.uint8)
    h, w = sk.shape
    total = 0.0

    neighbours = [
        (-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
        (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),
        (1,-1,math.sqrt(2)),(1,1,math.sqrt(2))
    ]

    ys, xs = np.where(sk > 0)

    for y,x in zip(ys,xs):
        for dy,dx,wgt in neighbours:
            ny,nx = y+dy,x+dx
            if 0 <= ny < h and 0 <= nx < w and sk[ny,nx] > 0:
                total += wgt

    return total/2.0

@app.post("/estimate")
def estimate(req: ImageRequest):

    resp = requests.get(req.image_url)
    img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    _,binary = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_area = 100
    mask = np.zeros_like(binary)

    for i in range(1,num_labels):
        area = stats[i,cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == i] = 255

    ys,xs = np.where(mask>0)

    x1,x2 = xs.min(), xs.max()
    y1,y2 = ys.min(), ys.max()

    sign_width_px = x2-x1
    sign_height_px = y2-y1

    crop_mask = mask[y1:y2+1, x1:x2+1]

    skeleton = skeletonize(crop_mask>0)

    line_length_px = measure_line_length_px(skeleton)

    line_length_ratio = line_length_px / sign_width_px
    sign_aspect_ratio = sign_height_px / sign_width_px

    return {
        "line_length_ratio": line_length_ratio,
        "sign_aspect_ratio": sign_aspect_ratio,
        "line_length_px": line_length_px,
        "sign_width_px": sign_width_px,
        "sign_height_px": sign_height_px
    }
