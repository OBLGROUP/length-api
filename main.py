from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/estimate")
def estimate(req: ImageRequest):
    return {
        "received_url": req.image_url,
        "line_length_ratio": 2.22,
        "sign_aspect_ratio": 0.81,
        "line_length_px": 1734,
        "sign_width_px": 782,
        "sign_height_px": 634
    }
