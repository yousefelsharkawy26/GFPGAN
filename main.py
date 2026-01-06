from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import shutil
import os
import cv2
import numpy as np
import torch
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import io

app = FastAPI()

# Global variables to hold models
restorer = None
bg_upsampler = None

def download_models():
    """Download necessary models if they don't exist."""
    print("Checking models...")
    os.makedirs('experiments/pretrained_models', exist_ok=True)
    os.makedirs('gfpgan/weights', exist_ok=True)

    # Model URLs
    v1_4_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    v1_4_path = 'experiments/pretrained_models/GFPGANv1.4.pth'

    # Check and download main model
    if not os.path.exists(v1_4_path):
        if not os.path.exists('gfpgan/weights/GFPGANv1.4.pth'):
            print(f"Downloading GFPGANv1.4 model...")
            os.system(f'wget {v1_4_url} -O {v1_4_path}')
        else:
            # Copy from local weights if exists
            shutil.copy('gfpgan/weights/GFPGANv1.4.pth', v1_4_path)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global restorer, bg_upsampler

    download_models()

    # 1. Setup Background Upsampler (RealESRGAN)
    if torch.cuda.is_available():
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True)
    else:
        # CPU mode - disable background upsampler or use lighter version?
        # For compatibility, we'll disable it or warn user.
        print("CUDA not available. Background upsampler might be slow or disabled.")
        bg_upsampler = None

    # 2. Setup GFPGAN Restorer (v1.4 by default)
    model_path = 'experiments/pretrained_models/GFPGANv1.4.pth'

    # Fallback to weights dir if not found in experiments
    if not os.path.exists(model_path):
         model_path = 'gfpgan/weights/GFPGANv1.4.pth'

    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=bg_upsampler
    )
    print("Models loaded successfully.")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    version: str = Form("v1.4"),
    scale: float = Form(2.0),
    weight: float = Form(0.5)
):
    """
    Endpoint to enhance an image.
    """
    global restorer

    if not restorer:
        raise HTTPException(status_code=500, detail="Model not initialized")

    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Adapt options if necessary (switching versions dynamically is expensive,
    # for now we stick to v1.4 or would need to reload model)
    # If strictly needed, we can expand logic to reload 'restorer' based on 'version' param.

    try:
        # Inference
        # has_aligned=False, only_center_face=False, paste_back=True
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=weight
        )

        # Resize if scale != 2
        if scale != 2:
            h, w = restored_img.shape[0:2]
            interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
            restored_img = cv2.resize(restored_img, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)

        # Encode back to image
        res, im_png = cv2.imencode(".png", restored_img)
        return Response(content=im_png.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "online", "model": "GFPGAN v1.4"}
