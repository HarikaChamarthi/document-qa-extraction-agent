
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
import cv2
import pytesseract


# Create the FastAPI application
app = FastAPI(title="AI Engineering Practice API")


# -----------------------------
# Pydantic User schema
# -----------------------------
class User(BaseModel):
    name: str
    age: int


# -----------------------------
# Temporary in-memory user data
# -----------------------------
users = [
    {
        "id": 1,
        "name": "Harika",
        "age": 22
    }
]


# -----------------------------
# GET /users
# Get all users
# -----------------------------
@app.get("/users")
def get_users():
    return users


# -----------------------------
# POST /users
# Create a new user
# -----------------------------
@app.post("/users", status_code=201)
def add_user(user: User):
    new_user = {
        "id": len(users) + 1,
        "name": user.name,
        "age": user.age
    }

    users.append(new_user)
    return new_user


# -----------------------------
# GET /hello
# -----------------------------
@app.get("/hello")
def hello():
    return {"message": "Hello Harika"}


# -----------------------------
# GET /inspect
# Inspect incoming HTTP request
# -----------------------------
@app.get("/inspect")
async def inspect_request(request: Request):
    return {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query_parameters": dict(request.query_params)
    }


# -----------------------------
# POST /user
# Validate incoming user data
# -----------------------------
@app.post("/user")
def create_user(user: User):
    return {
        "message": "User received successfully",
        "name": user.name,
        "age": user.age
    }


# -----------------------------
# POST /upload
# Upload image and extract text using OCR
# -----------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Read uploaded file as raw bytes
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # Decode uploaded bytes into an OpenCV image
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Reject invalid or unsupported images
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please upload a valid PNG or JPEG."
        )

    # Convert color image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu thresholding to create a binary image
    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Extract all recognized text
    text = pytesseract.image_to_string(threshold)

    # Extract recognized words, positions, and confidence scores
    ocr_data = pytesseract.image_to_data(
        threshold,
        output_type=pytesseract.Output.DICT
    )

    words = []

    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()

        try:
            confidence = float(ocr_data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1.0

        # Keep only non-empty words
        if word:
            words.append({
                "text": word,
                "confidence": confidence,
                "x": ocr_data["left"][i],
                "y": ocr_data["top"][i],
                "width": ocr_data["width"][i],
                "height": ocr_data["height"][i]
            })

    # Return OCR output and image metadata
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "byte_size": len(file_bytes),
        "color_shape": list(image.shape),
        "gray_shape": list(gray.shape),
        "threshold_shape": list(threshold.shape),
        "color_dtype": str(image.dtype),
        "gray_dtype": str(gray.dtype),
        "threshold_dtype": str(threshold.dtype),
        "ocr_text": text,
        "ocr_words": words
    }
