from app_dependency import *
from fastapi import FastAPI,File,UploadFile,Form
from typing import List,Optional
import uuid

app = FastAPI()
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), instruction: Optional[str] = Form("")):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        # Store uploaded file details
        file_paths = []
        for file in files:
            # Generate unique filename
            file_ext = file.filename.split('.')[-1]
            unique_filename = f"{uuid.uuid4()}.{file_ext}"
            file_path = os.path.join(upload_dir, unique_filename)

            # Save the file
            try:
                with open(file_path, "wb") as buffer:
                    # Read the uploaded file in chunks
                    contents = await file.read()
                    buffer.write(contents)
                    file_paths.append(file_path)
            except Exception as e:
                return {"error":"Unable to save file"}
        images = upload_to_gemini(file_paths)
        responses = describe(instruction, images)
        return responses
    except Exception as e:
        return {"error":str(e)}