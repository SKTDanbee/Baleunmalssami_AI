# main.py
# from typing import Union
from fastapi import FastAPI, UploadFile, File
import report_generate as report_generate
import shutil
from datetime import datetime

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Changed this endpoint to accept file uploads
@app.post("/generate-report/")
async def generate_report(file: UploadFile = File(...)):
    file_location = f"./{file.filename}"
    
    # Save the uploaded file to the local file system
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # Generate the report using the uploaded file
    report = report_generate.generate_report(file_location)
    
    return {
        "user_id": "test_user",
        "report": report,
        "timestamp": datetime.now().strftime("%Y-%m-%d")
        }
