import os
import shutil
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    print("API KEY missing")
    exit()

rf = Roboflow(api_key=api_key)
project = rf.workspace("smartinventorymonitor").project("package-v2-aifao")

# Data download
print("Downloading...")
dataset = project.version(1).download("yolov8")

# Folder organization
if os.path.exists("data"):
    shutil.rmtree("data")

shutil.move(dataset.location, "data")
print("Dataset ready in data/")