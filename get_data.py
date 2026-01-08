import os
from roboflow import Roboflow
from dotenv import load_dotenv # Carica la libreria
import shutil

# Carica le variabili dal file .env
load_dotenv()

# Recupera la chiave dalla variabile d'ambiente
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("❌ Errore: ROBOFLOW_API_KEY non trovata nel file .env")

# Inizializza Roboflow usando la variabile
rf = Roboflow(api_key=api_key)

# ... il resto del codice rimane uguale ...
project = rf.workspace("abc-d9ezq").project("package-v2")
version = project.version(4)
dataset = version.download("yolov8")

# --- ORGANIZZAZIONE MLOPS ---
# Spostiamo tutto in 'data/' per coerenza con il resto del progetto
if os.path.exists("data"):
    shutil.rmtree("data")

# Roboflow crea una cartella chiamata 'Package-V2-4'
source_folder = "Package-V2-4" 

if os.path.exists(source_folder):
    shutil.move(source_folder, "data")
    print("✅ Dataset spostato con successo in 'data/'")
else:
    print(f"⚠️ Nota: Controlla se il nome della cartella scaricata è diverso da {source_folder}")

print("🚀 Pronto per il training!")