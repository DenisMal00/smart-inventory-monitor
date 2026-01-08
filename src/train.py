from ultralytics import YOLO
import torch
import os

def run_training(epochs=1, imgsz=640):
    # 1. Rilevamento Hardware (Ottimizzato per Mac M1/M2/M3)
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Utilizzo GPU Apple Silicon (MPS)")
    else:
        device = "cpu"
        print("🐢 GPU non trovata, utilizzo CPU")

    # 2. Caricamento Modello Pre-addestrato
    # Carichiamo 'yolov8n.pt' (Nano) per velocità o 'yolov8s.pt' (Small) per precisione
    model = YOLO('yolov8n.pt')

    # 3. Configurazione Percorsi
    # Assicurati che data.yaml sia nella cartella data/
    yaml_path = os.path.abspath("data/data.yaml")

    # 4. Avvio Addestramento
    print(f"🏋️ Inizio addestramento per {epochs} epoche...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,          # Se il Mac scalda troppo o crasha, scendi a 8
        device=device,
        project="models",  # Cartella dove verranno salvati i pesi
        name="inventory_monitor",
        exist_ok=True,     # Sovrascrive la cartella se esiste già
        plots=True         # Genera grafici di performance (utili per il portfolio)
    )

    print(f"✅ Addestramento completato! Risultati in: models/inventory_monitor")
    
    # 5. Export del modello per l'API (formato leggero)
    model.export(format="onnx")

if __name__ == "__main__":
    # --- TEST DI PROVA ---
    # Prima di dormire, lancia con 1 epoca per vedere se il sistema regge
    run_training(epochs=1, imgsz=320)