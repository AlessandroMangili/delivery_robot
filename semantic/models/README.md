# Pesi dei modelli (fuori da git)

`seg_infer_node` cerca `model_path` in questa cartella (in `start_mapping.launch.py` si sceglie con `modello:=`, default `yolo26n-sem_eliza.pt`), installata in `share/semantic/models/` da `colcon build`.

| File | Modello | Classi |
|---|---|---|
| `yolo26n-sem.pt` | YOLO26n segmentazione semantica (Ultralytics), pesi Cityscapes originali | 19 classi Cityscapes |
| `yolo26n-sem_eliza.pt` | lo stesso modello dopo il fine-tuning su eliza del 13/10/2026 (319 fotogrammi Husky, AdamW, epoca 37); default di `start_mapping.launch.py` | 19 classi Cityscapes |

Copia qui il file `.pt` prima di compilare. I pesi non vanno committati (`.gitignore`).
