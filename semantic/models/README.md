# Pesi dei modelli (fuori da git)

`seg_infer_node` cerca `model_path` (default `yolo26n-sem.pt`) in questa cartella, installata in `share/semantic/models/` da `colcon build`.

| File | Modello | Classi |
|---|---|---|
| `yolo26n-sem.pt` | YOLO26n segmentazione semantica (Ultralytics) | 19 classi Cityscapes |

Copia qui il file `.pt` prima di compilare. I pesi non vanno committati (`.gitignore`).
