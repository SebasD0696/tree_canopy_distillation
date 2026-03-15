Arquitectura


tree_canopy_distillation/
│
├─ models/                   ← aquí están todos tus modelos
│   ├─ teacher_segformer.py
│   ├─ mae_teacher.py
│   ├─ student_segformer.py
│   ├─ mae_encoder.py
│   └─ __init__.py
│
├─ cross_scale_mae/           ← repositorio MAE descargado
│   └─ ... (todos los archivos del Cross-Scale MAE)
│
├─ training/                 ← scripts de entrenamiento / experimentos
│   ├─ train_distillation.py  ← este script
│   └─ ... otros scripts de prueba / utils
│
├─ data/                 ← tus imágenes, máscaras, grids, etc.
│
├─ outputs/                  ← checkpoints, logs, etc.
│
└─ utils/                    ← funciones auxiliares (metrics, data utils, etc.)
