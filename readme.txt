                                          ============================================
                                                ✦Comparación General – Ensayos✦       
                                          ============================================

➤Este repositorio contiene varios ensayos de redes neuronales convolucionales para clasificar el alfabeto ASL. 
➤Cada ensayo incluye su propio modelo, configuración y resultados, permitiendo comparar cómo cambian el rendimiento y la estabilidad al modificar arquitectura, activación, regularización y optimizador.

💠Se muestra un resumen de los ensayos ordenados de mejor a peor rendimiento

-----------------------------------------------------------------------------------------------------------------------------

🚀Ensayo Preentrenado - EfficientNet-B0

-Transfer learning desde ImageNet.
-Solo se ajustan los últimos bloques(7 y 8).
-El clasificador final sustituido por una capa lineal de 29 clases.
-Excelente equilibrio entre rendimiento y coste computacional (~5.3M parámetros).
-Aumentos de datos avanzados: RandomCrop, Rotation, Flip, ColorJitter.
-Optimización con AdamW + CrossEntropyLoss.
-Mayor precisión y capacidad de generalización obtenida.

🏋️‍♂️Ensayo 1 - Arquitectura Optimizada

-CNN progresiva 16→32→64→128.
-LazyConv2D + BatchNorm + ReLU + MaxPooling.
-Dropout2D + Dropout para regularización.
-Clasificador denso: 1024 → 256 → salida (con BatchNorm y ReLU).
-Aumentos completos: Resize 128, Crop, Flip, Rotación ±15°, ColorJitter.
-Entrenamiento estable: Adam (wd=1e-4) + ReduceLROnPlateau.
-Modelo final equilibrado y estable sin preentrenado.

⚡️Ensayo 2 - CNN Ligera y Equilibrada

-Activación Mish para mejorar suavidad del gradiente.
-Dropout2D (0.1) en los bloques y Dropout (0.3) en el clasificador.
-AdaptiveAvgPool2d(1×1) para reducir parámetros.
-Optimización con AdamW (lr=1e-3).
-Aumentos geométricos suaves (Crop, Flip, Rotación).
-Rápida, eficiente y con muy buen rendimiento para su tamaño.

📉Ensayo 3 - CNN Profunda y Estrecha

-Activación GELU, ideal para redes profundas con canales reducidos.
-Dropout (0.2) en el clasificador.
-AdaptiveAvgPool2d(1×1).
-Optimización con Adamax (lr=1e-3).
-Aumentos más completos: Crop, Flip, Rotación.
-Usa CosineAnnealingLR para una reducción suave del LR.
-Explora el límite inferior de capacidad con buena estabilidad; mejora respecto al Ensayo 4.

☠️Ensayo 4 — CNN Muy Simple 

-Solo 2 bloques convolucionales: 32→64.
-Activación SiLU, BatchNorm y MaxPool.
-Dropout moderado en el clasificador.
-AdaptiveAvgPool2d(2×2).
-Optimización con RMSprop (lr=5e-4).
-Aumentos mínimos: Flip horizontal y normalización.
-Utiliza LinearLR para un calentamiento progresivo del learning rate.
-Modelo base usado como referencia.
