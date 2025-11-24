                                                    ============================================
                                                          ✦Comparación General – Ensayos✦       
                                                    ============================================

➤Este repositorio contiene varios ensayos de redes neuronales convolucionales para clasificar el alfabeto ASL. 
➤Cada ensayo incluye su propio modelo, configuración y resultados, permitiendo comparar cómo cambian el rendimiento y la estabilidad al modificar arquitectura, activación, regularización y optimizador.

🟧Se muestra un resumen de los ensayos ordenados de mejor a peor rendimiento

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

💠Ensayo Preentrenado - EfficientNet-B0

-Transfer learning desde ImageNet.
-Solo se ajustan los últimos bloques y el clasificador.
-Mayor precisión y mejor generalización.

💠Ensayo 1 - Arquitectura Optimizada

-CNN 16→32→64→128.
-BatchNorm, ReLU, Dropout y ReduceLROnPlateau.
-Modelo equilibrado y estable sin preentrenado.

💠Ensayo 2 - CNN Ligera

-Arquitectura 16→32→32→64.
-Mish + Dropout2D + AdamW.
-Rápida, eficiente y con muy buen rendimiento para su tamaño.

💠Ensayo 4 - CNN Profunda y Estrecha

-Filtros mínimos: 1→2→4→8→16.
-GELU + Adamax.
-Explora el límite inferior de capacidad; mejora respecto al Ensayo 3.

💠Ensayo 3 — CNN Simple 

-Solo 2 bloques convolucionales: 32→64.
-SiLU + RMSprop.
-Modelo base usado como referencia.
