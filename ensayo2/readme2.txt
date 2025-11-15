🧪 Ensayo 2 – CNN ligera 4 bloques (16–32–32–64) con AdamW

En este segundo ensayo se realizó una reestructuración profunda de la arquitectura con el objetivo de ligerizar el modelo, mejorar la estabilidad del aprendizaje y conservar la capacidad de representación necesaria para reconocer letras del lenguaje de signos de forma robusta.

🔹 1. Arquitectura general

Se diseñó una red convolucional con 4 bloques de procesamiento:

16 → 32 → 32 → 64

Cada bloque incluye:

- Conv2d
- BatchNorm2d
- Activación Mish
- MaxPool2d
- Dropout2d moderado 

Después del cuerpo convolucional se aplica:
- Global Average Pooling (GAP)
- Dropout
- Una sola capa lineal (64 → num_classes)

Esto elimina por completo los clasificadores densos grandes y hace que toda la capacidad provenga de las convoluciones.

Beneficios:
- Mucho menos parámetros totales
- Mejor generalización
- Entrenamiento más estable
- Menor riesgo de sobreajuste

🔹 2. Capacidad convolucional

El uso de 4 bloques con un patrón progresivo
  16 → 32 → 32 → 64
permite extraer características visuales más profundas sin volver el modelo pesado.

El bloque doble de 32 canales mejora:

- estabilidad del gradiente
- sensibilidad a detalles finos en la mano
- precisión en gestos complejos

🔹 3. Función de activación: Mish

Se reemplazó ReLU por Mish, una activación suave y continua que:

- conserva información en valores negativos
- mejora la propagación del gradiente
- ayuda a modelos pequeños/medianos a converger mejor
- produce representaciones más ricas para visión

🔹 4. Regularización

El Ensayo 2 combina dos formas de regularización:
- Dropout2d(0.1) en convoluciones
- Dropout(0.3) en la capa final

Esto estabiliza el entrenamiento sin inhibir la capacidad de representación.

🔹 5. Clasificador final (GAP + Linear)

En lugar de múltiples capas densas, ahora se utiliza:
  AdaptiveAvgPool2d((1,1))
  Flatten
  Linear(64 → num_classes)

Ventajas:
- reducción drástica de parámetros
- mejor uso de la información convolucional
- red más rápida y más robusta

🔹 6. Optimizador: AdamW

Se adoptó AdamW con:

  lr = 0.001
  weight_decay = 1e-4

Beneficios:
- separa la regularización del gradiente
- mejora la estabilidad del entrenamiento
- alcanza mejor generalización

🔹 7. Transformaciones de entrenamiento

Se ajustó el esquema de data augmentation para que sea suave pero efectivo:
  
  Resize(96×96)
  RandomCrop(padding=4)
  RandomHorizontalFlip(0.5)
  RandomRotation(5°)
  ToTensor()
  Normalize(...)


Cambios clave:
- menor rotación (5°) para no deformar el gesto
- menor padding (4) para mantener la mano centrada
- se elimina ColorJitter para evitar ruido innecesario

🔹 8. Tamaño de imagen: 96×96

La resolución se redujo a 96×96, ofreciendo:
- entrenamiento más rápido
- menor memoria
- suficiente detalle para distinguir gestos
- menor tendencia al sobreajuste
