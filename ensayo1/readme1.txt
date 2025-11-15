🧪 Ensayo 1 – Arquitectura base del modelo CNN

En este primer ensayo se diseñó una arquitectura convolucional base enfocada en establecer una línea de referencia para los siguientes experimentos.
El modelo combina capas convolucionales con normalización, activaciones ReLU y un bloque denso final para la clasificación.

🔹 1. Estructura general de la red

La red se compone de tres bloques convolucionales seguidos de un bloque totalmente conectado (fully connected).
Cada bloque convolucional aplica la siguiente secuencia:

  Convolución → Batch Normalization → ReLU → MaxPooling

Esto permite:
- Extraer características espaciales relevantes.
- Normalizar la activación de cada lote, acelerando el entrenamiento.
- Reducir progresivamente el tamaño espacial de las imágenes, concentrando la información.
- El uso de nn.MaxPool2d(kernel_size=2, stride=2) reduce a la mitad las dimensiones después de cada convolución, facilitando un aprendizaje jerárquico de patrones.

🔹 2. Función de activación

- Usada: nn.ReLU()

Justificación:
ReLU (Rectified Linear Unit) es una función de activación ampliamente utilizada por su simplicidad y eficiencia.
Presenta las siguientes ventajas:
- Reduce el problema del gradiente desapareciente.
- Acelera la convergencia.
- Introduce no linealidad sin aumentar demasiado el costo computacional.
Sin embargo, puede presentar el problema del “dying ReLU”, en el que ciertas neuronas dejan de activarse si sus pesos se saturan en valores negativos.

🔹 3. Capa de clasificación (Fully Connected Block)

El modelo original utiliza tres capas densas consecutivas con normalización por lotes y dropout, con la estructura:
  fc1 → BatchNorm → ReLU → Dropout  
  fc2 → BatchNorm → ReLU → Dropout  
  fc3 → Clasificación final

Este bloque permite al modelo:
- Combinar las características extraídas por las convoluciones.
- Aprender relaciones no lineales entre los mapas de activación.
- Realizar la predicción final para las num_classes categorías.
- El uso de BatchNorm1d y Dropout(0.3) reduce el sobreajuste y estabiliza el aprendizaje, a costa de un mayor número de parámetros.

🔹 4. Optimizador

- Usado: optim.Adam(lr=0.001, weight_decay=1e-4)

Justificación:
Adam combina las ventajas de AdaGrad y RMSProp, ajustando dinámicamente la tasa de aprendizaje por parámetro.
Es un optimizador eficiente y ampliamente utilizado en redes profundas debido a su rápida convergencia y estabilidad.
El parámetro weight_decay introduce una ligera regularización L2 para prevenir sobreajuste.

🔹 5. Regularización

- Usado: nn.Dropout2d(0.1)

Justificación:
El uso del dropout2d + dropout ya que: 
- Dropout2d apaga canales completos (feature maps) en una capa convolucional.
- Mientras que Dropout “normal” apaga neuronas individuales aleatoriamente.

Esto nos permite reducir el sobreajuste y mejorar la generalización del modelo.
