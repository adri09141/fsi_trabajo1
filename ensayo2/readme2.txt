🧪 Comparativa principal entre Ensayo 1 y Ensayo 

En este segundo ensayo se realizaron ajustes estructurales, funcionales y de activación con el objetivo de ligerizar el modelo, mejorar la estabilidad del aprendizaje y mantener la capacidad de representación necesaria para la detección precisa de letras en lenguaje de signos.

🔹 1. Arquitectura general

- Antes (Ensayo 1):
  Red convolucional con tres bloques Conv–BatchNorm–ReLU–Pool y un clasificador totalmente conectado con tres capas densas (fc1, fc2, fc3).

- Ahora (Ensayo 2):
  Se amplió la parte convolucional a cuatro bloques (16 → 32 → 32 → 64) para una extracción de características más jerárquica, y se eliminó el clasificador denso en favor de una etapa de Global Average Pooling (GAP) seguida de una sola capa lineal.

Justificación:
El uso de nn.AdaptiveAvgPool2d((1, 1)) permite condensar la información espacial sin necesidad de aplanar todo el tensor, reduciendo millones de parámetros y mejorando la eficiencia computacional.
Esto hace que el modelo sea:
- Más compacto y rápido de entrenar
- Menos propenso al sobreajuste
- Más generalizable en validación

🔹 2. Capacidad convolucional

- Antes:
  Tres capas convolucionales (16 → 32 → 64) seguidas de capas densas con más de 1 millón de parámetros.

- Ahora:
  Cuatro capas convolucionales (16 → 32 → 32 → 64), todas normalizadas con BatchNorm2d y activadas con Mish.

Beneficio:
Este patrón progresivo permite extraer características visuales más ricas sin recurrir a capas densas costosas.
La repetición de dos bloques con 32 canales estabiliza el flujo de gradiente y mejora la sensibilidad a variaciones sutiles en las formas de las manos.

🔹 3. Función de activación

- Antes (Ensayo 1): nn.ReLU()

- Ahora (Ensayo 2): nn.Mish()

Justificación:
Mish es una activación más suave y continua que ReLU, definida como x * tanh(softplus(x)).
Proporciona una mejor propagación de gradientes en valores negativos, facilitando una convergencia más estable y mejor precisión final, especialmente en tareas visuales complejas como la interpretación de gestos o letras manuales.

🔹 4. Regularización

- Antes: nn.Dropout(0.3)

- Ahora: nn.Dropout(0.15)

Justificación:
La reducción del dropout rate es coherente con la simplificación del modelo.
Con menos capas densas, el riesgo de sobreajuste disminuye, por lo que un valor moderado (0.15) mantiene la regularización sin afectar la retención de características relevantes.

🔹 5. Clasificador final

- Antes (Ensayo 1):

  self.fc1 = nn.LazyLinear(1024)
  self.bn_fc1 = nn.BatchNorm1d(1024)
  self.fc2 = nn.Linear(1024, 256)
  self.bn_fc2 = nn.BatchNorm1d(256)
  self.fc3 = nn.Linear(256, num_classes)

- Ahora (Ensayo 2):

  self.gap = nn.AdaptiveAvgPool2d((1, 1))
  self.fc = nn.Linear(64, num_classes)

Justificación:
El nuevo clasificador reduce enormemente el número de parámetros y prioriza la información proveniente de las capas convolucionales, lo que mejora la generalización y la estabilidad de la validación.

🔹 6. Optimizador

- Antes: optim.Adam(lr=0.001, weight_decay=1e-4)

- Ahora: optim.AdamW(lr=0.002, weight_decay=1e-4)

Justificación:

AdamW separa correctamente la penalización por pesos del cálculo del gradiente, lo que produce un entrenamiento más estable y mejor control de regularización.
Esto es especialmente útil en redes con BatchNorm y Mish, que tienden a generar gradientes más suaves.

🔹 7. Train_transform

- Antes (Ensayo 1):

  train_transform = transforms.Compose([
      transforms.Resize(img_size),
      transforms.RandomCrop(img_size, padding=8),  # más barato que RandomResizedCrop
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(10),               # reemplaza RandomAffine
      transforms.ColorJitter(
          brightness=0.1,
          contrast=0.1,
          saturation=0.05
      ),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

- Ahora (Ensayo 2):

  train_transform = transforms.Compose([
      transforms.Resize(img_size),
      transforms.RandomCrop(img_size, padding=4),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(5),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

Justificación:

El nuevo esquema de data augmentation busca un equilibrio entre diversidad y consistencia visual, reduciendo el grado de aleatoriedad aplicado a las imágenes para evitar que el modelo aprenda patrones deformados o irreales de las manos.
Los cambios principales y su impacto:
  🔸 Reducción de padding en el recorte (8 → 4):
  Menor desplazamiento aleatorio del contenido visual. Favorece que las manos se mantengan centradas, lo que ayuda a conservar las proporciones naturales del gesto.

  🔸 Rotación más leve (10° → 5°):
  Mejora la estabilidad de la convergencia. En lenguaje de signos, las rotaciones excesivas pueden alterar completamente el significado del gesto.

  🔸 Eliminación de ColorJitter:
  Aunque útil para iluminación variable, su eliminación evita introducir ruido cromático innecesario, ya que las condiciones de captura en el dataset son       relativamente homogéneas.
  
  🔸 Normalización constante:
  Mantener los valores centrados en torno a cero mejora la estabilidad de las activaciones, especialmente con BatchNorm y Mish.

En conjunto, el nuevo train_transform genera un aprendizaje más robusto y consistente, reduciendo la variabilidad espuria mientras conserva la capacidad de generalización.

🔹 8. img_size 

- Antes (Ensayo 1): (128, 128)

- Ahora (Ensayo 2): (96, 96)

Justificación:
Reducir la resolución a 96×96 píxeles optimiza el equilibrio entre detalle visual y coste computacional.
Las letras del lenguaje de signos presentan formas bien definidas que pueden capturarse adecuadamente a esta resolución sin pérdida significativa de información discriminante.

Ventajas:
- Entrenamiento más rápido y eficiente, permitiendo mayor número de épocas sin sobrecargar GPU.
- Menor riesgo de sobreajuste, al reducir el volumen de píxeles redundantes.
- Mejor compatibilidad con redes más ligeras, manteniendo una representación suficiente para distinguir gestos similares (como “M” vs “N”).

