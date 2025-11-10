🧪 Comparativa principal entre Ensayo 1 y Ensayo 3

En este segundo ensayo se realizaron ajustes estructurales y de optimización con el objetivo de simplificar la red, mejorar la estabilidad del entrenamiento y 
reducir el número total de parámetros, manteniendo un buen poder de representación para la clasificación de letras en lenguaje de signos.

🔹 1. Arquitectura general

- Antes (Ensayo 1)

  Red convolucional con tres bloques Conv–BatchNorm–ReLU–Pool y un clasificador totalmente conectado con tres capas densas (fc1, fc2, fc3).

- Ahora (Ensayo 3)

  Se amplió la parte convolucional a cuatro bloques (mayor profundidad), pero se eliminó el clasificador denso y se reemplazó por una 
  combinación de Global Average Pooling (GAP) seguido de una sola capa Linear.

Justificación:
El uso de nn.AdaptiveAvgPool2d((1, 1)) permite condensar la información espacial de cada canal sin necesidad de aplanar todo el tensor, reduciendo así millones de parámetros de las capas densas.
Esto da como resultado un modelo:
- Más compacto
- Más rápido de entrenar
- Con menor riesgo de sobreajuste

🔹 2. Capacidad convolucional

- Antes:

  Último bloque con 64 canales tras tres convoluciones (conv1–conv3).

- Ahora:

  Se añadió una cuarta capa convolucional (conv4) para llegar también a 64 canales, pero distribuyendo mejor la extracción de características (8 → 16 → 32 → 64).

Beneficio:
Este escalado progresivo permite una mejor jerarquía de representación visual y aprovecha mejor la profundidad de la red antes del pooling global.

🔹 3. Clasificador final

- Antes (Ensayo 1):

  self.fc1 = nn.LazyLinear(1024)
  self.bn_fc1 = nn.BatchNorm1d(1024)
  self.fc2 = nn.Linear(1024, 256)
  self.bn_fc2 = nn.BatchNorm1d(256)
  self.fc3 = nn.Linear(256, num_classes)

- Ahora (Ensayo 3):

  self.gap = nn.AdaptiveAvgPool2d((1, 1))
  self.fc = nn.Linear(64, num_classes)

Justificación:
El nuevo clasificador con GAP:
- Reduce enormemente los parámetros entrenables.
- Aumenta la regularización implícita.
- Hace que la red dependa más de las activaciones convolucionales que de las capas densas, mejorando la generalización.

🔹 4. Optimizador

- Antes: optim.Adam(lr=0.001, weight_decay=1e-4)

- Ahora: optim.AdamW(lr=0.002, weight_decay=1e-4)

Justificación:
AdamW mejora el control de la regularización al separar el weight decay del gradiente. Esto evita un mal ajuste del peso y suele ofrecer:
- Entrenamientos más estables
- Mejor desempeño en validación
- Convergencia más predecible en redes con BatchNorm


