🧪 Comparativa principal entre Ensayo 1 y Ensayo 2

Para este segundo ensayo se realizaron ajustes estructurales y funcionales en la red con el objetivo de mejorar la eficiencia del entrenamiento y reducir la complejidad del modelo, manteniendo la capacidad de generalización. Los principales cambios son los siguientes:

🔹 1. Optimizador

- Antes: optim.Adam
- Ahora: optim.AdamW

Justificación:
AdamW es una versión mejorada de Adam que separa explícitamente la regularización por peso (weight decay) del cálculo del gradiente.
Esto proporciona una mejor estabilidad del entrenamiento, reduce el sobreajuste y mejora la convergencia, especialmente en redes con normalización por lotes (BatchNorm). 

🔹 3. Cambio en la capa de clasificación

- Antes (Ensayo 1):

  self.fc1 = nn.LazyLinear(out_features=1024)
  self.bn_fc1 = nn.BatchNorm1d(1024)
  self.fc2 = nn.Linear(1024, 256)
  self.bn_fc2 = nn.BatchNorm1d(256)
  self.fc3 = nn.Linear(256, num_classes)

- Ahora (Ensayo 2):
  self.gap = nn.AdaptiveAvgPool2d((1, 1))
  self.fc = nn.Linear(64, num_classes)

Justificación:
Se reemplazó la arquitectura densa por una etapa de Global Average Pooling (GAP), que reduce cada mapa de características a un único valor promedio.
Esto:
- Disminuye drásticamente el número de parámetros.
- Reduce el riesgo de overfitting.
- Compensa el mayor costo de Mish al simplificar el clasificador.

Hace que la red sea más ligera y generalizable.
