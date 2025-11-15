Ensayo 1 – Descripción del modelo CNN

Este modelo utiliza una arquitectura CNN clásica compuesta por tres bloques convolucionales seguidos de un clasificador denso de tres capas. Es una red funcional pero con un número elevado de parámetros en la parte final, lo que la hace susceptible al sobreajuste.

1. Arquitectura general

El modelo está formado por:

Bloques convolucionales:
1. Conv2d -> BatchNorm -> ReLU -> MaxPool2d  
   - 16 canales  
   - Reduce a la mitad las dimensiones espaciales

2. Conv2d -> BatchNorm -> ReLU -> MaxPool2d  
   - 32 canales  
   - Segunda reducción a la mitad

3. Conv2d -> BatchNorm -> ReLU -> MaxPool2d  
   - 64 canales  
   - Tercera reducción a la mitad

También se incluye Dropout2d(0.1) como regularización en estas capas.

2. Tamaño del tensor en cada etapa (entrada 128x128)

Despues del primer bloque: 64x64  
Despues del segundo bloque: 32x32  
Despues del tercer bloque: 16x16

Antes del clasificador el tensor tiene forma [B, 64, 16, 16], equivalente a 16384 valores por muestra.

3. Clasificador totalmente conectado

Despues de aplanar el tensor se pasa por:

1. Capa linear de 16384 a 1024  
   Incluye BatchNorm1d, ReLU y Dropout(0.3)

2. Capa linear de 1024 a 256  
   Incluye BatchNorm1d, ReLU y Dropout(0.3)

3. Capa final de 256 a num_classes

Este bloque contiene la mayoría de los parámetros del modelo.

4. Activación

La función de activación utilizada es ReLU en todas las capas.

5. Regularización

Se usa Dropout(0.3) en las capas densas y Dropout2d(0.1) en las convolucionales.

6. Optimizador

El entrenamiento se realiza con Adam, learning rate 0.001 y weight decay 1e-4.

Resumen del Ensayo 1

Es una CNN tradicional con tres bloques convolucionales (16, 32 y 64 canales), activaciones ReLU y un clasificador grande de tres capas. Es un modelo sencillo de implementar, potente, pero con un número alto de parámetros densos que pueden favorecer el sobreajuste.
