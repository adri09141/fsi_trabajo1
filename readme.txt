Antes de ejecutar el entrenamiento, lo primero es configurar la variable USE_SUBSET en el fichero dataset.py:

- USE_SUBSET = True: la red neuronal utilizará únicamente una parte de las imágenes, definida por la variable SUBSET_SIZE. 
  Esto permite entrenar más rápido y probar configuraciones sin procesar todo el dataset.
- USE_SUBSET = False: la red neuronal utilizará todas las imágenes del dataset. Esto es más realista, pero puede tardar bastante 
  (¡posiblemente más de una hora incluso con GPU!).

A continuación, abre el fichero train.py y ajusta la variable NUM_EPOCHS_TO_TRAIN, que define el número de épocas que entrenará la red:

- Se recomienda un valor entre 10 y 30.
- Valores más bajos pueden no ser suficientes para que la red aprenda correctamente, mientras que valores muy altos pueden tardar demasiado y 
  hacer difícil visualizar la evolución en las gráficas.

Una vez configurado esto, ejecuta el script y verás cómo la red neuronal empieza a aprender.
