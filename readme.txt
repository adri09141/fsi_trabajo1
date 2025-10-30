Antes de comenzar a entrenar la red neuronal, lo primero que debes hacer es configurar la variable USE_SUBSET en el archivo dataset.py:

- USE_SUBSET = True: la red solo utilizará una parte de las imágenes definidas por la variable SUBSET_SIZE. Esto es útil si quieres entrenar rápidamente y probar configuraciones sin procesar todo el dataset.

- USE_SUBSET = False: la red usará todas las imágenes disponibles. Esto es más realista y permitirá que la red aprenda mejor, pero puede tardar bastante, incluso más de una hora aunque uses GPU.

A continuación, abre el archivo train.py y ajusta la variable NUM_EPOCHS_TO_TRAIN, que determina cuántas épocas entrenará la red:

- Se recomienda un valor entre 10 y 30.

- Valores más bajos pueden no ser suficientes para que la red aprenda correctamente, mientras que valores muy altos aumentan el tiempo de entrenamiento y pueden dificultar visualizar la evolución en las gráficas.

Una vez configuradas estas opciones, ejecuta el script. Verás cómo la red neuronal comienza a aprender, mostrando información del progreso por cada época.

Cuando se complete el número de épocas definido, el script generará automáticamente una gráfica del aprendizaje, mostrando cómo evolucionaron la pérdida y la precisión durante el entrenamiento. Tras esto, también podrás ver el resultado del test, evaluando el desempeño final de la red en datos que nunca vio durante el entrenamiento.

Si quieres profundizar más, puedes abrir el archivo matriz.py y ejecutarlo para generar una matriz de confusión, lo que te permitirá ver en qué clases la red se confundió más y analizar los errores de manera más detallada.
