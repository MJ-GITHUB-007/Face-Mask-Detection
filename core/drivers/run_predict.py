from warnings import filterwarnings
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'core'))
filterwarnings(action='ignore')

import pretty_errors

from container.engine import Predict

image_path = input("\nPath of image to predict : ").strip()

plot_image = input("Display image? [Y/n] : ").strip().lower()
if plot_image in {'n', 'no'}:
    plot_image = False
else:
    plot_image = True
  
predictor = Predict()
predictor.predict(image_path, display_image=plot_image)