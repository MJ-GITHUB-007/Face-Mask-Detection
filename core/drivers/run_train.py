from warnings import filterwarnings
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'core'))
filterwarnings(action='ignore')

import pretty_errors

from main_dir.mask_script import Train

epochs = input("\nNumber of epochs to train : ").strip()
try:
    epochs = int(epochs)
except:
    raise Exception("Input number 1 or 2")

existing_model = input("Train existing model? [Y/n] : ").strip().lower()
if existing_model in {'n', 'no'}:
    new_model = True
else:
    new_model = False

live_plot = input("Display live plot? [Y/n] : ").strip().lower()
if live_plot in {'n', 'no'}:
    live_plot = False
else:
    live_plot = True
 
trainer = Train(batch_size=16, val_batch_size=8, learning_rate=1e-6, start_new=new_model, liveplot=live_plot)
trainer.train(num_epochs=epochs)