import tkinter as tk
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

root = tk.Tk()
root.title("TensorFlow GUI Test")
root.geometry("400x200")
tk.Label(root, text="TensorFlow and Tkinter are both working!").pack(pady=40)
root.mainloop()
