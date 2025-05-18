import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os
import tensorflow as tf
import numpy as np

# --- CONFIGURATION ---
SAVE_DIR = os.getcwd()
BG_IMAGE_PATH = "bgimagefinal.jpg"
ABOUT_BG_IMAGE_PATH = "about_bg.jpg"
CAMERA_ICON_PATH = "camera_icon.jpg"
ABOUT_ICON_PATH = "about_icon.jpg"
RESULT_ICON_PATH = "logo.png"
MODEL_PATH = "skin_disease_model.h5"
CAMERA_INDEX = 0  # Index for dermatoscope camera

# --- LOAD MODEL ---
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ['Eczema', 'Psoriasis', 'Rosacea', 'Melanoma', 'Healthy']  # Adjust to your model

def get_next_image_filename():
    i = 1
    while True:
        filename = f"photo_{i}.jpg"
        if not os.path.exists(os.path.join(SAVE_DIR, filename)):
            return os.path.join(SAVE_DIR, filename)
        i += 1

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Match model input
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_disease(img):
    preprocessed = preprocess_image(img)
    preds = model.predict(preprocessed)
    idx = np.argmax(preds)
    confidence = preds[0][idx]
    return CLASS_NAMES[idx], confidence

# --- GUI INIT ---
root = tk.Tk()
root.title("🩺 Skin Disease Detection System")
root.attributes("-fullscreen", True)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# --- BACKGROUND ---
bg_img = Image.open(BG_IMAGE_PATH).resize((screen_width, screen_height), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_img)
tk.Label(root, image=bg_photo).place(x=0, y=0, relwidth=1, relheight=1)

# --- TITLE ---
tk.Label(
    root, text="🩺 Skin Disease Detection System",
    font=("Helvetica", 32, "bold"),
    fg="white", bg="#000000"
).place(x=355, y=30)

# --- CAMERA + PREDICTION WINDOW ---
def open_camera():
    root.withdraw()
    cam_win = tk.Toplevel()
    cam_win.title("Live Camera")
    cam_win.geometry(f"{screen_width}x{screen_height}")
    cam_win.configure(bg="black")

    cam = cv2.VideoCapture(CAMERA_INDEX)
    lmain = tk.Label(cam_win)
    lmain.pack()

    preview_label = tk.Label(cam_win, bg="black")
    preview_label.place(relx=0.5, rely=0.4, anchor="center")
    preview_label.lower()

    result_label = tk.Label(cam_win, font=("Helvetica", 20, "bold"), fg="white", bg="black")
    result_label.place(relx=0.5, rely=0.8, anchor="center")

    def capture_predict():
        ret, frame = cam.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((600, 400), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(pil_img)

            preview_label.configure(image=imgtk)
            preview_label.image = pil_img
            preview_label.imgtk = imgtk
            preview_label.lift()

            pred_class, conf = predict_disease(rgb)
            result_label.config(text=f"Prediction: {pred_class} ({conf*100:.2f}%)")

            save_path = get_next_image_filename()
            cv2.imwrite(save_path, frame)
            messagebox.showinfo("Saved", f"📸 Image saved as {os.path.basename(save_path)}")

    def close_cam():
        cam.release()
        cam_win.destroy()
        root.deiconify()

    btn_frame = tk.Frame(cam_win, bg="black")
    btn_frame.pack(side="bottom", pady=30)

    tk.Button(btn_frame, text="📸 Capture & Predict", font=("Helvetica", 14, "bold"),
              bg="#007acc", fg="white", command=capture_predict).pack(side="left", padx=20)
    tk.Button(btn_frame, text="🏠 Home", font=("Helvetica", 14, "bold"),
              bg="#28a745", fg="white", command=close_cam).pack(side="left", padx=20)
    tk.Button(btn_frame, text="❌ Exit", font=("Helvetica", 14, "bold"),
              bg="#dc3545", fg="white", command=close_cam).pack(side="left", padx=20)

    def show_frame():
        ret, frame = cam.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
        lmain.after(10, show_frame)

    show_frame()

# --- ABOUT WINDOW ---
def open_about():
    about_win = tk.Toplevel()
    about_win.title("About Project")
    about_win.geometry("800x600")

    about_bg = Image.open(ABOUT_BG_IMAGE_PATH).resize((800, 600), Image.Resampling.LANCZOS)
    about_bg_photo = ImageTk.PhotoImage(about_bg)
    tk.Label(about_win, image=about_bg_photo).place(x=0, y=0, relwidth=1, relheight=1)
    about_win.about_img_ref = about_bg_photo

    tk.Label(about_win, text="📘 About the Project", font=("Helvetica", 24, "bold"),
             fg="white", bg="#000000").pack(pady=20)

    about_text = (
        "🔍 This Skin Disease Detection System uses a dermatoscope camera\n"
        "and a TensorFlow model to identify skin conditions from images.\n\n"
        "✨ Features:\n"
        "  ✅ Real-time dermatoscope imaging\n"
        "  ✅ Instant prediction with accuracy\n"
        "  ✅ Easy-to-use GUI on Raspberry Pi\n"
        "🛠️ Built with Python, Tkinter, OpenCV, and TensorFlow."
    )

    frame = tk.Frame(about_win, bg="#000000")
    frame.pack(fill="both", expand=True, padx=30)
    tk.Label(frame, text=about_text, font=("Helvetica", 14),
             fg="white", bg="#000000", justify="left").pack(anchor="w")

    tk.Button(about_win, text="🏠 Home", font=("Helvetica", 12, "bold"),
              bg="#28a745", fg="white", command=about_win.destroy).place(x=740, y=10, width=50, height=30)

# --- ICON BUTTONS ---
camera_img = Image.open(CAMERA_ICON_PATH).resize((240, 180), Image.Resampling.LANCZOS)
camera_photo = ImageTk.PhotoImage(camera_img)
tk.Button(root, image=camera_photo, command=open_camera, borderwidth=0, bg=root["bg"]).place(
    x=screen_width//2 + 305, y=(screen_height//2) - 140)

about_img = Image.open(ABOUT_ICON_PATH).resize((240, 180), Image.Resampling.LANCZOS)
about_photo = ImageTk.PhotoImage(about_img)
tk.Button(root, image=about_photo, command=open_about, borderwidth=0, bg=root["bg"]).place(
    x=screen_width//2 + 305, y=(screen_height//2) - 340)

# --- FOOTER & CLOSE ---
tk.Label(root, text="Developed by Your Team | 2025", font=("Helvetica", 10), fg="white", bg="black").pack(side="bottom", pady=10)
tk.Button(root, text="❌", font=("Helvetica", 16, "bold"), fg="white", bg="#dc3545", command=root.destroy).place(x=screen_width - 50, y=10, width=40, height=40)

root.mainloop()
