import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os
import tensorflow as tf
import numpy as np

# --------------------- SkinClassifier Class ---------------------
class SkinClassifier:
    def __init__(self, model_path, layer_name, x_train_mean, x_train_std, class_names):
        self.model = tf.keras.models.load_model(model_path)
        self.layer_name = layer_name
        self.x_train_mean = x_train_mean
        self.x_train_std = x_train_std
        self.class_names = class_names

    def preprocess(self, image):
        image = cv2.resize(image, (100, 75))
        image = image.astype('float32')
        image = (image - self.x_train_mean) / self.x_train_std
        return image

    def predict(self, image):
        image_processed = self.preprocess(image)
        image_batch = np.expand_dims(image_processed, axis=0)
        preds = self.model.predict(image_batch, verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]
        return class_idx, confidence, preds

    def get_gradcam_heatmap(self, image, class_index):
        image_processed = self.preprocess(image)
        img_tensor = tf.expand_dims(image_processed, axis=0)

        grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_gradcam(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
        return overlayed

    def classify(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        class_idx, confidence, _ = self.predict(image)
        heatmap = self.get_gradcam_heatmap(image, class_idx)
        image_uint8 = np.uint8((image - np.min(image)) / (np.max(image) - np.min(image)) * 255)
        overlay_img = self.overlay_gradcam(image_uint8, heatmap)

        return {
            "predicted_class": self.class_names[class_idx],
            "confidence": float(confidence),
            "gradcam_overlay": overlay_img,
            "heatmap": heatmap
        }

# --------------------- Constants ---------------------
SAVE_DIR = os.getcwd()
BG_IMAGE_PATH = "bgimagefinal.jpg"
CAMERA_ICON_PATH = "camera_icon.jpg"
ABOUT_ICON_PATH = "aboutimg.jpg"
RESULT_ICON_PATH = "logo.png"
RESULT_IMAGE_PATH = "photo_3.jpg"
ABOUT_BG_IMAGE_PATH = "aboutimgfinal.jpg"
GALLERY_ICON_PATH = "gallery_icon.jpg"
GALLERY_IMAGES_DIR = "gallery_images"
MODEL_PATH = "skin_modelnew.h5"

# --------------------- Initialize Classifier ---------------------
x_train_mean = 159.88411714650246
x_train_std = 46.45448942251351
class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
               'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']
classifier = SkinClassifier(MODEL_PATH, layer_name='conv2d_4', x_train_mean=x_train_mean,
                            x_train_std=x_train_std, class_names=class_names)

# --------------------- GUI Setup ---------------------
root = tk.Tk()
root.title("\U0001F489 Skin Disease Detection System")
root.attributes("-fullscreen", True)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

bg_img = Image.open(BG_IMAGE_PATH).resize((screen_width, screen_height), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_img)
tk.Label(root, image=bg_photo).place(x=0, y=0, relwidth=1, relheight=1)

def get_next_image_filename():
    i = 1
    while True:
        filename = f"photo_{i}.jpg"
        if not os.path.exists(os.path.join(SAVE_DIR, filename)):
            return os.path.join(SAVE_DIR, filename)
        i += 1

def open_camera():
    root.withdraw()
    cam_window = tk.Toplevel()
    cam_window.attributes('-fullscreen', True)
    cam_window.configure(bg="black")

    cap = cv2.VideoCapture(0)
    lmain = tk.Label(cam_window, bg="black")
    lmain.pack(fill="both", expand=True)

    preview_label = tk.Label(cam_window, bg="black")
    preview_label.place(relx=0.5, rely=0.3, anchor="center")
    preview_label.lower()

    def capture_image():
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img).resize((200, 150), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(pil_img)
            preview_label.configure(image=imgtk)
            preview_label.image = pil_img
            preview_label.imgtk = imgtk
            preview_label.lift()

            save_btn.place(relx=0.35, rely=0.75, anchor="center", width=80, height=30)
            discard_btn.place(relx=0.65, rely=0.75, anchor="center", width=80, height=30)

    def save_image(pil_img):
        file_path = get_next_image_filename()
        pil_img.save(file_path)
        img_cv = cv2.imread(file_path)
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        result = classifier.classify(img_cv_rgb)
        cv2.imwrite(RESULT_IMAGE_PATH, cv2.cvtColor(result['gradcam_overlay'], cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Prediction", f"\U0001F4C8 Predicted: {result['predicted_class']}\nConfidence: {result['confidence']:.2f}")
        hide_preview()

    def hide_preview():
        preview_label.lower()
        preview_label.configure(image=None)
        save_btn.place_forget()
        discard_btn.place_forget()

    def close_camera():
        cap.release()
        cam_window.destroy()
        root.deiconify()

    def update_frame():
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image).resize((screen_width, screen_height), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
        lmain.after(10, update_frame)

    capture_btn = tk.Button(cam_window, text="\U0001F4F8 Capture", font=("Helvetica", 10, "bold"), bg="#007acc", fg="white", command=capture_image)
    home_btn = tk.Button(cam_window, text="\U0001F3E0 Home", font=("Helvetica", 10, "bold"), bg="#28a745", fg="white", command=close_camera)
    exit_btn = tk.Button(cam_window, text="\u274C Exit", font=("Helvetica", 10, "bold"), bg="#dc3545", fg="white", command=close_camera)
    save_btn = tk.Button(cam_window, text="\U0001F4BE Save", font=("Helvetica", 10, "bold"), bg="#28a745", fg="white", command=lambda: save_image(preview_label.image))
    discard_btn = tk.Button(cam_window, text="\U0001F5D1 Discard", font=("Helvetica", 10, "bold"), bg="#dc3545", fg="white", command=hide_preview)

    capture_btn.place(relx=0.2, rely=0.9, anchor="center", width=90, height=30)
    home_btn.place(relx=0.5, rely=0.9, anchor="center", width=90, height=30)
    exit_btn.place(relx=0.8, rely=0.9, anchor="center", width=90, height=30)

    update_frame()

def show_result_image():
    if not os.path.exists(RESULT_IMAGE_PATH):
        messagebox.showerror("Error", "Result image not found.")
        return

    result_win = tk.Toplevel()
    result_win.title("GradCAM Output")
    result_win.geometry("480x320")
    result_win.configure(bg="white")

    result_img = Image.open(RESULT_IMAGE_PATH).resize((440, 240), Image.Resampling.LANCZOS)
    result_photo = ImageTk.PhotoImage(result_img)
    tk.Label(result_win, image=result_photo, bg="white").pack(padx=10, pady=10)
    result_win.result_img_ref = result_photo

    tk.Button(result_win, text="Close", font=("Helvetica", 10, "bold"), bg="#dc3545", fg="white", command=result_win.destroy).pack(pady=5)

def open_about():
    about_win = tk.Toplevel()
    about_win.title("About Project")
    about_win.geometry("480x320")

    about_bg_img = Image.open(ABOUT_BG_IMAGE_PATH).resize((480, 320), Image.Resampling.LANCZOS)
    about_bg_photo = ImageTk.PhotoImage(about_bg_img)
    tk.Label(about_win, image=about_bg_photo).place(x=0, y=0, relwidth=1, relheight=1)
    about_win.about_bg_ref = about_bg_photo

    tk.Button(about_win, text="\U0001F3E0", font=("Helvetica", 12, "bold"), fg="white", bg="#007acc", command=about_win.destroy, bd=0, highlightthickness=0).place(x=420, y=10, width=40, height=40)

def open_gallery():
    if not os.path.exists(GALLERY_IMAGES_DIR):
        messagebox.showerror("Error", f"Gallery folder '{GALLERY_IMAGES_DIR}' not found.")
        return

    gallery_win = tk.Toplevel()
    gallery_win.title("Select Image from Gallery")
    gallery_win.geometry("480x320")
    gallery_win.configure(bg="white")

    images = []
    for i in range(1, 10):  # image1.jpg to image9.jpg
        img_path = os.path.join(GALLERY_IMAGES_DIR, f"image{i}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((100, 75), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            images.append((photo, img_path))

    def on_image_click(path):
        img_cv = cv2.imread(path)
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        result = classifier.classify(img_cv_rgb)
        cv2.imwrite(RESULT_IMAGE_PATH, cv2.cvtColor(result['gradcam_overlay'], cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Prediction", f"\U0001F4C8 Predicted: {result['predicted_class']}\nConfidence: {result['confidence']:.2f}")
        gallery_win.destroy()

    for idx, (photo, path) in enumerate(images):
        btn = tk.Button(gallery_win, image=photo, command=lambda p=path: on_image_click(p))
        btn.image = photo
        row = idx // 3
        col = idx % 3
        btn.grid(row=row, column=col, padx=10, pady=10)

    tk.Button(gallery_win, text="Close", font=("Helvetica", 10, "bold"), bg="#dc3545", fg="white", command=gallery_win.destroy).grid(row=3, column=1, pady=10)

# --------------------- GUI Buttons ---------------------
result_img_icon = Image.open(RESULT_ICON_PATH).resize((40, 40), Image.Resampling.LANCZOS)
result_img_photo = ImageTk.PhotoImage(result_img_icon)
tk.Button(root, image=result_img_photo, command=show_result_image, borderwidth=0, bg=root["bg"]).place(x=400, y=70)

camera_img = Image.open(CAMERA_ICON_PATH).resize((40, 40), Image.Resampling.LANCZOS)
camera_photo = ImageTk.PhotoImage(camera_img)
tk.Button(root, image=camera_photo, command=open_camera, borderwidth=0, bg=root["bg"]).place(x=400, y=130)

about_img = Image.open(ABOUT_ICON_PATH).resize((40, 40), Image.Resampling.LANCZOS)
about_photo = ImageTk.PhotoImage(about_img)
tk.Button(root, image=about_photo, command=open_about, borderwidth=0, bg=root["bg"]).place(x=400, y=190)

gallery_img = Image.open(GALLERY_ICON_PATH).resize((40, 40), Image.Resampling.LANCZOS)
gallery_photo = ImageTk.PhotoImage(gallery_img)
tk.Button(root, image=gallery_photo, command=open_gallery, borderwidth=0, bg=root["bg"]).place(x=400, y=250)

tk.Label(root, text="Developed by Team I/O | 2025", font=("Helvetica", 12), fg="#319ECA", bg="#D2F1FC").pack(side="bottom", pady=5)
tk.Button(root, text="\u274C", font=("Helvetica", 12, "bold"), fg="white", bg="#dc3545", command=root.destroy, bd=0, highlightthickness=0).place(x=430, y=10, width=40, height=40)

root.mainloop()
