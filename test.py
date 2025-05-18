import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os
import tensorflow as tf
import numpy as np

# --------------------- SkinClassifier Class ---------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from keras.models import load_model

# --------------------- Constants ---------------------
MODEL_PATH = "model.h5"
lesion_type_dict = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi (normal)',
    5: 'Melanoma',
    6: 'Vascular lesions'
}

# --------------------- Load Model ---------------------
from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH)

# --------------------- Preprocessing ---------------------
def preprocess_image(img_path, size=71):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0), img

# --------------------- Grad-CAM ---------------------
def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy(), class_idx.numpy(), predictions.numpy()

# --------------------- Overlay Heatmap ---------------------
def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
    return overlayed

# --------------------- Predict & Plot ---------------------

def gradcam_predict(img_path, save_path='gradcam_output.png', conv_layer_name='block1_conv1'):
    img_array, img_rgb = preprocess_image(img_path)
    img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)

    heatmap, pred_class, preds = make_gradcam_heatmap(model, img_array, conv_layer_name)

    overlayed = overlay_gradcam(img_rgb_uint8, heatmap)

    predicted_label = lesion_type_dict.get(pred_class, "Unknown")
    confidence = float(preds[0][pred_class])  # Convert to float

    print(f"Saved Grad-CAM output to {save_path}")

    return {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "gradcam_overlay": overlayed,
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
MODEL_PATH = "model.h5"



#classifier = SkinClassifier(MODEL_PATH, layer_name='conv2d_4', x_train_mean=x_train_mean,
                            #x_train_std=x_train_std, class_names=class_names)

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
import numpy as np
import tensorflow as tf
import cv2

# Load your model

from keras.models import load_model
model = load_model(MODEL_PATH, compile=False)


# Define class labels
lesion_type_dict = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi (normal)',
    5: 'Melanoma',
    6: 'Vascular lesions',
}


# Real-time Grad-CAM with Webcam
#def live_gradcam():
   



def preprocess_frame(frame, size=71):
    original_frame = frame.copy()
    resized = cv2.resize(frame, (size, size))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    return np.expand_dims(img_rgb, axis=0), original_frame



def open_camera():
    root.withdraw()
    cam_window = tk.Toplevel()
    cam_window.attributes('-fullscreen', True)
    cam_window.configure(bg="black")

    screen_width = cam_window.winfo_screenwidth()
    screen_height = cam_window.winfo_screenheight()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    lmain = tk.Label(cam_window, bg="black")
    lmain.place(x=0, y=0, relwidth=1, relheight=1)

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

            save_btn.place(relx=0.9, rely=0.45, anchor="center", width=130, height=40)
            discard_btn.place(relx=0.9, rely=0.53, anchor="center", width=130, height=40)

    def save_image(pil_img):
        file_path = get_next_image_filename()
        pil_img.save(file_path)
        img_cv = cv2.imread(file_path)
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        result = gradcam_predict(file_path)
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

    gradcam_enabled = tk.BooleanVar(value=True)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            input_tensor, display_frame = preprocess_frame(frame)
            rgb_uint8 = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            if gradcam_enabled.get():
                try:
                    heatmap, pred_class, preds = make_gradcam_heatmap(model, input_tensor, 'block1_conv1')
                    overlayed = overlay_gradcam(rgb_uint8, heatmap)

                    label = lesion_type_dict.get(pred_class, "Unknown")
                    confidence = preds[0][pred_class] * 100
                    cv2.putText(overlayed, f"{label}: {confidence:.2f}%", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                except Exception as e:
                    print(f"Grad-CAM error: {e}")
                    overlayed = rgb_uint8
            else:
                overlayed = rgb_uint8

            cv2image = cv2.cvtColor(overlayed, cv2.COLOR_RGB2RGBA)
            img = Image.fromarray(cv2image).resize((screen_width, screen_height), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)

        lmain.after(10, update_frame)

    # ---------- Buttons ----------
    button_style = {
        "font": ("Helvetica", 12, "bold"),
        "bg": "#007acc",
        "fg": "white",
        "bd": 0,
        "highlightthickness": 0
    }

    capture_btn = tk.Button(cam_window, text="\U0001F4F8 Capture", **button_style, command=capture_image)
    # home_btn = tk.Button(cam_window, text="\U0001F3E0 Home", bg="#28a745", fg="white", font=("Helvetica", 12, "bold"),
    #                      bd=0, highlightthickness=0, command=close_camera)
    exit_btn = tk.Button(cam_window, text="\u274C Exit", bg="#dc3545", fg="white", font=("Helvetica", 12, "bold"),
                         bd=0, highlightthickness=0, command=close_camera)

    save_btn = tk.Button(cam_window, text="\U0001F4BE Save", bg="#28a745", fg="white", font=("Helvetica", 12, "bold"),
                         bd=0, highlightthickness=0, command=lambda: save_image(preview_label.image))
    discard_btn = tk.Button(cam_window, text="\U0001F5D1 Discard", bg="#dc3545", fg="white", font=("Helvetica", 12, "bold"),
                            bd=0, highlightthickness=0, command=hide_preview)

    toggle_btn = tk.Checkbutton(
        cam_window,
        text="Grad-CAM ON/OFF",
        font=("Helvetica", 12, "bold"),
        bg="#ffc107",
        fg="black",
        variable=gradcam_enabled,
        onvalue=True,
        offvalue=False,
        width=18,
        height=2,
        relief="flat"
    )

    # Position buttons (right side, vertically spaced)
    toggle_btn.place(relx=0.60, rely=0.86, anchor="ne")

    capture_btn.place(relx=0.2, rely=0.9, anchor="center", width=130, height=40)
    # home_btn.place(relx=0.5, rely=0.9, anchor="center", width=130, height=40)
    exit_btn.place(relx=0.8, rely=0.9, anchor="center", width=130, height=40)

    # Ensure buttons stay on top
    for btn in [capture_btn, exit_btn, toggle_btn]:
        btn.lift()

    update_frame()



def show_result_image():
    if not os.path.exists(RESULT_IMAGE_PATH):
        messagebox.showerror("Error", "Result image not found.")
        return

    result_win = tk.Toplevel()
    result_win.title("GradCAM Output")
    result_win.geometry("480x480")
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
        result = gradcam_predict(path)
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
tk.Button(root, image=result_img_photo, command=show_result_image, borderwidth=0, bg=root["bg"]).place(x=1100, y=180)

camera_img = Image.open(CAMERA_ICON_PATH).resize((40, 40), Image.Resampling.LANCZOS)
camera_photo = ImageTk.PhotoImage(camera_img)
tk.Button(root, image=camera_photo, command=open_camera, borderwidth=0, bg=root["bg"]).place(x=1100, y=250)

about_img = Image.open(ABOUT_ICON_PATH).resize((40, 40), Image.Resampling.LANCZOS)
about_photo = ImageTk.PhotoImage(about_img)
tk.Button(root, image=about_photo, command=open_about, borderwidth=0, bg=root["bg"]).place(x=1100, y=310)

gallery_img = Image.open(GALLERY_ICON_PATH).resize((40, 40), Image.Resampling.LANCZOS)
gallery_photo = ImageTk.PhotoImage(gallery_img)
tk.Button(root, image=gallery_photo, command=open_gallery, borderwidth=0, bg=root["bg"]).place(x=1100, y=370)

tk.Label(root, text="Developed by Team I/O | 2025", font=("Helvetica", 12), fg="#319ECA", bg="#D2F1FC").pack(side="bottom", pady=5)
tk.Button(root, text="\u274C", font=("Helvetica", 12, "bold"), fg="white", bg="#dc3545", command=root.destroy, bd=0, highlightthickness=0).place(x=1150, y=120, width=40, height=40)

root.mainloop()
