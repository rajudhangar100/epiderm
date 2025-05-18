import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk, ImageEnhance
import os

# Save path
SAVE_PATH = "captured_photo.jpg"

# Initialize main window
root = tk.Tk()
root.title("🩺 Skin Disease Detection System")
root.geometry("700x500")
root.resizable(False, False)

# Load and process background image
bg_img_path = "bgimage.jpg"  # Ensure this file is in the same directory
bg_img = Image.open(bg_img_path)
bg_img = bg_img.resize((700, 500), Image.Resampling.LANCZOS)

# Reduce opacity
bg_img = bg_img.convert("RGBA")
alpha = bg_img.split()[3]
alpha = ImageEnhance.Brightness(alpha).enhance(0.5)
bg_img.putalpha(alpha)
bg_photo = ImageTk.PhotoImage(bg_img)

# Background label
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Overlay frame for widgets
overlay = tk.Frame(root, bg="", width=700, height=500)
overlay.place(x=0, y=0)

# Function to show camera window
def open_camera():
    root.withdraw()
    cam_window = tk.Toplevel()
    cam_window.title("Live Camera")
    cam_window.geometry("700x500")
    cam_window.configure(bg="black")
    cam_window.resizable(False, False)

    lmain = tk.Label(cam_window)
    lmain.pack()

    cap = cv2.VideoCapture(0)

    def show_frame():
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
        if cam_window.winfo_exists():
            lmain.after(10, show_frame)

    def capture_image():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(SAVE_PATH, frame)
            messagebox.showinfo("Image Saved", f"📸 Image saved as {SAVE_PATH}")

    def exit_camera():
        cap.release()
        cam_window.destroy()
        root.destroy()

    def go_home():
        cap.release()
        cam_window.destroy()
        root.deiconify()

    # Capture Button
    capture_btn = tk.Button(
        cam_window, text="📸 Capture", font=("Helvetica", 12, "bold"),
        bg="#5ab2c5", fg="white", command=capture_image,
        relief="flat", bd=0
    )
    capture_btn.place(x=290, y=420, width=120, height=40)

    # Exit Button
    exit_btn = tk.Button(
        cam_window, text="❌", font=("Helvetica", 14, "bold"),
        bg="#d9534f", fg="white", command=exit_camera,
        relief="flat", bd=0
    )
    exit_btn.place(x=650, y=20, width=40, height=40)

    # Home Button
    home_btn = tk.Button(
        cam_window, text="🏠", font=("Helvetica", 14, "bold"),
        bg="#5ab2c5", fg="white", command=go_home,
        relief="flat", bd=0
    )
    home_btn.place(x=600, y=20, width=40, height=40)

    show_frame()

# ----------------------- GUI Layout -----------------------

# Title
title_label = tk.Label(
    overlay, text="🩺 Skin Disease Detection System",
    font=("Helvetica", 22, "bold"), fg="#5cdffc", bg=""
)
title_label.pack(pady=30)

# Description
description = (
    "This system captures skin images using a live camera\n"
    "and helps in detecting skin diseases using AI."
)
desc_label = tk.Label(
    overlay, text=description, font=("Helvetica", 13), fg="#ffffff", bg=""
)
desc_label.pack(pady=10)

# Camera Button
camera_btn = tk.Button(
    overlay, text="📷 Open Camera", font=("Helvetica", 14, "bold"),
    bg="#5ab2c5", fg="white", padx=20, pady=10, command=open_camera,
    relief="flat", bd=0
)
camera_btn.pack(pady=40)

# Footer
footer_label = tk.Label(
    overlay, text="Developed by Your Team | 2025",
    font=("Helvetica", 10), fg="#eeeeee", bg=""
)
footer_label.pack(side="bottom", pady=10)

# -----------------------
root.mainloop()
