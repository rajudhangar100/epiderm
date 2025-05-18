# EPIDERM GUI - Front Page
# Python 3.12 compatible
# Designed for Raspberry Pi 4B with 3.5" TFT Touch Display

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import sys

class EpidermApp:
    def _init_(self, root):
        self.root = root
        self.root.title("EPIDERM - Early Skin Disease Detection")
        self.root.configure(bg='black')
        self.root.geometry("480x320")  # Typical 3.5" TFT resolution
        self.root.attributes('-fullscreen', False)

        # Load logo if available
        logo_path = "epiderm_logo.png"
        if os.path.exists(logo_path):
            try:
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((120, 120))
                self.logo = ImageTk.PhotoImage(logo_img)
                logo_label = tk.Label(self.root, image=self.logo, bg="black")
                logo_label.pack(pady=5)
            except Exception as e:
                print(f"Error loading logo: {e}")

        # Welcome Message
        welcome_label = tk.Label(
            self.root,
            text="Welcome to EPIDERM\nEarly Skin Disease Detection",
            font=("Helvetica", 14, "bold"),
            fg="white",
            bg="black",
            justify="center"
        )
        welcome_label.pack(pady=5)

        # Button Frame
        button_frame = tk.Frame(self.root, bg='black')
        button_frame.pack(pady=10)

        # Touch-friendly Buttons
        button_style = {
            'font': ("Helvetica", 12),
            'width': 20,
            'height': 2,
            'bd': 0,
            'relief': tk.RAISED
        }

        start_btn = tk.Button(
            button_frame,
            text="Start Scan",
            bg="#4CAF50",
            fg="white",
            command=self.start_scan,
            **button_style
        )
        start_btn.pack(pady=5)

        view_btn = tk.Button(
            button_frame,
            text="View Past Scans",
            bg="#2196F3",
            fg="white",
            command=self.view_scans,
            **button_style
        )
        view_btn.pack(pady=5)

        exit_btn = tk.Button(
            button_frame,
            text="Exit",
            bg="#f44336",
            fg="white",
            command=self.root.quit,
            **button_style
        )
        exit_btn.pack(pady=5)

    def start_scan(self):
        print("Start Scan clicked - Placeholder for scan screen navigation")

    def view_scans(self):
        print("View Past Scans clicked - Placeholder for record screen navigation")

if '__name__' == "_main_":
    root = tk.Tk()
    app = EpidermApp(root)
    root.mainloop()