"""
app.py - Interface graphique pour tester le modèle
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
import time
import psutil
import threading

sys.path.append(str(Path(__file__).parent))
import config


class ModelTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mine Detection Model Tester")
        self.root.geometry("800x600")

        self.interpreter = None
        self.model_loaded = False

        self.setup_ui()

    def setup_ui(self):
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Charger Modèle", command=self.load_model,
                  bg="#4CAF50", fg="white", font=("Arial", 12), padx=20).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="Sélectionner Image", command=self.select_image,
                  bg="#2196F3", fg="white", font=("Arial", 12), padx=20).pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.root, text="Aucun modèle chargé",
                                     font=("Arial", 10), fg="red")
        self.status_label.pack(pady=5)

        self.image_frame = tk.Frame(self.root, bg="gray", width=400, height=400)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, text="Aucune image", bg="gray")
        self.image_label.pack(expand=True)

        result_frame = tk.LabelFrame(self.root, text="Résultats", font=("Arial", 12, "bold"))
        result_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(result_frame, height=8, font=("Courier", 11))
        self.result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Sélectionner le modèle TFLite",
            filetypes=[("TFLite", "*.tflite")]
        )

        if model_path:
            try:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.model_loaded = True
                self.status_label.config(text=f"✓ Modèle chargé: {Path(model_path).name}",
                                         fg="green")
            except Exception as e:
                self.status_label.config(text=f"✗ Erreur: {e}", fg="red")

    def select_image(self):
        if not self.model_loaded:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "⚠️  Chargez d'abord un modèle!")
            return

        image_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )

        if image_path:
            self.display_image(image_path)
            threading.Thread(target=self.predict_image, args=(image_path,), daemon=True).start()

    def display_image(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

    def predict_image(self, image_path):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "⏳ Prédiction en cours...\n")

        img = Image.open(image_path).convert('RGB')
        img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
        img_array = np.array(img, dtype=np.uint8)
        img_array = np.expand_dims(img_array, axis=0)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Mesures performances
        process = psutil.Process()
        cpu_times_before = process.cpu_times()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        self.interpreter.set_tensor(input_details[0]['index'], img_array)

        start = time.time()
        self.interpreter.invoke()
        latency = (time.time() - start) * 1000

        cpu_times_after = process.cpu_times()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        cpu_time = (cpu_times_after.user - cpu_times_before.user) * 1000  # ms
        mem_delta = mem_after - mem_before

        output = self.interpreter.get_tensor(output_details[0]['index'])

        if output_details[0]['dtype'] == np.uint8:
            output = output.astype(np.float32) / 255.0

        predictions = output[0]

        # Affichage
        self.result_text.delete(1.0, tk.END)

        top_idx = np.argmax(predictions)
        confidence = predictions[top_idx]

        self.result_text.insert(tk.END, "═" * 50 + "\n", "header")
        self.result_text.insert(tk.END, "PRÉDICTION\n", "header")
        self.result_text.insert(tk.END, "═" * 50 + "\n\n", "header")

        if confidence < 0.5:
            self.result_text.insert(tk.END, "⚠️  DANGER - CONFIANCE FAIBLE\n\n", "warning")
        else:
            self.result_text.insert(tk.END, "✓ CONFIANCE ACCEPTABLE\n\n", "success")

        self.result_text.insert(tk.END, f"Classe: {config.CLASSES[top_idx]}\n", "bold")
        self.result_text.insert(tk.END, f"Confiance: {confidence:.1%}\n\n", "bold")

        # Top 3
        self.result_text.insert(tk.END, "Top 3:\n", "header")
        top3_idx = np.argsort(predictions)[-3:][::-1]
        for i, idx in enumerate(top3_idx, 1):
            self.result_text.insert(tk.END, f"  {i}. {config.CLASSES[idx]:15} {predictions[idx]:6.1%}\n")

        # Performances
        self.result_text.insert(tk.END, "\n" + "─" * 50 + "\n", "header")
        self.result_text.insert(tk.END, "PERFORMANCES\n", "header")
        self.result_text.insert(tk.END, "─" * 50 + "\n\n", "header")

        self.result_text.insert(tk.END, f"Latence:      {latency:6.2f} ms\n")
        self.result_text.insert(tk.END, f"Temps CPU:    {cpu_time:6.2f} ms\n")
        self.result_text.insert(tk.END, f"Mémoire:      {mem_after:6.1f} MB ({mem_delta:+.1f} MB)\n")

        # Tags couleur
        self.result_text.tag_config("header", foreground="#1976D2", font=("Courier", 11, "bold"))
        self.result_text.tag_config("bold", font=("Courier", 11, "bold"))
        self.result_text.tag_config("warning", foreground="#FF5722", font=("Courier", 12, "bold"))
        self.result_text.tag_config("success", foreground="#4CAF50", font=("Courier", 12, "bold"))


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelTesterApp(root)
    root.mainloop()