"""
╔══════════════════════════════════════════════════════════════════════╗
║   RECONNAISSANCE CHIFFRES & LETTRES — Interface CNN Unifié           ║
║                                                                      ║
║                                                                      ║
║   Modèle : CNN 36 classes (0-9 + A-Z)                                ║
║   Utilisation :                                                      ║
║     1. Charger le modèle (best_model.h5 ou model_unified.h5)         ║
║     2. Dessiner un caractère dans la zone de dessin                  ║
║     3. Cliquer sur Reconnaître                                       ║
║     4. Voir le résultat et les probabilités                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # ← désactive oneDNN

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import scipy.ndimage

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

CANVAS_SIZE  = 280          # zone de dessin 280×280 px
IMG_SIZE     = 28           # taille entrée modèle
DIGITS       = [str(i) for i in range(10)]
LETTERS      = [chr(i) for i in range(65, 91)]
CLASSES      = DIGITS + LETTERS   # 36 classes

# Palette couleurs
C = {
    'bg':         '#0f0f1a',
    'panel':      '#1a1a2e',
    'border':     '#2d2d4e',
    'accent':     '#4361ee',
    'accent2':    '#7209b7',
    'success':    '#2ec4b6',
    'warning':    '#ff9f1c',
    'error':      '#e63946',
    'text':       '#e8e8f0',
    'text_dim':   '#6b7280',
    'canvas_bg':  '#ffffff',
    'canvas_ink': '#000000',
    'digit_bar':  '#4361ee',
    'letter_bar': '#e07c24',
}

# ─────────────────────────────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────────────────────────────

def load_model_from_path(path):
    """Charge le modèle Keras depuis un fichier .h5 ou .keras."""
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    model = tf.keras.models.load_model(path)
    print(f"[INFO] Modèle chargé : {path}")
    print(f"[INFO] Input shape   : {model.input_shape}")
    print(f"[INFO] Output shape  : {model.output_shape}")
    return model


# ─────────────────────────────────────────────────────────────────────
# PRÉTRAITEMENT DE L'IMAGE DESSINÉE
# ─────────────────────────────────────────────────────────────────────

def preprocess_canvas(pil_image):
    """
    Transforme l'image dessinée (280×280, fond blanc, tracé noir)
    en array (1, 28, 28, 1) float32 [0-1] prêt pour le modèle.

    Étapes :
    1. Conversion en niveaux de gris
    2. Détection de la bounding box du tracé
    3. Recadrage + padding carré
    4. Redimensionnement à 28×28
    5. Inversion (fond noir, tracé blanc) — convention MNIST
    6. Normalisation [0-1]
    """
    # Niveaux de gris
    gray = pil_image.convert('L')
    arr  = np.array(gray, dtype='float32')

    # Binarisation pour trouver le tracé
    binary = (arr < 200).astype(np.uint8)

    if binary.sum() == 0:
        return None   # canvas vide

    # Bounding box du tracé
    rows = np.where(binary.sum(axis=1) > 0)[0]
    cols = np.where(binary.sum(axis=0) > 0)[0]
    r1, r2 = rows[0], rows[-1]
    c1, c2 = cols[0], cols[-1]

    # Recadrage avec un peu de marge
    pad = 20
    r1 = max(0, r1 - pad)
    r2 = min(arr.shape[0] - 1, r2 + pad)
    c1 = max(0, c1 - pad)
    c2 = min(arr.shape[1] - 1, c2 + pad)

    cropped = gray.crop((c1, r1, c2, r2))

    # Padding carré
    w, h = cropped.size
    s    = max(w, h)
    sq   = Image.new('L', (s, s), color=255)
    sq.paste(cropped, ((s - w) // 2, (s - h) // 2))

    # Redimensionnement 28×28
    resized = sq.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # Normalisation
    arr28 = np.array(resized, dtype='float32') / 255.0

    # Inversion : fond noir (0), tracé blanc (1) — convention MNIST
    arr28 = 1.0 - arr28

    # Légère amélioration du contraste
    arr28 = np.clip(arr28 * 1.5, 0, 1)

    return arr28.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def add_noise_to_array(arr, noise_type='gaussian', level=0.15):
    """Ajoute du bruit à l'array prétraité."""
    if noise_type == 'gaussian':
        return np.clip(arr + np.random.normal(0, level, arr.shape), 0, 1)
    elif noise_type == 'salt_pepper':
        noisy = arr.copy()
        n = int(level * IMG_SIZE * IMG_SIZE)
        for _ in range(n):
            noisy[0, np.random.randint(0, IMG_SIZE),
                     np.random.randint(0, IMG_SIZE), 0] = 1.0
        for _ in range(n):
            noisy[0, np.random.randint(0, IMG_SIZE),
                     np.random.randint(0, IMG_SIZE), 0] = 0.0
        return noisy
    return arr


# ─────────────────────────────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────────────────────────────

def predict(model, arr):
    """
    Retourne le top-5 des prédictions.
    arr : (1, 28, 28, 1) float32
    """
    probs    = model.predict(arr, verbose=0)[0]   # (36,)
    top5_idx = np.argsort(probs)[::-1][:5]
    return probs, top5_idx


# ─────────────────────────────────────────────────────────────────────
# INTERFACE GRAPHIQUE
# ─────────────────────────────────────────────────────────────────────

class App:

    def __init__(self, root):
        self.root  = root
        self.model = None

        self.root.title("CNN — Reconnaissance Chiffres & Lettres | Projet IA M1")
        self.root.configure(bg=C['bg'])
        self.root.resizable(True, True)
        self.root.minsize(900, 620)

        # Canvas PIL
        self.pil_image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        # Variables
        self.drawing   = False
        self.last_x    = None
        self.last_y    = None
        self.brush_var = tk.IntVar(value=14)
        self.noise_var = tk.BooleanVar(value=False)
        self.noise_type_var  = tk.StringVar(value='gaussian')
        self.noise_level_var = tk.DoubleVar(value=0.15)
        self.model_path_var  = tk.StringVar(value='best_model.h5')

        self._build_ui()

        # Essai de chargement automatique
        self._auto_load_model()

    # ── Chargement automatique ─────────────────────────────────────────
    def _auto_load_model(self):
        """Essaie de charger best_model.h5 ou model_unified.h5 automatiquement."""
        for fname in ['best_model.h5', 'model_unified.h5', 'model_unified.keras']:
            if os.path.exists(fname):
                try:
                    self.model = load_model_from_path(fname)
                    self.model_path_var.set(fname)
                    self._set_status(f"✅ Modèle chargé automatiquement : {fname}", C['success'])
                    self._update_model_badge(True, fname)
                    return
                except Exception as e:
                    print(f"[WARN] Impossible de charger {fname} : {e}")
        self._set_status("⚠  Aucun modèle trouvé — cliquez sur 'Charger modèle'", C['warning'])

    # ── Construction UI ────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=C['accent2'], pady=10)
        hdr.pack(fill='x')
        tk.Label(hdr,
                 text="  CNN — Reconnaissance de Caractères  (0-9 + A-Z)",
                 font=('Consolas', 13, 'bold'),
                 bg=C['accent2'], fg='#ffffff').pack(side='left', padx=20)
        tk.Label(hdr,
                 text="  ",
                 font=('Consolas', 9),
                 bg=C['accent2'], fg='#ccd6ff').pack(side='right', padx=20)

        # ── Corps ─────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=C['bg'])
        body.pack(fill='both', expand=True, padx=16, pady=12)

        left  = tk.Frame(body, bg=C['bg'])
        left.pack(side='left', fill='both', expand=False)

        right = tk.Frame(body, bg=C['bg'])
        right.pack(side='left', fill='both', expand=True, padx=(16, 0))

        self._build_left(left)
        self._build_right(right)

        # ── Status bar ────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Prêt")
        self.status_lbl = tk.Label(
            self.root, textvariable=self.status_var,
            bg='#111122', fg=C['text_dim'],
            font=('Consolas', 9), anchor='w', padx=12, pady=5
        )
        self.status_lbl.pack(fill='x', side='bottom')

    # ── Panneau gauche (dessin) ────────────────────────────────────────
    def _build_left(self, parent):

        # Chargement modèle
        load_card = self._card(parent, "◉  Modèle")
        lf = tk.Frame(load_card, bg=C['panel'], padx=10, pady=8)
        lf.pack(fill='x')

        tk.Entry(lf, textvariable=self.model_path_var,
                 width=22, font=('Consolas', 9),
                 bg='#0f0f1a', fg=C['text'],
                 insertbackground=C['text'],
                 relief='flat', bd=0).pack(side='left', padx=(0, 6))

        self._btn(lf, "📂 Parcourir", self._browse_model,
                  C['border'], side='left', padx=(0, 4))
        self._btn(lf, "⚡ Charger",   self._load_model,
                  C['accent'], side='left')

        self.model_badge = tk.Label(load_card,
                                    text="✗  Non chargé",
                                    font=('Consolas', 9, 'bold'),
                                    bg=C['panel'], fg=C['error'],
                                    padx=10, pady=4)
        self.model_badge.pack(anchor='w')

        # Zone de dessin
        draw_card = self._card(parent, "✏  Zone de dessin")

        opts = tk.Frame(draw_card, bg=C['panel'], padx=10, pady=6)
        opts.pack(fill='x')
        tk.Label(opts, text="Pinceau :", bg=C['panel'],
                 fg=C['text_dim'], font=('Consolas', 9)).pack(side='left')
        tk.Scale(opts, from_=6, to=30, orient='horizontal',
                 variable=self.brush_var, length=100,
                 bg=C['panel'], fg=C['text'],
                 troughcolor=C['border'],
                 highlightthickness=0, bd=0,
                 showvalue=True).pack(side='left', padx=(4, 0))

        # Canvas dessin
        cf = tk.Frame(draw_card, bg=C['border'], padx=2, pady=2)
        cf.pack(padx=10, pady=(4, 6))
        self.canvas = tk.Canvas(cf,
                                width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg=C['canvas_bg'], cursor='crosshair',
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind('<ButtonPress-1>',   self._start_draw)
        self.canvas.bind('<B1-Motion>',       self._draw)
        self.canvas.bind('<ButtonRelease-1>', self._stop_draw)

        # Boutons
        btn_row = tk.Frame(draw_card, bg=C['panel'], padx=10)
        btn_row.pack(fill='x', pady=(0, 10))
        self._btn(btn_row, "🔍  Reconnaître", self._predict,
                  C['accent'], side='left', padx=(0, 8))
        self._btn(btn_row, "🗑  Effacer", self._clear,
                  C['error'], side='left')

        # Options bruit
        noise_card = self._card(parent, "🌫  Tester avec bruit")
        nf = tk.Frame(noise_card, bg=C['panel'], padx=10, pady=8)
        nf.pack(fill='x')

        tk.Checkbutton(nf, text="Activer le bruit",
                       variable=self.noise_var,
                       bg=C['panel'], fg=C['text'],
                       selectcolor=C['border'],
                       activebackground=C['panel'],
                       font=('Consolas', 9)).grid(row=0, column=0,
                                                   columnspan=2, sticky='w')

        tk.Label(nf, text="Type :", bg=C['panel'],
                 fg=C['text_dim'], font=('Consolas', 9)).grid(
                     row=1, column=0, sticky='w', pady=(4, 0))
        for i, (val, lbl) in enumerate([('gaussian', 'Gaussien'),
                                         ('salt_pepper', 'Sel & Poivre')]):
            tk.Radiobutton(nf, text=lbl, variable=self.noise_type_var,
                           value=val, bg=C['panel'], fg=C['text'],
                           selectcolor=C['border'],
                           activebackground=C['panel'],
                           font=('Consolas', 9)).grid(
                               row=1, column=i+1, padx=4, pady=(4, 0))

        tk.Label(nf, text="Niveau :", bg=C['panel'],
                 fg=C['text_dim'], font=('Consolas', 9)).grid(
                     row=2, column=0, sticky='w', pady=(4, 0))
        tk.Scale(nf, from_=0.05, to=0.50, resolution=0.05,
                 orient='horizontal', variable=self.noise_level_var,
                 length=160, bg=C['panel'], fg=C['text'],
                 troughcolor=C['border'], highlightthickness=0, bd=0,
                 showvalue=True).grid(row=2, column=1,
                                       columnspan=2, pady=(4, 0))

    # ── Panneau droit (résultats) ──────────────────────────────────────
    def _build_right(self, parent):

        # Résultat principal
        res_card = self._card(parent, "◈  Résultat")
        rf = tk.Frame(res_card, bg=C['panel'], padx=16, pady=12)
        rf.pack(fill='x')

        self.result_char_var = tk.StringVar(value="?")
        self.result_conf_var = tk.StringVar(value="—")
        self.result_type_var = tk.StringVar(value="—")

        tk.Label(rf, textvariable=self.result_char_var,
                 font=('Consolas', 72, 'bold'),
                 bg=C['panel'], fg=C['success'],
                 width=3, anchor='center').pack(side='left')

        info = tk.Frame(rf, bg=C['panel'])
        info.pack(side='left', padx=20)

        tk.Label(info, text="Caractère prédit :",
                 bg=C['panel'], fg=C['text_dim'],
                 font=('Consolas', 9)).pack(anchor='w')
        tk.Label(info, textvariable=self.result_char_var,
                 bg=C['panel'], fg=C['text'],
                 font=('Consolas', 18, 'bold')).pack(anchor='w')

        tk.Label(info, text="Confiance :",
                 bg=C['panel'], fg=C['text_dim'],
                 font=('Consolas', 9)).pack(anchor='w', pady=(8, 0))
        tk.Label(info, textvariable=self.result_conf_var,
                 bg=C['panel'], fg=C['warning'],
                 font=('Consolas', 18, 'bold')).pack(anchor='w')

        tk.Label(info, text="Type :",
                 bg=C['panel'], fg=C['text_dim'],
                 font=('Consolas', 9)).pack(anchor='w', pady=(8, 0))
        tk.Label(info, textvariable=self.result_type_var,
                 bg=C['panel'], fg=C['text'],
                 font=('Consolas', 11)).pack(anchor='w')

        # Aperçu de l'image prétraitée
        preview_card = self._card(parent, "◎  Image envoyée au modèle (28×28)")
        pf = tk.Frame(preview_card, bg=C['panel'], padx=16, pady=8)
        pf.pack(fill='x')
        self.preview_label = tk.Label(pf, bg=C['panel'],
                                       text="(dessinez puis cliquez Reconnaître)",
                                       fg=C['text_dim'],
                                       font=('Consolas', 9))
        self.preview_label.pack(side='left')

        # Top 5 probabilités
        prob_card = self._card(parent, "▦  Top 5 probabilités")
        self.prob_frame = tk.Frame(prob_card, bg=C['panel'], padx=16, pady=8)
        self.prob_frame.pack(fill='both', expand=True)

        # Probabilités par groupe
        group_card = self._card(parent, "◑  Chiffres vs Lettres")
        self.group_frame = tk.Frame(group_card, bg=C['panel'], padx=16, pady=8)
        self.group_frame.pack(fill='x')

    # ── Helpers UI ─────────────────────────────────────────────────────
    def _card(self, parent, title=None):
        outer = tk.Frame(parent, bg=C['border'], padx=1, pady=1)
        outer.pack(fill='x', pady=(0, 10))
        inner = tk.Frame(outer, bg=C['panel'])
        inner.pack(fill='both')
        if title:
            hdr = tk.Frame(inner, bg='#12122a')
            hdr.pack(fill='x')
            tk.Label(hdr, text=title,
                     font=('Consolas', 9, 'bold'),
                     bg='#12122a', fg=C['accent'],
                     pady=5, padx=10, anchor='w').pack(fill='x')
            tk.Frame(inner, bg=C['border'], height=1).pack(fill='x')
        return inner

    def _btn(self, parent, text, cmd, color, **pk):
        side = pk.pop('side', None)
        padx = pk.pop('padx', 0)
        pady = pk.pop('pady', 0)
        b = tk.Button(parent, text=text, command=cmd,
                      bg=color, fg=C['text'],
                      font=('Consolas', 9, 'bold'),
                      relief='flat', padx=10, pady=6,
                      cursor='hand2', bd=0,
                      activebackground=color,
                      activeforeground=C['text'])
        kw = {}
        if side: kw['side'] = side
        if padx: kw['padx'] = padx
        if pady: kw['pady'] = pady
        b.pack(**kw)
        return b

    def _set_status(self, msg, color=None):
        self.status_var.set(msg)
        if color:
            self.status_lbl.config(fg=color)

    def _update_model_badge(self, loaded, name=''):
        if loaded:
            self.model_badge.config(
                text=f"  Chargé : {os.path.basename(name)}",
                fg=C['success'])
        else:
            self.model_badge.config(text="✗  Non chargé", fg=C['error'])

    # ── Dessin ─────────────────────────────────────────────────────────
    def _start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def _draw(self, event):
        if not self.drawing:
            return
        x, y, r = event.x, event.y, self.brush_var.get()
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                fill=C['canvas_ink'],
                                outline=C['canvas_ink'])
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=0)
        self.last_x, self.last_y = x, y

    def _stop_draw(self, event):
        self.drawing = False

    def _clear(self):
        self.canvas.delete('all')
        self.pil_image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)
        self.result_char_var.set("?")
        self.result_conf_var.set("—")
        self.result_type_var.set("—")
        for w in self.prob_frame.winfo_children():
            w.destroy()
        for w in self.group_frame.winfo_children():
            w.destroy()
        self.preview_label.config(
            image='', text="(dessinez puis cliquez Reconnaître)")
        self._set_status("Canvas effacé", C['text_dim'])

    # ── Chargement modèle ──────────────────────────────────────────────
    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Sélectionnez le modèle",
            filetypes=[
                ("Modèles Keras", "*.h5 *.keras"),
                ("Tous les fichiers", "*.*")
            ]
        )
        if path:
            self.model_path_var.set(path)

    def _load_model(self):
        path = self.model_path_var.get().strip()
        if not path:
            messagebox.showwarning("Chemin vide", "Entrez ou sélectionnez un fichier modèle.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Fichier introuvable", f"Fichier non trouvé :\n{path}")
            return
        try:
            self._set_status(f"Chargement du modèle {path}...", C['warning'])
            self.root.update()
            self.model = load_model_from_path(path)
            self._update_model_badge(True, path)
            self._set_status(f" Modèle chargé : {path}", C['success'])
        except Exception as e:
            self._update_model_badge(False)
            messagebox.showerror("Erreur chargement", str(e))

    # ── Prédiction ─────────────────────────────────────────────────────
    def _predict(self):
        if self.model is None:
            messagebox.showwarning("Pas de modèle",
                                   "Chargez d'abord un modèle.")
            return

        arr = preprocess_canvas(self.pil_image)
        if arr is None:
            messagebox.showinfo("Canvas vide",
                                "Dessinez un caractère avant de reconnaître.")
            return

        # Ajout de bruit si activé
        if self.noise_var.get():
            arr = add_noise_to_array(
                arr,
                noise_type=self.noise_type_var.get(),
                level=self.noise_level_var.get()
            )

        # Prédiction
        probs, top5 = predict(self.model, arr)
        best_idx    = top5[0]
        best_char   = CLASSES[best_idx]
        best_conf   = probs[best_idx]

        # Type
        char_type = "Chiffre (0-9)" if best_char in DIGITS else "Lettre (A-Z)"

        # Mise à jour résultat principal
        self.result_char_var.set(best_char)
        self.result_conf_var.set(f"{best_conf*100:.1f}%")
        self.result_type_var.set(char_type)

        # Aperçu image 28×28
        self._show_preview(arr[0, :, :, 0])

        # Top 5
        self._show_top5(probs, top5)

        # Chiffres vs Lettres
        self._show_groups(probs)

        # Status
        color_status = (C['success'] if best_conf > 0.80
                        else C['warning'] if best_conf > 0.55 else C['error'])
        self._set_status(
            f"Résultat : « {best_char} »  |  Confiance : {best_conf*100:.1f}%"
            f"  |  Type : {char_type}",
            color_status
        )

    def _show_preview(self, arr28):
        """Affiche l'image 28×28 envoyée au modèle (agrandie ×4)."""
        # Agrandissement ×4 pour la visualisation
        scale = 4
        img   = Image.fromarray((arr28 * 255).astype(np.uint8), mode='L')
        img   = img.resize((28 * scale, 28 * scale), Image.NEAREST)

        # Inversion pour afficher (fond blanc, tracé noir)
        img_show = ImageOps.invert(img)

        from PIL import ImageTk
        self._preview_tk = ImageTk.PhotoImage(img_show)
        self.preview_label.config(image=self._preview_tk, text='')

    def _show_top5(self, probs, top5_idx):
        """Affiche les barres de probabilité top-5."""
        for w in self.prob_frame.winfo_children():
            w.destroy()

        bar_width = 260

        for rank, idx in enumerate(top5_idx):
            char  = CLASSES[idx]
            prob  = probs[idx]
            color = C['digit_bar'] if char in DIGITS else C['letter_bar']
            if rank == 0:
                color = C['success']

            row = tk.Frame(self.prob_frame, bg=C['panel'])
            row.pack(fill='x', pady=2)

            # Rang
            tk.Label(row, text=f"#{rank+1}",
                     bg=C['panel'], fg=C['text_dim'],
                     font=('Consolas', 8), width=3).pack(side='left')

            # Caractère
            tk.Label(row, text=char,
                     bg=C['panel'], fg=color,
                     font=('Consolas', 11, 'bold'), width=3).pack(side='left')

            # Barre
            bg_bar = tk.Frame(row, bg=C['border'],
                              height=14, width=bar_width)
            bg_bar.pack(side='left', padx=6)
            bg_bar.pack_propagate(False)
            fill_w = max(2, int(prob * bar_width))
            tk.Frame(bg_bar, bg=color, height=14,
                     width=fill_w).pack(side='left', fill='y')

            # Pourcentage
            tk.Label(row, text=f"{prob*100:5.1f}%",
                     bg=C['panel'], fg=C['text'],
                     font=('Consolas', 9), width=7).pack(side='left')

            # Type
            type_lbl = "chiffre" if char in DIGITS else "lettre"
            tk.Label(row, text=type_lbl,
                     bg=C['panel'], fg=C['text_dim'],
                     font=('Consolas', 8)).pack(side='left', padx=4)

    def _show_groups(self, probs):
        """Affiche la somme des probabilités chiffres vs lettres."""
        for w in self.group_frame.winfo_children():
            w.destroy()

        prob_digits  = float(probs[:10].sum())
        prob_letters = float(probs[10:].sum())

        for label, val, color in [
            ("Chiffres (0-9)", prob_digits,  C['digit_bar']),
            ("Lettres  (A-Z)", prob_letters, C['letter_bar']),
        ]:
            row = tk.Frame(self.group_frame, bg=C['panel'])
            row.pack(fill='x', pady=3)

            tk.Label(row, text=label,
                     bg=C['panel'], fg=C['text_dim'],
                     font=('Consolas', 9), width=16,
                     anchor='w').pack(side='left')

            bg_bar = tk.Frame(row, bg=C['border'], height=16, width=200)
            bg_bar.pack(side='left', padx=8)
            bg_bar.pack_propagate(False)
            fill_w = max(2, int(val * 200))
            tk.Frame(bg_bar, bg=color, height=16,
                     width=fill_w).pack(side='left', fill='y')

            tk.Label(row, text=f"{val*100:.1f}%",
                     bg=C['panel'], fg=color,
                     font=('Consolas', 10, 'bold')).pack(side='left', padx=8)


# ─────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    root = tk.Tk()
    app  = App(root)
    root.mainloop()
