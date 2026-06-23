"""
CodeAlpha Internship - Task 3: Music Generation with AI
========================================================
Generates original MIDI music using a trained LSTM neural network.

Pipeline:
  1. Built-in seed sequences (Classical / Jazz / Blues styles)
  2. LSTM model learns note-to-note patterns
  3. Model generates a new note sequence
  4. Sequence is written to a MIDI file
  5. GUI lets user pick style, sequence length, and temperature

Features:
  - Three music styles: Classical, Jazz, Blues
  - Adjustable generation length (32–256 notes)
  - Temperature slider to control creativity / randomness
  - Real-time training progress bar
  - Save MIDI file to disk
  - Play MIDI directly from the app (via pygame or system player)

Requirements:
    pip install numpy tensorflow mido pygame
"""

import os
import sys
import time
import random
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ── Third-party imports ───────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    raise SystemExit("Missing: pip install numpy")

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    raise SystemExit("Missing: pip install mido")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
    from tensorflow.keras.utils import to_categorical
    TF_OK = True
except ImportError:
    TF_OK = False

# Optional MIDI playback
try:
    import pygame
    pygame.mixer.init()
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

# ── Seed sequences per style ──────────────────────────────────────────────────
# Notes encoded as MIDI pitch numbers (0-127)
STYLES = {
    "Classical": {
        "description": "Bach-inspired counterpoint patterns",
        "tempo": 500000,   # 120 BPM
        "seed": [
            60, 64, 67, 72, 67, 64, 60, 62, 65, 69, 74, 69, 65, 62,
            60, 55, 57, 60, 62, 64, 67, 69, 71, 72, 71, 69, 67, 65,
            64, 62, 60, 59, 57, 55, 53, 52, 53, 55, 57, 59, 60, 64,
            67, 71, 72, 71, 69, 67, 64, 62, 60, 57, 55, 53, 52, 50,
            48, 50, 52, 53, 55, 57, 60, 62, 64, 65, 67, 69, 71, 72,
        ],
        "velocity": 80,
        "duration": 480,
    },
    "Jazz": {
        "description": "Bebop-style chromatic runs and extensions",
        "tempo": 600000,   # 100 BPM
        "seed": [
            60, 63, 65, 66, 68, 70, 72, 70, 68, 66, 65, 63, 60, 58,
            56, 55, 56, 58, 60, 63, 65, 67, 68, 70, 72, 75, 77, 75,
            72, 70, 68, 67, 65, 63, 60, 58, 55, 53, 55, 58, 60, 62,
            63, 65, 67, 70, 72, 73, 75, 77, 80, 77, 75, 73, 72, 70,
            68, 67, 65, 63, 62, 60, 58, 57, 55, 53, 52, 50, 48, 50,
        ],
        "velocity": 90,
        "duration": 240,
    },
    "Blues": {
        "description": "12-bar blues pentatonic scale patterns",
        "tempo": 666666,   # 90 BPM
        "seed": [
            45, 48, 50, 51, 53, 57, 60, 57, 53, 51, 50, 48,
            45, 48, 50, 53, 57, 58, 60, 58, 57, 53, 50, 48,
            45, 50, 53, 57, 60, 63, 65, 63, 60, 57, 53, 50,
            48, 50, 53, 57, 60, 62, 63, 62, 60, 57, 53, 50,
            45, 48, 53, 57, 60, 62, 63, 60, 57, 53, 48, 45,
            48, 53, 57, 60, 63, 65, 67, 65, 63, 60, 57, 53,
        ],
        "velocity": 100,
        "duration": 320,
    },
}

VOCAB_SIZE = 128   # Full MIDI note range


# ── LSTM Model ────────────────────────────────────────────────────────────────
def build_model(seq_len: int, vocab: int = VOCAB_SIZE) -> "tf.keras.Model":
    model = Sequential([
        Embedding(vocab, 64, input_length=seq_len),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.2),
        Dense(vocab, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def prepare_sequences(notes: list, seq_len: int = 32):
    """Convert note list to (X, y) training pairs."""
    X, y = [], []
    for i in range(len(notes) - seq_len):
        X.append(notes[i: i + seq_len])
        y.append(notes[i + seq_len])
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=VOCAB_SIZE)
    return X, y


def sample_note(probabilities: np.ndarray, temperature: float = 1.0) -> int:
    """Sample from distribution with temperature scaling."""
    probs = np.log(probabilities + 1e-8) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))
    return int(np.random.choice(len(probs), p=probs))


def generate_notes(model, seed: list, length: int,
                   seq_len: int, temperature: float) -> list:
    generated = list(seed[-seq_len:])
    result = []
    for _ in range(length):
        inp = np.array([generated[-seq_len:]])
        probs = model.predict(inp, verbose=0)[0]
        note = sample_note(probs, temperature)
        result.append(note)
        generated.append(note)
    return result


# ── MIDI writer ───────────────────────────────────────────────────────────────
def notes_to_midi(notes: list, style_cfg: dict, output_path: str):
    midi = MidiFile(type=0, ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)

    track.append(MetaMessage("set_tempo", tempo=style_cfg["tempo"], time=0))
    track.append(MetaMessage("track_name", name=f"CodeAlpha_AI_Music", time=0))

    vel  = style_cfg["velocity"]
    dur  = style_cfg["duration"]

    for note in notes:
        note = max(0, min(127, int(note)))
        track.append(Message("note_on",  note=note, velocity=vel, time=0))
        track.append(Message("note_off", note=note, velocity=0,   time=dur))

    midi.save(output_path)


# ── Fallback: rule-based generation (no TensorFlow) ──────────────────────────
def rule_based_generate(seed: list, length: int, temperature: float) -> list:
    """
    Markov-chain-style generation when TensorFlow is unavailable.
    Builds a bigram transition table from the seed.
    """
    transitions: dict = {}
    for i in range(len(seed) - 1):
        n = seed[i]
        nxt = seed[i + 1]
        transitions.setdefault(n, []).append(nxt)

    result = []
    current = seed[-1]
    for _ in range(length):
        nexts = transitions.get(current, seed)
        # Apply temperature: low temp → deterministic, high temp → random
        if temperature < 0.5:
            current = max(set(nexts), key=nexts.count)
        else:
            current = random.choice(nexts)
        result.append(current)
    return result


# ── GUI ───────────────────────────────────────────────────────────────────────
class MusicGenApp(tk.Tk):
    BG      = "#0d1117"
    PANEL   = "#161b22"
    ACCENT  = "#58a6ff"
    GREEN   = "#3fb950"
    YELLOW  = "#d29922"
    TEXT    = "#c9d1d9"
    MUTED   = "#6e7681"
    BTN_FG  = "#ffffff"

    SEQ_LEN = 32    # context window for LSTM

    def __init__(self):
        super().__init__()
        self.title("🎵 AI Music Generator — CodeAlpha Task 3")
        self.geometry("720x660")
        self.resizable(False, False)
        self.configure(bg=self.BG)
        self.model = None
        self.last_midi_path = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=self.PANEL, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎵  AI Music Generator",
                 font=("Segoe UI", 18, "bold"),
                 fg=self.ACCENT, bg=self.PANEL).pack()
        tk.Label(hdr,
                 text="LSTM Neural Network  •  MIDI Output  •  CodeAlpha Internship",
                 font=("Segoe UI", 9), fg=self.MUTED, bg=self.PANEL).pack()

        # ── Settings panel ────────────────────────────────────────────────────
        settings = tk.LabelFrame(self, text="  Generation Settings  ",
                                 font=("Segoe UI", 10, "bold"),
                                 fg=self.TEXT, bg=self.PANEL,
                                 relief="flat", bd=2)
        settings.pack(fill="x", padx=25, pady=18)

        # Style
        row1 = tk.Frame(settings, bg=self.PANEL)
        row1.pack(fill="x", padx=15, pady=8)
        tk.Label(row1, text="Music Style:", width=16, anchor="w",
                 font=("Segoe UI", 10), fg=self.TEXT, bg=self.PANEL).pack(side="left")
        self.style_var = tk.StringVar(value="Classical")
        for style in STYLES:
            rb = tk.Radiobutton(row1, text=style, variable=self.style_var,
                                value=style, font=("Segoe UI", 10),
                                fg=self.TEXT, bg=self.PANEL,
                                selectcolor=self.BG,
                                activebackground=self.PANEL,
                                command=self._update_desc)
            rb.pack(side="left", padx=10)
        self.desc_lbl = tk.Label(settings,
                                 text=STYLES["Classical"]["description"],
                                 font=("Segoe UI", 9, "italic"),
                                 fg=self.MUTED, bg=self.PANEL)
        self.desc_lbl.pack(padx=15, anchor="w")

        # Note length
        row2 = tk.Frame(settings, bg=self.PANEL)
        row2.pack(fill="x", padx=15, pady=10)
        tk.Label(row2, text="Notes to Generate:", width=16, anchor="w",
                 font=("Segoe UI", 10), fg=self.TEXT, bg=self.PANEL).pack(side="left")
        self.notes_var = tk.IntVar(value=64)
        scale = tk.Scale(row2, from_=32, to=256, orient="horizontal",
                         variable=self.notes_var, length=300,
                         bg=self.PANEL, fg=self.TEXT,
                         troughcolor="#21262d",
                         highlightthickness=0, bd=0,
                         sliderrelief="flat")
        scale.pack(side="left")
        self.notes_lbl = tk.Label(row2, textvariable=self.notes_var,
                                  width=4, font=("Segoe UI", 10, "bold"),
                                  fg=self.ACCENT, bg=self.PANEL)
        self.notes_lbl.pack(side="left", padx=5)

        # Temperature
        row3 = tk.Frame(settings, bg=self.PANEL)
        row3.pack(fill="x", padx=15, pady=(0, 12))
        tk.Label(row3, text="Creativity (Temp):", width=16, anchor="w",
                 font=("Segoe UI", 10), fg=self.TEXT, bg=self.PANEL).pack(side="left")
        self.temp_var = tk.DoubleVar(value=0.8)
        t_scale = tk.Scale(row3, from_=0.1, to=2.0, resolution=0.1,
                           orient="horizontal", variable=self.temp_var,
                           length=300, bg=self.PANEL, fg=self.TEXT,
                           troughcolor="#21262d",
                           highlightthickness=0, bd=0,
                           sliderrelief="flat")
        t_scale.pack(side="left")
        tk.Label(row3, textvariable=self.temp_var,
                 width=4, font=("Segoe UI", 10, "bold"),
                 fg=self.YELLOW, bg=self.PANEL).pack(side="left", padx=5)

        # Training epochs (only visible when TF available)
        if TF_OK:
            row4 = tk.Frame(settings, bg=self.PANEL)
            row4.pack(fill="x", padx=15, pady=(0, 12))
            tk.Label(row4, text="Training Epochs:", width=16, anchor="w",
                     font=("Segoe UI", 10), fg=self.TEXT, bg=self.PANEL).pack(side="left")
            self.epochs_var = tk.IntVar(value=30)
            for val, lbl in [(10, "Fast"), (30, "Balanced"), (60, "Quality")]:
                rb = tk.Radiobutton(row4, text=f"{lbl} ({val})",
                                    variable=self.epochs_var, value=val,
                                    font=("Segoe UI", 10),
                                    fg=self.TEXT, bg=self.PANEL,
                                    selectcolor=self.BG,
                                    activebackground=self.PANEL)
                rb.pack(side="left", padx=8)

        # ── Progress / log ────────────────────────────────────────────────────
        log_frame = tk.LabelFrame(self, text="  Log  ",
                                  font=("Segoe UI", 10, "bold"),
                                  fg=self.TEXT, bg=self.PANEL, relief="flat")
        log_frame.pack(fill="x", padx=25, pady=(0, 10))

        self.log_text = tk.Text(log_frame, height=6,
                                font=("Consolas", 9),
                                bg="#0d1117", fg=self.TEXT,
                                relief="flat", state="disabled",
                                padx=8, pady=6)
        self.log_text.pack(fill="x", padx=6, pady=6)

        self.progress = ttk.Progressbar(log_frame, mode="determinate",
                                        maximum=100, value=0)
        self.progress.pack(fill="x", padx=6, pady=(0, 8))

        # ── Action buttons ────────────────────────────────────────────────────
        btn_row = tk.Frame(self, bg=self.BG)
        btn_row.pack(pady=10)

        self.gen_btn = tk.Button(
            btn_row, text="🎼  Generate Music",
            font=("Segoe UI", 12, "bold"),
            bg=self.ACCENT, fg=self.BTN_FG, relief="flat",
            cursor="hand2", padx=20, pady=10,
            command=self._start_generation)
        self.gen_btn.pack(side="left", padx=8)

        self.save_btn = tk.Button(
            btn_row, text="💾  Save MIDI",
            font=("Segoe UI", 12),
            bg="#21262d", fg=self.TEXT, relief="flat",
            cursor="hand2", padx=20, pady=10,
            state="disabled",
            command=self._save_midi)
        self.save_btn.pack(side="left", padx=8)

        if PYGAME_OK:
            self.play_btn = tk.Button(
                btn_row, text="▶  Play",
                font=("Segoe UI", 12),
                bg=self.GREEN, fg=self.BTN_FG, relief="flat",
                cursor="hand2", padx=20, pady=10,
                state="disabled",
                command=self._play_midi)
            self.play_btn.pack(side="left", padx=8)

        # Status
        self.status_var = tk.StringVar(value="Ready — configure settings and click Generate!")
        tk.Label(self, textvariable=self.status_var,
                 font=("Segoe UI", 9), fg=self.MUTED, bg=self.BG).pack(pady=(0, 8))

    def _update_desc(self):
        style = self.style_var.get()
        self.desc_lbl.config(text=STYLES[style]["description"])

    def _log(self, msg: str):
        self.log_text.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # ── Generation pipeline ───────────────────────────────────────────────────
    def _start_generation(self):
        self.gen_btn.config(state="disabled", text="⏳ Working…")
        self.save_btn.config(state="disabled")
        if PYGAME_OK:
            self.play_btn.config(state="disabled")
        self.progress["value"] = 0
        threading.Thread(target=self._generation_worker, daemon=True).start()

    def _generation_worker(self):
        style_name = self.style_var.get()
        style_cfg  = STYLES[style_name]
        n_notes    = self.notes_var.get()
        temperature = self.temp_var.get()
        seed       = style_cfg["seed"]

        self.after(0, self._log, f"Selected style: {style_name}")
        self.after(0, self._log, f"Seed sequence length: {len(seed)} notes")

        if TF_OK:
            epochs = self.epochs_var.get()
            self.after(0, self._log, "Preparing training sequences…")
            X, y = prepare_sequences(seed * 6, self.SEQ_LEN)
            self.after(0, self._log, f"Training LSTM model for {epochs} epochs…")

            model = build_model(self.SEQ_LEN)

            # Custom callback to update progress bar
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(cb_self):
                    super().__init__()
                    cb_self.epoch = 0

                def on_epoch_end(cb_self, epoch, logs=None):
                    cb_self.epoch = epoch + 1
                    pct = int((epoch + 1) / epochs * 70)
                    self.after(0, lambda p=pct: setattr(self.progress, "value", p) or
                               self.progress.config(value=p))
                    loss = logs.get("loss", 0)
                    if (epoch + 1) % 10 == 0:
                        self.after(0, self._log,
                                   f"  Epoch {epoch+1}/{epochs}  loss={loss:.4f}")

            model.fit(X, y, epochs=epochs, batch_size=64,
                      verbose=0, callbacks=[ProgressCallback()])
            self.model = model

            self.after(0, self._log, "Generating notes with LSTM…")
            generated = generate_notes(model, seed, n_notes,
                                       self.SEQ_LEN, temperature)
        else:
            self.after(0, self._log,
                       "⚠ TensorFlow not found — using Markov chain generation.")
            generated = rule_based_generate(seed, n_notes, temperature)

        self.after(0, lambda: self.progress.config(value=85))
        self.after(0, self._log, f"Generated {len(generated)} notes.")

        # Write MIDI
        out_path = os.path.join(
            os.path.expanduser("~"),
            f"CodeAlpha_{style_name}_music.mid"
        )
        self.after(0, self._log, "Writing MIDI file…")
        notes_to_midi(generated, style_cfg, out_path)
        self.last_midi_path = out_path

        self.after(0, lambda: self.progress.config(value=100))
        self.after(0, self._log, f"✅ MIDI saved: {out_path}")
        self.after(0, self._on_generation_done, out_path)

    def _on_generation_done(self, path: str):
        self.gen_btn.config(state="normal", text="🎼  Generate Music")
        self.save_btn.config(state="normal")
        if PYGAME_OK:
            self.play_btn.config(state="normal")
        self.status_var.set(f"✅ Music generated! File: {os.path.basename(path)}")
        messagebox.showinfo(
            "Generation Complete",
            f"MIDI file saved to:\n{path}\n\n"
            "Use any MIDI player (e.g., VLC, Windows Media Player, GarageBand) to listen."
        )

    def _save_midi(self):
        if not self.last_midi_path or not os.path.exists(self.last_midi_path):
            messagebox.showwarning("No File", "Generate music first.")
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".mid",
            filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")],
            initialfile=os.path.basename(self.last_midi_path)
        )
        if dest:
            import shutil
            shutil.copy2(self.last_midi_path, dest)
            self._log(f"Saved copy to: {dest}")

    def _play_midi(self):
        if not self.last_midi_path or not os.path.exists(self.last_midi_path):
            messagebox.showwarning("No File", "Generate music first.")
            return
        try:
            pygame.mixer.music.load(self.last_midi_path)
            pygame.mixer.music.play()
            self.status_var.set("▶ Playing MIDI…")
            self._log("Playing MIDI via pygame…")
        except Exception as exc:
            messagebox.showerror("Playback Error", str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = MusicGenApp()
    app.mainloop()
