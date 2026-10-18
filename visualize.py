import time
import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import main as ml

TICK_MS = 30
MAX_SPEED = 200
DEFAULT_SPEED = 10
COLOR_0 = "tab:blue"
COLOR_1 = "tab:orange"

LIGHT_THEME = {
    "bg": "#f0f0f0", "fg": "black", "button": "#e1e1e1", "button_active": "#d0d0d0",
    "field": "white", "trough": "#c8c8c8", "figure": "white", "axes": "white",
    "grid": "#b0b0b0", "line": "black",
}
DARK_THEME = {
    "bg": "#1e1e1e", "fg": "white", "button": "#333333", "button_active": "#454545",
    "field": "#2b2b2b", "trough": "#3a3a3a", "figure": "#1e1e1e", "axes": "#121212",
    "grid": "#505050", "line": "white",
}


def boundary_points(w, b, xlim, ylim):
    """Two endpoints of the line w1*x1 + w2*x2 + b = 0, or None if there is no line yet."""
    if abs(w[0]) < 1e-12 and abs(w[1]) < 1e-12:
        return None
    # Solve for whichever coordinate has the larger weight, so the line never divides by ~0
    if abs(w[1]) >= abs(w[0]):
        x1 = np.array(xlim)
        x2 = -(w[0] * x1 + b) / w[1]
    else:
        x2 = np.array(ylim)
        x1 = -(w[1] * x2 + b) / w[0]
    return x1, x2


class TrainingApp:
    def __init__(self, root):
        self.root = root
        root.title("ML1 - Logistic regression, live training")

        rng = np.random.default_rng(ml.SEED)
        X, y = ml.make_data(rng)
        self.X_train, self.y_train, self.X_test, self.y_test = ml.split(X, y, rng)

        self.running = False
        self.showing_loss = False
        self.dark_mode = False
        self.theme = LIGHT_THEME
        self.pending_epochs = 0.0
        self.last_tick = time.perf_counter()

        self._build_controls()
        self._build_dots_screen()
        self._build_loss_screen()
        self.dots_frame.pack(fill=tk.BOTH, expand=True)

        self.apply_theme()
        self.reset()
        self.root.after(TICK_MS, self.tick)

    # ---------- Layout ----------

    def _build_controls(self):
        bar = tk.Frame(self.root, padx=8, pady=6)
        bar.pack(fill=tk.X)

        self.play_button = tk.Button(bar, text="Play", width=8, command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Step", width=8, command=self.step_button).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Reset", width=8, command=self.reset).pack(side=tk.LEFT, padx=2)

        tk.Label(bar, text="   Speed (epochs/sec):").pack(side=tk.LEFT)
        self.speed = tk.IntVar(value=DEFAULT_SPEED)
        tk.Scale(bar, from_=1, to=MAX_SPEED, orient=tk.HORIZONTAL, length=220,
                 variable=self.speed, showvalue=False).pack(side=tk.LEFT)
        tk.Spinbox(bar, from_=1, to=MAX_SPEED, width=5,
                   textvariable=self.speed).pack(side=tk.LEFT, padx=4)

        self.screen_button = tk.Button(bar, text="Show loss graph", width=16,
                                       command=self.toggle_screen)
        self.screen_button.pack(side=tk.RIGHT, padx=2)
        self.theme_button = tk.Button(bar, text="Dark mode", width=10, command=self.toggle_theme)
        self.theme_button.pack(side=tk.RIGHT, padx=2)

        self.status = tk.Label(self.root, anchor="w", padx=10, font=("Consolas", 10))
        self.status.pack(fill=tk.X)

    def _build_dots_screen(self):
        self.dots_frame = tk.Frame(self.root)
        self.dots_fig = Figure(figsize=(7, 6), dpi=100)
        ax = self.dots_fig.add_subplot()
        self.dots_ax = ax

        for X, y, filled, name in ((self.X_train, self.y_train, True, "train"),
                                   (self.X_test, self.y_test, False, "test")):
            for label, color in ((0, COLOR_0), (1, COLOR_1)):
                pts = X[y == label]
                ax.scatter(pts[:, 0], pts[:, 1],
                           facecolors=color if filled else "none", edgecolors=color,
                           alpha=0.75, label=f"class {label} ({name})")

        all_X = np.vstack([self.X_train, self.X_test])
        margin = 1.0
        self.xlim = (all_X[:, 0].min() - margin, all_X[:, 0].max() + margin)
        self.ylim = (all_X[:, 1].min() - margin, all_X[:, 1].max() + margin)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        self.boundary_line, = ax.plot([], [], "--", linewidth=2, label="decision boundary")
        self.dots_legend = ax.legend(loc="lower right", fontsize=8)

        self.dots_canvas = FigureCanvasTkAgg(self.dots_fig, master=self.dots_frame)
        self.dots_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_loss_screen(self):
        self.loss_frame = tk.Frame(self.root)
        self.loss_fig = Figure(figsize=(7, 6), dpi=100)
        ax = self.loss_fig.add_subplot()
        self.loss_ax = ax

        self.loss_line, = ax.plot([], [])
        ax.set_xlim(0, ml.EPOCHS)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss (BCE)")
        ax.set_title("Loss curve")

        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, master=self.loss_frame)
        self.loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- Training ----------

    def reset(self):
        self.running = False
        self.play_button.config(text="Play")
        self.pending_epochs = 0.0
        self.w = np.zeros(self.X_train.shape[1])
        self.b = 0.0
        self.epoch = 0
        self.losses = []
        self.redraw()

    def finished(self):
        return self.epoch >= ml.EPOCHS

    def step_once(self):
        if self.finished():
            return
        self.w, self.b, loss = ml.train_step(self.X_train, self.y_train,
                                             self.w, self.b, ml.LEARNING_RATE)
        self.losses.append(loss)
        self.epoch += 1

    def step_button(self):
        self.step_once()
        self.redraw()

    def toggle_play(self):
        if self.running:
            self.running = False
        else:
            if self.finished():
                self.reset()
            self.running = True
        self.play_button.config(text="Pause" if self.running else "Play")

    def current_speed(self):
        try:
            return min(max(int(self.speed.get()), 1), MAX_SPEED)
        except (tk.TclError, ValueError):
            return DEFAULT_SPEED

    def tick(self):
        now = time.perf_counter()
        elapsed = now - self.last_tick
        self.last_tick = now
        if self.running:
            # Redrawing takes longer than TICK_MS, so use the real elapsed time to hold the chosen speed
            self.pending_epochs += self.current_speed() * elapsed
            n = int(self.pending_epochs)
            self.pending_epochs -= n
            for _ in range(n):
                self.step_once()
            if n:
                self.redraw()
            if self.finished():
                self.running = False
                self.play_button.config(text="Play")
        self.root.after(TICK_MS, self.tick)

    # ---------- Drawing ----------

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.theme = DARK_THEME if self.dark_mode else LIGHT_THEME
        self.theme_button.config(text="Light mode" if self.dark_mode else "Dark mode")
        self.apply_theme()
        self.redraw()

    def apply_theme(self):
        t = self.theme
        self.root.config(bg=t["bg"])
        self._style_widget(self.root, t)

        for fig, ax in ((self.dots_fig, self.dots_ax), (self.loss_fig, self.loss_ax)):
            fig.set_facecolor(t["figure"])
            ax.set_facecolor(t["axes"])
            ax.tick_params(colors=t["fg"])
            ax.xaxis.label.set_color(t["fg"])
            ax.yaxis.label.set_color(t["fg"])
            ax.title.set_color(t["fg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(t["fg"])

        self.loss_ax.grid(True, color=t["grid"], alpha=0.6)
        self.boundary_line.set_color(t["line"])
        self.dots_legend.get_frame().set_facecolor(t["axes"])
        self.dots_legend.get_frame().set_edgecolor(t["grid"])
        for text in self.dots_legend.get_texts():
            text.set_color(t["fg"])
        # The legend caches its line sample, so it needs the new boundary color too
        self.dots_legend.get_lines()[-1].set_color(t["line"])

        self.dots_canvas.draw_idle()
        self.loss_canvas.draw_idle()

    def _style_widget(self, widget, t):
        kind = widget.winfo_class()
        if kind == "Button":
            widget.config(bg=t["button"], fg=t["fg"], activebackground=t["button_active"],
                          activeforeground=t["fg"])
        elif kind == "Label":
            widget.config(bg=t["bg"], fg=t["fg"])
        elif kind == "Scale":
            widget.config(bg=t["bg"], fg=t["fg"], troughcolor=t["trough"],
                          activebackground=t["button_active"], highlightbackground=t["bg"])
        elif kind == "Spinbox":
            widget.config(bg=t["field"], fg=t["fg"], insertbackground=t["fg"],
                          buttonbackground=t["button"], highlightbackground=t["bg"])
        elif kind in ("Frame", "Canvas"):
            widget.config(bg=t["bg"], highlightbackground=t["bg"])
        for child in widget.winfo_children():
            self._style_widget(child, t)

    def toggle_screen(self):
        self.showing_loss = not self.showing_loss
        if self.showing_loss:
            self.dots_frame.pack_forget()
            self.loss_frame.pack(fill=tk.BOTH, expand=True)
            self.screen_button.config(text="Back to dots")
        else:
            self.loss_frame.pack_forget()
            self.dots_frame.pack(fill=tk.BOTH, expand=True)
            self.screen_button.config(text="Show loss graph")
        self.redraw()

    def redraw(self):
        train_acc = ml.accuracy(self.y_train, ml.predict(self.X_train, self.w, self.b))
        test_acc = ml.accuracy(self.y_test, ml.predict(self.X_test, self.w, self.b))
        loss_text = f"{self.losses[-1]:.4f}" if self.losses else "  -   "
        self.status.config(text=(
            f"Epoch {self.epoch:3d}/{ml.EPOCHS}   loss {loss_text}   "
            f"train acc {train_acc:6.2%}   test acc {test_acc:6.2%}   "
            f"w = [{self.w[0]:.3f}, {self.w[1]:.3f}]   b = {self.b:.3f}"
        ))

        if self.showing_loss:
            self.loss_line.set_data(range(1, len(self.losses) + 1), self.losses)
            top = max(self.losses) * 1.05 if self.losses else 1.0
            self.loss_ax.set_ylim(0, top)
            self.loss_canvas.draw_idle()
        else:
            pts = boundary_points(self.w, self.b, self.xlim, self.ylim)
            self.boundary_line.set_data(*(pts if pts else ([], [])))
            self.dots_ax.set_title(f"Epoch {self.epoch}  (filled = train, hollow = test)",
                                   color=self.theme["fg"])
            self.dots_canvas.draw_idle()


def run():
    root = tk.Tk()
    TrainingApp(root)
    root.mainloop()


if __name__ == "__main__":
    run()
