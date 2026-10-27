"""
===============================================================
 PROYECTO TIPO A - LABERINTO 10x10 RESUELTO CON A* (A-ESTRELLA)
 VERSIÓN CON INTERFAZ GRÁFICA + ÁRBOL DE DECISIONES
===============================================================

LIBRERÍAS Y TECNOLOGÍAS USADAS (para la documentación)
--------------------------------------------------------------
1. tkinter
   - Viene incluida con Python (NO se instala con pip).
   - Se usa para toda la interfaz: la ventana principal, el
     Canvas donde se dibuja el tablero 10x10, los botones,
     las etiquetas y el cuadro de texto con scroll (log).

2. tkinter.scrolledtext
   - Submódulo de tkinter (tampoco se instala aparte).
   - Da un cuadro de texto con barra de desplazamiento, usado
     para mostrar el log paso a paso (nodo, g, h, f).

3. heapq
   - Módulo estándar de Python.
   - Implementa una cola de prioridad (min-heap): es el corazón
     de A*, ya que siempre saca primero el nodo con menor f.

4. matplotlib
   - Librería EXTERNA, hay que instalarla una sola vez con:
         pip install matplotlib
   - Se usa SOLO para dibujar el árbol de decisiones (árbol de
     expansión) en una ventana aparte.

5. matplotlib.backends.backend_tkagg (FigureCanvasTkAgg)
   - Es el "puente" que permite meter una gráfica de matplotlib
     dentro de una ventana de Tkinter, en vez de abrir una
     ventana flotante aparte de matplotlib.

Instalación necesaria antes de correr este archivo:
    pip install matplotlib

(tkinter y heapq ya vienen con cualquier instalación normal de
Python en Windows/Mac/Linux, no requieren instalación extra)
===============================================================
"""

"""
===============================================================
 PROYECTO TIPO A - LABERINTO 10x10 RESUELTO CON A* (A-ESTRELLA)
 VERSIÓN CON INTERFAZ GRÁFICA Y DISEÑO EXACTO AL DE REFERENCIA
===============================================================
"""

import heapq
import tkinter as tk
from tkinter import scrolledtext, messagebox
from collections import defaultdict

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------------------
# 1. DEFINICIÓN DEL LABERINTO (PROYECTO TIPO A: 10x10)
# ---------------------------------------------------------------
DIM_FILAS = 10
DIM_COLUMNAS = 10

OBSTACULOS = {
    (1, 8),
    (2, 3), (2, 6), (2, 10),
    (3, 1), (3, 3), (3, 8),
    (4, 3), (4, 10),
    (5, 5), (5, 7), (5, 8),
    (6, 3), (6, 4), (6, 6), (6, 10),
    (7, 1), (7, 2), (7, 4),
    (8, 10),
    (9, 3), (9, 4), (9, 5), (9, 7), (9, 8),
}

COSTOS_ESPECIALES = {
    (1, 3): 4, (1, 5): 2,
    (2, 1): 4, (2, 5): 3, (2, 7): 2,
    (3, 4): 2, (3, 6): 4, (3, 7): 3,
    (4, 1): 3, (4, 2): 3, (4, 6): 3, (4, 9): 2,
    (5, 2): 5, (5, 3): 3,
    (6, 5): 2, (6, 8): 4,
    (7, 6): 5, (7, 7): 2,
    (8, 2): 3, (8, 4): 2, (8, 6): 3, (8, 7): 2, (8, 9): 3,
    (10, 3): 3, (10, 5): 3, (10, 9): 2,
}


def obtener_costo_celda(fila, col):
    if (fila, col) in OBSTACULOS:
        return None
    return COSTOS_ESPECIALES.get((fila, col), 1)


def heuristica_manhattan(nodo, meta):
    return abs(nodo[0] - meta[0]) + abs(nodo[1] - meta[1])


# ---------------------------------------------------------------
# 2. ALGORITMO A*
# ---------------------------------------------------------------
def a_estrella(inicio, meta):
    if inicio in OBSTACULOS or meta in OBSTACULOS:
        return None, None, [], {}, "El inicio o la meta están sobre un obstáculo."

    contador = 0
    frontera = []
    h_ini = heuristica_manhattan(inicio, meta)
    heapq.heappush(frontera, (h_ini, h_ini, contador, inicio))

    g_score = {inicio: 0}
    padres = {inicio: None}
    cerrada = set()
    historial = []

    while frontera:
        f_act, h_act, _, actual = heapq.heappop(frontera)

        if actual in cerrada:
            continue
        cerrada.add(actual)
        historial.append((actual, g_score[actual], h_act, f_act))

        if actual == meta:
            camino = []
            curr = meta
            while curr is not None:
                camino.append(curr)
                curr = padres[curr]
            camino.reverse()
            return camino, g_score[meta], historial, padres, "Ruta óptima encontrada."

        for df, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            vf, vc = actual[0] + df, actual[1] + dc
            vecino = (vf, vc)
            if 1 <= vf <= DIM_FILAS and 1 <= vc <= DIM_COLUMNAS:
                costo_paso = obtener_costo_celda(vf, vc)
                if costo_paso is None or vecino in cerrada:
                    continue

                tentativo_g = g_score[actual] + costo_paso
                if vecino not in g_score or tentativo_g < g_score[vecino]:
                    g_score[vecino] = tentativo_g
                    padres[vecino] = actual
                    h_vecino = heuristica_manhattan(vecino, meta)
                    f_vecino = tentativo_g + h_vecino
                    contador += 1
                    heapq.heappush(frontera, (f_vecino, h_vecino, contador, vecino))

    return None, None, historial, padres, "No existe camino posible hacia la meta."


# ---------------------------------------------------------------
# 3. DISPOSICIÓN JERÁRQUICA DEL ÁRBOL
# ---------------------------------------------------------------
def construir_layout_arbol(padres, raiz):
    hijos = defaultdict(list)
    for nodo, padre in padres.items():
        if padre is not None:
            hijos[padre].append(nodo)
    for k in hijos:
        hijos[k].sort()

    posiciones = {}
    contador_x = [0]

    def asignar(nodo, profundidad):
        if not hijos[nodo]:
            x = contador_x[0]
            contador_x[0] += 1
        else:
            xs = [asignar(h, profundidad + 1) for h in hijos[nodo]]
            x = sum(xs) / len(xs)
        posiciones[nodo] = (x, -profundidad)
        return x

    asignar(raiz, 0)
    return posiciones, hijos


# ---------------------------------------------------------------
# 4. COMPONENTE DE BOTÓN SÓLIDO (COMPATIBILIDAD MAC Y WINDOWS)
# ---------------------------------------------------------------
class BotonSolido(tk.Label):
    """Renderiza botones con fondo sólido y bordes nítidos sin que macOS los degrade."""
    def __init__(self, master, text, command, bg, fg="white", font=("Segoe UI", 9, "bold"), padx=12, pady=5, state="normal"):
        super().__init__(master, text=text, bg=bg, fg=fg, font=font, padx=padx, pady=pady, relief="raised", bd=1, cursor="pointinghand")
        self.command = command
        self._bg_orig = bg
        self._fg_orig = fg
        self._state = state
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)
        if state == "disabled":
            self.set_state("disabled")

    def _on_click(self, event):
        if self._state == "normal":
            self.config(relief="sunken")

    def _on_release(self, event):
        if self._state == "normal":
            self.config(relief="raised")
            if self.command:
                self.command()

    def set_state(self, state):
        self._state = state
        if state == "disabled":
            self.config(bg="#e0e0e0", fg="#888888", cursor="arrow", relief="groove")
        else:
            self.config(bg=self._bg_orig, fg=self._fg_orig, cursor="pointinghand", relief="raised")


# ---------------------------------------------------------------
# 5. INTERFAZ GRÁFICA (IDÉNTICA AL MODELO DE REFERENCIA)
# ---------------------------------------------------------------
TAM_CELDA = 45
MARGEN = 35

COLOR_VENTANA = "#f0f2f5"
COLOR_FONDO_LIBRE = "#ffffff"
COLOR_FONDO_COSTO = "#fff3b0"
COLOR_OBSTACULO = "#c0392b"
COLOR_INICIO = "#27ae60"
COLOR_META = "#2980b9"
COLOR_EXPANDIDO = "#aed6f1"
COLOR_CAMINO = "#f39c12"
COLOR_BORDE = "#7f8c8d"


class AppLaberinto:
    def __init__(self, root):
        self.root = root
        self.root.title("A* - Proyecto Tipo A (Laberinto 10x10)")
        self.root.configure(bg=COLOR_VENTANA)

        self.inicio = None
        self.meta = None
        self.modo = "inicio"
        self.historial = []
        self.padres = {}
        self.camino = None
        self.g_score = {}
        self.anim_index = 0

        self._construir_interfaz()
        self._dibujar_tablero()

    def _construir_interfaz(self):
        contenedor = tk.Frame(self.root, bg=COLOR_VENTANA)
        contenedor.pack(padx=12, pady=12)

        ancho = DIM_COLUMNAS * TAM_CELDA + MARGEN
        alto = DIM_FILAS * TAM_CELDA + MARGEN
        self.canvas = tk.Canvas(contenedor, width=ancho, height=alto, bg="#ffffff", highlightthickness=1, highlightbackground="#bdc3c7")
        self.canvas.grid(row=0, column=0, rowspan=10)
        self.canvas.bind("<Button-1>", self._click_canvas)

        panel = tk.Frame(contenedor, bg=COLOR_VENTANA)
        panel.grid(row=0, column=1, sticky="n", padx=(14, 0))

        tk.Label(panel, text="1) Elige qué marcar y haz click en el tablero",
                 font=("Segoe UI", 9, "bold"), bg=COLOR_VENTANA, fg="#111111").pack(anchor="w", pady=(0, 4))

        frame_modo = tk.Frame(panel, bg=COLOR_VENTANA)
        frame_modo.pack(anchor="w", pady=4)

        self.btn_modo_inicio = BotonSolido(frame_modo, text="Marcar Inicio",
                                           command=lambda: self._set_modo("inicio"),
                                           bg=COLOR_INICIO, fg="#ffffff")
        self.btn_modo_inicio.pack(side="left", padx=2)

        self.btn_modo_meta = BotonSolido(frame_modo, text="Marcar Meta",
                                         command=lambda: self._set_modo("meta"),
                                         bg=COLOR_META, fg="#ffffff")
        self.btn_modo_meta.pack(side="left", padx=2)

        self.lbl_estado = tk.Label(panel, text="Modo actual: INICIO", font=("Segoe UI", 9),
                                   bg=COLOR_VENTANA, fg="#333333")
        self.lbl_estado.pack(anchor="w", pady=(2, 10))

        tk.Label(panel, text="2) Ejecuta el algoritmo",
                 font=("Segoe UI", 9, "bold"), bg=COLOR_VENTANA, fg="#111111").pack(anchor="w", pady=(0, 4))

        self.btn_ejecutar = BotonSolido(panel, text="Ejecutar A*", command=self._ejecutar,
                                        bg="#2c3e50", fg="#ffffff", pady=6)
        self.btn_ejecutar.pack(fill="x", pady=3)

        self.btn_reiniciar = BotonSolido(panel, text="Reiniciar", command=self._reiniciar,
                                         bg="#e4e7eb", fg="#222222", pady=5)
        self.btn_reiniciar.pack(fill="x", pady=2)

        self.btn_arbol = BotonSolido(panel, text="Ver Árbol de Decisiones", command=self._mostrar_arbol,
                                     bg="#e4e7eb", fg="#222222", pady=5, state="disabled")
        self.btn_arbol.pack(fill="x", pady=(10, 4))

        tk.Label(panel, text="Log paso a paso (g, h, f):",
                 font=("Segoe UI", 9, "bold"), bg=COLOR_VENTANA, fg="#111111").pack(anchor="w", pady=(8, 2))

        # Cuadro de log exactamente igual: fondo blanco y tipografía oscura
        self.log = scrolledtext.ScrolledText(panel, width=44, height=18, font=("Consolas", 8),
                                             bg="#ffffff", fg="#111111", relief="solid", bd=1)
        self.log.pack()

    def _set_modo(self, modo):
        self.modo = modo
        self.lbl_estado.config(text=f"Modo actual: {modo.upper()}")

    def _coords_celda(self, fila, col):
        x0 = MARGEN + (col - 1) * TAM_CELDA
        y0 = MARGEN + (fila - 1) * TAM_CELDA
        return x0, y0, x0 + TAM_CELDA, y0 + TAM_CELDA

    def _dibujar_tablero(self):
        self.canvas.delete("all")
        for c in range(1, DIM_COLUMNAS + 1):
            x = MARGEN + (c - 1) * TAM_CELDA + TAM_CELDA / 2
            self.canvas.create_text(x, MARGEN / 2, text=str(c), font=("Segoe UI", 8), fill="#333333")

        for f in range(1, DIM_FILAS + 1):
            y = MARGEN + (f - 1) * TAM_CELDA + TAM_CELDA / 2
            self.canvas.create_text(MARGEN / 2, y, text=str(f), font=("Segoe UI", 8), fill="#333333")

        for f in range(1, DIM_FILAS + 1):
            for c in range(1, DIM_COLUMNAS + 1):
                pos = (f, c)
                x0, y0, x1, y1 = self._coords_celda(f, c)
                color = COLOR_FONDO_LIBRE
                texto = ""

                if pos in OBSTACULOS:
                    color = COLOR_OBSTACULO
                    texto = "X"
                elif pos in COSTOS_ESPECIALES:
                    color = COLOR_FONDO_COSTO
                    texto = str(COSTOS_ESPECIALES[pos])

                if pos == self.inicio:
                    color = COLOR_INICIO
                    texto = "S"
                elif pos == self.meta:
                    color = COLOR_META
                    texto = "G"

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=COLOR_BORDE)
                if texto:
                    fg = "white" if (pos == self.inicio or pos == self.meta or pos in OBSTACULOS) else "#333333"
                    self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=texto,
                                            fill=fg, font=("Segoe UI", 9, "bold"))

    def _colorear_celda(self, pos, color):
        f, c = pos
        x0, y0, x1, y1 = self._coords_celda(f, c)
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=COLOR_BORDE)
        texto = ""
        if pos == self.inicio:
            texto = "S"
        elif pos == self.meta:
            texto = "G"
        elif pos in COSTOS_ESPECIALES:
            texto = str(COSTOS_ESPECIALES[pos])
        if texto:
            fg = "white" if (pos == self.inicio or pos == self.meta) else "#333333"
            self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=texto,
                                    fill=fg, font=("Segoe UI", 9, "bold"))

    def _click_canvas(self, event):
        col = int((event.x - MARGEN) // TAM_CELDA) + 1
        fila = int((event.y - MARGEN) // TAM_CELDA) + 1
        if not (1 <= fila <= DIM_FILAS and 1 <= col <= DIM_COLUMNAS):
            return
        pos = (fila, col)
        if pos in OBSTACULOS:
            messagebox.showinfo("Obstáculo", "Esa celda es un obstáculo (X), elige otra.")
            return

        if self.modo == "inicio":
            self.inicio = pos
        else:
            self.meta = pos
        self._dibujar_tablero()

    def _log_linea(self, texto):
        self.log.insert(tk.END, texto + "\n")
        self.log.see(tk.END)

    def _ejecutar(self):
        if self.inicio is None or self.meta is None:
            messagebox.showwarning("Faltan datos", "Primero marca el Inicio y la Meta.")
            return

        self.log.delete("1.0", tk.END)
        self._dibujar_tablero()
        self.btn_arbol.set_state("disabled")

        camino, costo_total, historial, padres, mensaje = a_estrella(self.inicio, self.meta)
        self.camino = camino
        self.historial = historial
        self.padres = padres
        self.g_score = {n: g for n, g, h, f in historial}

        self._log_linea(mensaje)
        self._log_linea(f"Nodos expandidos: {len(historial)}\n")
        self.anim_index = 0
        self._animar_paso()

    def _animar_paso(self):
        if self.anim_index < len(self.historial):
            nodo, g, h, f = self.historial[self.anim_index]
            if nodo not in (self.inicio, self.meta):
                self._colorear_celda(nodo, COLOR_EXPANDIDO)
            self._log_linea(f"Paso {self.anim_index + 1:>3}: nodo={nodo}  g={g}  h={h}  f={f}")
            self.anim_index += 1
            self.root.after(50, self._animar_paso)
        else:
            self._finalizar()

    def _finalizar(self):
        if self.camino:
            for nodo in self.camino:
                if nodo not in (self.inicio, self.meta):
                    self._colorear_celda(nodo, COLOR_CAMINO)
            self._log_linea(f"\nCosto total óptimo g(meta) = {self.g_score[self.meta]}")
            self._log_linea(f"Camino: {self.camino}")
            self.btn_arbol.set_state("normal")
        else:
            self._log_linea("\nNo existe camino posible entre esas dos celdas.")

    def _reiniciar(self):
        self.inicio = None
        self.meta = None
        self.historial = []
        self.padres = {}
        self.camino = None
        self.g_score = {}
        self.log.delete("1.0", tk.END)
        self.btn_arbol.set_state("disabled")
        self._dibujar_tablero()

    def _mostrar_arbol(self):
        if not self.padres:
            return

        posiciones, hijos = construir_layout_arbol(self.padres, self.inicio)
        camino_set = set(self.camino) if self.camino else set()

        ventana = tk.Toplevel(self.root)
        ventana.title("Árbol de decisiones generado por A*")

        n_hojas = max(1, sum(1 for n in self.padres if not hijos[n]))
        fig_ancho = max(8, n_hojas * 0.6)
        fig = plt.Figure(figsize=(fig_ancho, 6), dpi=100)
        ax = fig.add_subplot(111)

        for nodo, padre in self.padres.items():
            if padre is not None:
                x0, y0 = posiciones[padre]
                x1, y1 = posiciones[nodo]
                en_camino = nodo in camino_set and padre in camino_set
                ax.plot([x0, x1], [y0, y1],
                        color="#e67e22" if en_camino else "#bdc3c7",
                        linewidth=2.2 if en_camino else 0.9,
                        zorder=1)

        for nodo, (x, y) in posiciones.items():
            if nodo == self.inicio:
                color = COLOR_INICIO
            elif nodo == self.meta:
                color = COLOR_META
            elif nodo in camino_set:
                color = COLOR_CAMINO
            else:
                color = "#aed6f1"
            ax.scatter([x], [y], s=380, color=color, edgecolors="#555", zorder=2)
            g_val = self.g_score.get(nodo, "?")
            ax.text(x, y, f"{nodo}\ng={g_val}", ha="center", va="center",
                    fontsize=6.2, zorder=3)

        ax.set_title("Árbol de decisiones (expansión real de A*)\n"
                     "Verde=inicio  Azul=meta  Naranja=camino óptimo", fontsize=10)
        ax.axis("off")
        fig.tight_layout()

        lienzo = FigureCanvasTkAgg(fig, master=ventana)
        lienzo.draw()
        lienzo.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    ventana_raiz = tk.Tk()
    app = AppLaberinto(ventana_raiz)
    ventana_raiz.mainloop()