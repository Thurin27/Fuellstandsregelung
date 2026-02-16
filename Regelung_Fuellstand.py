import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import math
    from collections import deque
    import altair as alt
    import pandas as pd

    return alt, deque, math, mo, np, pd


@app.function
def regelstrecke(t, x, y, k, Q_ab, tau):
    """
    Modelliert die Regelstrecke als PT1-Glied (Becken mit Zu- und Abfluss).

    Parameter:
        t   : Zeit (h)
        x   : Aktueller Füllstand (m)
        y   : Zufluss / Stellgröße (m³/h)
        k   : Verstärkung = 1 / Beckenfläche (1/m²)
        Q_ab: Abfluss (m³/h)
        tau : Zeitkonstante des PT1-Glieds (h)

    Rückgabe:
        Änderungsrate dx/dt (m/h)
    """
    return (k * (y - Q_ab)) / tau


@app.class_definition
class Regler:
    """
    PID-Regler mit Anti-Windup (Clamping) und Ableitungsfilter.

    Der Regler gibt einen unbegrenzten Wert y_R aus.
    Die Begrenzung auf physikalische Stellgrößen-Grenzen erfolgt
    extern in der Simulationsschleife; der Regler bekommt über
    die Methode anti_windup() Rückmeldung, ob begrenzt wurde.
    """

    def __init__(self, k_p, k_i=0, k_d=0, T_f=0.3):
        """
        Parameter:
            k_p : Proportionalverstärkung
            k_i : Integralverstärkung
            k_d : Differentialverstärkung
            T_f : Filterzeitkonstante für D-Anteil (h), 0 = kein Filter
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.T_f = T_f
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_d_filtered = 0.0
        self._last_error = 0.0
        self._last_dt = 0.0

    def compute(self, error, dt):
        """
        Berechnet den PID-Reglerausgang (unbegrenzt).

        Parameter:
            error : Regelabweichung (w - x)
            dt    : Zeitschritt (h)

        Rückgabe:
            y_R : Reglerausgang (unbegrenzt, dimensionslos)
        """
        self._last_error = error
        self._last_dt = dt

        # P-Anteil
        P = self.k_p * error

        # I-Anteil (vorläufig aufintegrieren)
        self.integral += error * dt
        I = self.k_i * self.integral

        # D-Anteil mit PT1-Filter
        if dt > 0:
            raw_d = (error - self.prev_error) / dt
            if self.T_f > 0:
                alpha = dt / (self.T_f + dt)
                d_filtered = self.prev_d_filtered + alpha * (raw_d - self.prev_d_filtered)
            else:
                d_filtered = raw_d
            self.prev_d_filtered = d_filtered
        else:
            d_filtered = 0.0

        D = self.k_d * d_filtered
        self.prev_error = error

        return P + I + D

    def anti_windup(self, was_clamped):
        """
        Anti-Windup (Clamping): Macht die letzte Integration rückgängig,
        wenn die Stellgröße extern begrenzt wurde.

        Parameter:
            was_clamped : True wenn die physikalische Stellgröße begrenzt wurde
        """
        if was_clamped:
            self.integral -= self._last_error * self._last_dt


@app.cell
def _(alt, deque, math, mo, np, pd, slh0, slqab):
    # ═══════════════════════════════════════════════════════════
    # Feste erweiterte Parameter (nicht durch Schüler einstellbar)
    # ═══════════════════════════════════════════════════════════
    DY_MAX = 40.0    # Max. Stellgrößenänderung (m³/h pro h) – Rate Limiter
    T_F = 0.3        # Filterzeitkonstante D-Anteil (h)
    T_DEAD = 0.05    # Totzeit (h) – kleine Verzögerung für Realismus

    def simulate(
        k_p,
        k_i,
        k_d,
        k=1.0,
        k_Q=30.0,
        tau=1.5,
        t_end=3,
        dt=0.001,
        setpoint=3.0,
        h_max=6.0,
        h_min=0.5,
    ):
        """
        Simuliert die Füllstandsregelung eines Rundbeckens.

        Stellgrößen-Logik:
            y_phys = k_Q + y_R * k_Q
            → y_R = 0  : Pumpe läuft im Arbeitspunkt (k_Q m³/h)
            → y_R > 0  : Pumpe läuft schneller
            → y_R < 0  : Pumpe läuft langsamer (bis aus)
            Begrenzung: 0 ≤ y_phys ≤ 3·k_Q
        """
        t_vals = np.arange(0, t_end, dt)
        x_vals = []
        y_vals = []
        error_vals = []
        qab_vals = []

        # Physikalische Stellgrößen-Grenzen
        Y_MIN = 0.0
        Y_MAX = 3 * k_Q

        # Regler initialisieren
        regler = Regler(k_p, k_i, k_d, T_f=T_F)

        x = slh0.value  # Startwert Füllstand

        # Totzeit-Ringpuffer
        dead_steps = max(1, int(T_DEAD / dt)) if T_DEAD > 0 else 0
        y_buffer = deque([k_Q] * (dead_steps + 1), maxlen=dead_steps + 1)

        # Rate Limiter: vorheriger Stellwert
        y_prev = k_Q

        for t in t_vals:
            error = setpoint - x

            # ── PID-Regler (gibt unbegrenzten Wert y_R aus) ──
            y_R = regler.compute(error, dt)

            # ── Stellgröße: Arbeitspunkt + Regler-Korrektur ──
            y_raw = k_Q + y_R * k_Q

            # ── Rate Limiter ──
            max_change = DY_MAX * dt
            dy = y_raw - y_prev
            if abs(dy) > max_change:
                y_raw = y_prev + max_change * (1 if dy > 0 else -1)

            # ── Stellgrößenbegrenzung (Pumpe: 0 bis 3·k_Q) ──
            y_clamped = max(Y_MIN, min(y_raw, Y_MAX))
            was_clamped = abs(y_raw - y_clamped) > 1e-9
            regler.anti_windup(was_clamped)

            y_prev = y_clamped

            # ── Totzeit ──
            y_buffer.append(y_clamped)
            y_out = y_buffer[0] if dead_steps > 0 else y_clamped

            # ── Abfluss (Torricelli) ──
            Q_ab = math.sqrt(max(0, 2 * 9.81 * x)) * slqab.value / 15

            # ── Regelstrecke (PT1) ──
            dxdt = regelstrecke(t, x, y_out, k, Q_ab, tau)
            x += dxdt * dt
            x = max(0, min(x, 8))

            # Werte speichern
            x_vals.append(x)
            y_vals.append(y_out)
            error_vals.append(error)
            qab_vals.append(Q_ab)

        # ── Downsampling (~2000 Punkte) ──
        n = len(t_vals)
        step = max(1, n // 2000)
        idx = range(0, n, step)

        data = pd.DataFrame({
            "Zeit (h)": t_vals[::step],
            "Füllstand (m)": [x_vals[i] for i in idx],
            "Zufluss (m³/h)": [y_vals[i] for i in idx],
            "Regelabweichung (m)": [error_vals[i] for i in idx],
            "Abfluss (m³/h)": [qab_vals[i] for i in idx],
        })

        # ═══════════════════════════════════════════════════════
        # Diagramme
        # ═══════════════════════════════════════════════════════
        COL_IST = "#1f77b4"
        COL_SOLL = "#d62728"
        COL_ZU = "#2ca02c"
        COL_AB = "#9467bd"
        COL_ERR = "#ff7f0e"
        COL_GRENZE = "#aaaaaa"

        # ── Chart 1: Füllstand mit Sollwert und Grenzlinien ──
        y_domain_max = max(h_max + 0.5, 7)

        line_ist = (
            alt.Chart(data)
            .mark_line(color=COL_IST, strokeWidth=2)
            .encode(
                x=alt.X("Zeit (h):Q", title="Zeit (h)"),
                y=alt.Y("Füllstand (m):Q", title="Füllstand (m)",
                         scale=alt.Scale(domain=[0, y_domain_max])),
            )
        )

        rule_soll = (
            alt.Chart(pd.DataFrame({"y": [setpoint]}))
            .mark_rule(color=COL_SOLL, strokeDash=[6, 4], strokeWidth=2)
            .encode(y="y:Q")
        )
        txt_soll = (
            alt.Chart(pd.DataFrame({
                "x": [t_end * 0.02], "y": [setpoint + 0.2],
                "t": [f"Sollwert {setpoint} m"]}))
            .mark_text(color=COL_SOLL, fontSize=11, fontWeight="bold", align="left")
            .encode(x="x:Q", y="y:Q", text="t:N")
        )

        rule_max = (
            alt.Chart(pd.DataFrame({"y": [h_max]}))
            .mark_rule(color=COL_GRENZE, strokeDash=[2, 2])
            .encode(y="y:Q")
        )
        txt_max = (
            alt.Chart(pd.DataFrame({
                "x": [t_end * 0.02], "y": [h_max + 0.2],
                "t": [f"Max {h_max} m"]}))
            .mark_text(color=COL_GRENZE, fontSize=10, align="left")
            .encode(x="x:Q", y="y:Q", text="t:N")
        )

        rule_min = (
            alt.Chart(pd.DataFrame({"y": [h_min]}))
            .mark_rule(color=COL_GRENZE, strokeDash=[2, 2])
            .encode(y="y:Q")
        )
        txt_min = (
            alt.Chart(pd.DataFrame({
                "x": [t_end * 0.02], "y": [h_min - 0.25],
                "t": [f"Min {h_min} m"]}))
            .mark_text(color=COL_GRENZE, fontSize=10, align="left")
            .encode(x="x:Q", y="y:Q", text="t:N")
        )

        chart1 = (
            (line_ist + rule_soll + txt_soll + rule_max + txt_max + rule_min + txt_min)
            .properties(width=700, height=280, title="Füllstandsverlauf")
            .configure_title(fontSize=14, anchor="start")
        )

        # ── Chart 2: Zufluss + Abfluss ──
        data_flow = pd.melt(
            data,
            id_vars=["Zeit (h)"],
            value_vars=["Zufluss (m³/h)", "Abfluss (m³/h)"],
            var_name="Größe",
            value_name="Volumenstrom (m³/h)",
        )

        chart2 = (
            alt.Chart(data_flow)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Zeit (h):Q", title="Zeit (h)"),
                y=alt.Y("Volumenstrom (m³/h):Q", title="Volumenstrom (m³/h)"),
                color=alt.Color("Größe:N",
                    scale=alt.Scale(
                        domain=["Zufluss (m³/h)", "Abfluss (m³/h)"],
                        range=[COL_ZU, COL_AB]),
                    legend=alt.Legend(title=None, orient="top")),
                strokeDash=alt.StrokeDash("Größe:N",
                    scale=alt.Scale(
                        domain=["Zufluss (m³/h)", "Abfluss (m³/h)"],
                        range=[[0], [6, 4]]),
                    legend=None),
            )
            .properties(width=700, height=220, title="Zufluss (Stellgröße) und Abfluss")
        )

        # ── Chart 3: Regelabweichung ──
        line_err = (
            alt.Chart(data)
            .mark_line(color=COL_ERR, strokeWidth=2)
            .encode(
                x=alt.X("Zeit (h):Q", title="Zeit (h)"),
                y=alt.Y("Regelabweichung (m):Q", title="Regelabweichung (m)"),
            )
        )
        rule_zero = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color="#333", strokeDash=[4, 4], strokeWidth=1)
            .encode(y="y:Q")
        )

        chart3 = (
            (line_err + rule_zero)
            .properties(width=700, height=180, title="Regelabweichung (e = Sollwert − Istwert)")
        )

        return mo.vstack([
            mo.ui.altair_chart(chart1),
            mo.ui.altair_chart(chart2),
            mo.ui.altair_chart(chart3),
        ])

    return (simulate,)


@app.cell
def _(mo):
    mo.md("""
    # Füllstandsregelung für ein Rundbecken
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Vorgaben
    - **Maximaler Füllstand:** 6 m
    - **Mindestfüllstand:** 0,5 m
    - **Regelgröße:** Wasserstand im Becken
    - **Stellgröße:** Zulaufvolumenstrom (Pumpe)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Anlagen-Einstellungen
    """)
    return


@app.cell
def _(mo):
    sldb = mo.ui.slider(0.5, 8.0, 0.5, value=3, label="Durchmesser (m)")
    slqab = mo.ui.slider(0, 100, 5, value=50, label="Abfluss (%)")
    slw = mo.ui.slider(0.5, 5.5, 0.1, value=3, label="Sollwert (m)")
    slh0 = mo.ui.slider(0.5, 5.5, 0.1, value=1, label="Startwert (m)")
    return sldb, slh0, slqab, slw


@app.cell
def _(mo, sldb, slh0, slqab, slw):
    vstack_w = mo.vstack([
        mo.md("### Sollwert Füllstand"), slw, mo.md(f"**{slw.value} m**"),
    ])
    vstack_h0 = mo.vstack([
        mo.md("### Startwert Füllstand"), slh0, mo.md(f"**{slh0.value} m**"),
    ])
    vstack_db = mo.vstack([
        mo.md("### Beckendurchmesser"), sldb, mo.md(f"**{sldb.value} m**"),
    ])
    vstack_qab = mo.vstack([
        mo.md("### Abfluss aus Becken"), slqab, mo.md(f"**{slqab.value} %**"),
    ])
    return vstack_db, vstack_h0, vstack_qab, vstack_w


@app.cell
def _(mo, vstack_db, vstack_h0, vstack_qab, vstack_w):
    mo.hstack([vstack_w, vstack_h0, vstack_db, vstack_qab])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Regler-Einstellungen (PID-Regler)
    """)
    return


@app.cell
def _(mo):
    slk_p = mo.ui.slider(0.0, 3.0, 0.1, value=0.0, label="K_P")
    slk_i = mo.ui.slider(0.0, 3.0, 0.1, value=0.0, label="K_I")
    slk_d = mo.ui.slider(0.0, 0.3, 0.01, value=0.0, label="K_D")
    return slk_d, slk_i, slk_p


@app.cell
def _(mo, slk_d, slk_i, slk_p):
    vstack_kp = mo.vstack([
        mo.md("### $K_P$ (P-Anteil)"), slk_p, mo.md(f"**{slk_p.value}**"),
    ])
    vstack_ki = mo.vstack([
        mo.md("### $K_I$ (I-Anteil)"), slk_i, mo.md(f"**{slk_i.value}**"),
    ])
    vstack_kd = mo.vstack([
        mo.md("### $K_D$ (D-Anteil)"), slk_d, mo.md(f"**{slk_d.value}**"),
    ])
    return vstack_kd, vstack_ki, vstack_kp


@app.cell
def _(mo, vstack_kd, vstack_ki, vstack_kp):
    mo.hstack([vstack_kp, vstack_ki, vstack_kd])
    return


@app.cell
def _(mo):
    slte = mo.ui.slider(1, 10, 0.5, value=5, label="Dauer (h)")
    return (slte,)


@app.cell
def _(mo, slte):
    mo.vstack([mo.md("### Simulationsdauer"), slte, mo.md(f"**{slte.value} h**")])
    return


@app.cell
def _(math, simulate, sldb, slk_d, slk_i, slk_p, slte, slw):
    chart = simulate(
        k_p=slk_p.value,
        k_i=slk_i.value,
        k_d=slk_d.value,
        k=1 / (math.pi * sldb.value**2 / 4),
        k_Q=30,
        tau=1.5,
        t_end=slte.value,
        dt=0.001,
        setpoint=slw.value,
        h_max=6.0,
        h_min=0.5,
    )
    return (chart,)


@app.cell
def _(mo):
    mo.md("""
    ## Ergebnisse der Simulation
    """)
    return


@app.cell
def _(chart):
    chart
    return


@app.cell
def _():
    #mo.md("""
    #---
    #### Hinweise zur Bedienung
    #
    #| Parameter | Wirkung |
    #|---|---|
    #| **$K_P$ erhöhen** | Schnellere Reaktion, aber mehr Schwingneigung |
    #| **$K_I$ erhöhen** | Beseitigt bleibende Regelabweichung, kann Überschwingen #verstärken |
    #| **$K_D$ erhöhen** | Dämpft Schwingungen, zu hohe Werte verstärken Rauschen |
    #| **Abfluss %** | Höherer Abfluss → Regler muss stärker arbeiten |
    #
    #**Tipp:** Zuerst nur $K_P$ einstellen bis der Füllstand in die Nähe des Sollwerts #kommt,
    #dann $K_I$ langsam erhöhen um die bleibende Abweichung zu beseitigen,
    #und zuletzt $K_D$ vorsichtig erhöhen um Schwingungen zu dämpfen.
    #""")
    return


if __name__ == "__main__":
    app.run()
