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
    Verwendet Derivative on Measurement (Ableitung der Messgroesse
    statt der Regelabweichung), um Derivative-Kick bei
    Sollwertaenderungen zu vermeiden -- wie in der industriellen Praxis.
    """

    def __init__(self, k_p, k_i=0, k_d=0, T_f=0.5):
        """
        Parameter:
            k_p : Proportionalverstaerkung
            k_i : Integralverstaerkung
            k_d : Differentialverstaerkung
            T_f : Filterzeitkonstante fuer D-Anteil (h), 0 = kein Filter
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.T_f = T_f
        self.integral = 0.0
        self.prev_x = None
        self.prev_d_filtered = 0.0
        self._last_error = 0.0
        self._last_dt = 0.0

    def compute(self, error, x_measured, dt):
        """
        Berechnet den PID-Reglerausgang (unbegrenzt).

        Parameter:
            error      : Regelabweichung (w - x)
            x_measured : Aktuelle Messgroesse (Fuellstand)
            dt         : Zeitschritt (h)

        Rueckgabe:
            y_R : Reglerausgang (unbegrenzt, dimensionslos)
        """
        self._last_error = error
        self._last_dt = dt

        # P-Anteil
        P = self.k_p * error

        # I-Anteil (vorlaeufig aufintegrieren)
        self.integral += error * dt
        I = self.k_i * self.integral

        # D-Anteil: Derivative on Measurement (dx/dt statt de/dt)
        if self.prev_x is None:
            self.prev_x = x_measured

        if dt > 0:
            dx_dt = (x_measured - self.prev_x) / dt
            if self.T_f > 0:
                alpha = dt / (self.T_f + dt)
                d_filtered = self.prev_d_filtered + alpha * (dx_dt - self.prev_d_filtered)
            else:
                d_filtered = dx_dt
            self.prev_d_filtered = d_filtered
        else:
            d_filtered = 0.0

        # Minus: Ableitung der Messgroesse (nicht des Fehlers)
        D = -self.k_d * d_filtered
        self.prev_x = x_measured

        return P + I + D

    def anti_windup(self, was_clamped):
        """
        Anti-Windup (Clamping): Macht die letzte Integration rueckgaengig,
        wenn die Stellgroesse extern begrenzt wurde.
        """
        if was_clamped:
            self.integral -= self._last_error * self._last_dt


@app.cell
def _(alt, deque, math, mo, np, pd, slh0):
    # ================================================================
    # Anlagenparameter (didaktisch optimiert)
    # ================================================================
    DY_MAX = 200.0   # Max. Stellgroessenaenderung (m3/h pro h)
    T_F = 0.5        # Filterzeitkonstante D-Anteil (h)
    T_DEAD = 0.10    # Totzeit (h)
    C_AB = 3.0       # Abflusskoeffizient

    def simulate(
        k_p,
        k_i,
        k_d,
        k=1.0,
        k_Q=20.0,
        tau=5.0,
        t_end=5,
        dt=0.001,
        setpoint=3.0,
        h_max=6.0,
        h_min=0.5,
    ):
        """
        Simuliert die Fuellstandsregelung eines Rundbeckens.

        Stellgroessen-Logik:
            y_phys = k_Q + y_R * k_Q
            -> y_R = 0  : Pumpe laeuft im Arbeitspunkt (k_Q m3/h)
            -> y_R > 0  : Pumpe laeuft schneller
            -> y_R < 0  : Pumpe laeuft langsamer (bis aus)
            Begrenzung: 0 <= y_phys <= 3*k_Q
        """
        t_vals = np.arange(0, t_end, dt)
        x_vals = []
        y_vals = []
        error_vals = []
        qab_vals = []

        Y_MIN = 0.0
        Y_MAX = 3 * k_Q

        regler = Regler(k_p, k_i, k_d, T_f=T_F)
        x = slh0.value

        dead_steps = max(1, int(T_DEAD / dt)) if T_DEAD > 0 else 0
        y_buffer = deque([k_Q] * (dead_steps + 1), maxlen=dead_steps + 1)
        y_prev = k_Q

        for t in t_vals:
            error = setpoint - x

            y_R = regler.compute(error, x, dt)
            y_raw = k_Q + y_R * k_Q

            max_change = DY_MAX * dt
            dy = y_raw - y_prev
            if abs(dy) > max_change:
                y_raw = y_prev + max_change * (1 if dy > 0 else -1)

            y_clamped = max(Y_MIN, min(y_raw, Y_MAX))
            was_clamped = abs(y_raw - y_clamped) > 1e-9
            regler.anti_windup(was_clamped)
            y_prev = y_clamped

            y_buffer.append(y_clamped)
            y_out = y_buffer[0] if dead_steps > 0 else y_clamped

            Q_ab = math.sqrt(max(0, 2 * 9.81 * x)) * C_AB

            dxdt = regelstrecke(t, x, y_out, k, Q_ab, tau)
            x += dxdt * dt
            x = max(0, min(x, 8))

            x_vals.append(x)
            y_vals.append(y_out)
            error_vals.append(error)
            qab_vals.append(Q_ab)

        # Downsampling (~500 Punkte)
        n = len(t_vals)
        step = max(1, n // 500)
        idx = range(0, n, step)

        data = pd.DataFrame({
            "Zeit (h)": t_vals[::step],
            "Fuellstand (m)": [x_vals[i] for i in idx],
            "Zufluss (m3/h)": [y_vals[i] for i in idx],
            "Regelabweichung (m)": [error_vals[i] for i in idx],
            "Abfluss (m3/h)": [qab_vals[i] for i in idx],
        })

        # ============================================================
        # Diagramme
        # ============================================================
        COL_IST = "#1f77b4"
        COL_SOLL = "#d62728"
        COL_ZU = "#2ca02c"
        COL_AB = "#9467bd"
        COL_ERR = "#ff7f0e"
        COL_GRENZE = "#aaaaaa"

        nearest = alt.selection_point(
            nearest=True, on="pointerover",
            fields=["Zeit (h)"], empty=False,
        )

        def make_selector(df):
            return (
                alt.Chart(df).mark_point(opacity=0)
                .encode(x="Zeit (h):Q")
                .add_params(nearest)
            )

        def make_vrule(df):
            return (
                alt.Chart(df)
                .mark_rule(color="#bbb", strokeWidth=1, strokeDash=[4, 3])
                .encode(x="Zeit (h):Q")
                .transform_filter(nearest)
            )

        # -- Chart 1: Fuellstand mit Sollwert, Toleranzband, Grenzen --
        y_domain_max = max(h_max + 0.5, 7)
        band_w = 0.05 * setpoint  # +/-5% Toleranzband

        line_ist = (
            alt.Chart(data)
            .mark_line(color=COL_IST, strokeWidth=2)
            .encode(
                x=alt.X("Zeit (h):Q", title=""),
                y=alt.Y("Fuellstand (m):Q", title="Fuellstand (m)",
                         scale=alt.Scale(domain=[0, y_domain_max])),
            )
        )

        tooltip_ist = (
            alt.Chart(data)
            .mark_circle(size=70, color=COL_IST)
            .encode(
                x="Zeit (h):Q",
                y="Fuellstand (m):Q",
                tooltip=[
                    alt.Tooltip("Zeit (h):Q", title="Zeit", format=".2f"),
                    alt.Tooltip("Fuellstand (m):Q", title="Fuellstand", format=".2f"),
                    alt.Tooltip("Regelabweichung (m):Q", title="Abweichung", format=".3f"),
                ],
                opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            )
        )

        rule_soll = (
            alt.Chart(pd.DataFrame({"y": [setpoint]}))
            .mark_rule(color=COL_SOLL, strokeDash=[6, 4], strokeWidth=2)
            .encode(y="y:Q")
        )
        txt_soll = (
            alt.Chart(pd.DataFrame({
                "x": [t_end * 0.02], "y": [setpoint + 0.25],
                "t": [f"Sollwert {setpoint} m"]}))
            .mark_text(color=COL_SOLL, fontSize=11, fontWeight="bold", align="left")
            .encode(x="x:Q", y="y:Q", text="t:N")
        )

        # Toleranzband +/-5%
        tolerance_band = (
            alt.Chart(pd.DataFrame({"y_lo": [setpoint - band_w], "y_hi": [setpoint + band_w]}))
            .mark_rect(color=COL_SOLL, opacity=0.12)
            .encode(y="y_lo:Q", y2="y_hi:Q")
        )
        txt_band = (
            alt.Chart(pd.DataFrame({
                "x": [t_end * 0.75], "y": [setpoint + band_w + 0.15],
                "t": ["+/-5 % Toleranzband"]}))
            .mark_text(color=COL_SOLL, fontSize=9, fontStyle="italic",
                       align="center", opacity=0.7)
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

        chart1 = alt.layer(
            tolerance_band, txt_band,
            line_ist, rule_soll, txt_soll, rule_max, txt_max, rule_min, txt_min,
            make_selector(data), make_vrule(data), tooltip_ist,
        ).properties(width=700, height=280, title="Fuellstandsverlauf")

        # -- Chart 2: Zufluss + Abfluss --
        flow_lines = (
            alt.Chart(data)
            .transform_fold(
                ["Zufluss (m3/h)", "Abfluss (m3/h)"],
                as_=["Groesse", "Volumenstrom (m3/h)"],
            )
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Zeit (h):Q", title=""),
                y=alt.Y("Volumenstrom (m3/h):Q", title="Volumenstrom (m3/h)"),
                color=alt.Color("Groesse:N",
                    scale=alt.Scale(
                        domain=["Zufluss (m3/h)", "Abfluss (m3/h)"],
                        range=[COL_ZU, COL_AB]),
                    legend=alt.Legend(title=None, orient="top")),
                strokeDash=alt.StrokeDash("Groesse:N",
                    scale=alt.Scale(
                        domain=["Zufluss (m3/h)", "Abfluss (m3/h)"],
                        range=[[0], [6, 4]]),
                    legend=None),
            )
        )

        flow_tooltips = (
            alt.Chart(data)
            .transform_fold(
                ["Zufluss (m3/h)", "Abfluss (m3/h)"],
                as_=["Groesse", "Volumenstrom (m3/h)"],
            )
            .mark_circle(size=70)
            .encode(
                x="Zeit (h):Q",
                y="Volumenstrom (m3/h):Q",
                color=alt.Color("Groesse:N",
                    scale=alt.Scale(
                        domain=["Zufluss (m3/h)", "Abfluss (m3/h)"],
                        range=[COL_ZU, COL_AB]),
                    legend=None),
                tooltip=[
                    alt.Tooltip("Zeit (h):Q", title="Zeit", format=".2f"),
                    alt.Tooltip("Groesse:N", title="Groesse"),
                    alt.Tooltip("Volumenstrom (m3/h):Q", title="m3/h", format=".1f"),
                ],
                opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            )
        )

        chart2 = alt.layer(
            flow_lines, make_selector(data), make_vrule(data), flow_tooltips,
        ).properties(width=700, height=220, title="Zufluss (Stellgroesse) und Abfluss")

        # -- Chart 3: Regelabweichung mit Toleranzband --
        line_err = (
            alt.Chart(data)
            .mark_line(color=COL_ERR, strokeWidth=2)
            .encode(
                x=alt.X("Zeit (h):Q", title="Zeit (h)"),
                y=alt.Y("Regelabweichung (m):Q", title="Regelabweichung (m)"),
            )
        )

        tooltip_err = (
            alt.Chart(data)
            .mark_circle(size=70, color=COL_ERR)
            .encode(
                x="Zeit (h):Q",
                y="Regelabweichung (m):Q",
                tooltip=[
                    alt.Tooltip("Zeit (h):Q", title="Zeit", format=".2f"),
                    alt.Tooltip("Regelabweichung (m):Q", title="Abweichung", format=".3f"),
                ],
                opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            )
        )

        err_tolerance = (
            alt.Chart(pd.DataFrame({"y_lo": [-band_w], "y_hi": [band_w]}))
            .mark_rect(color=COL_SOLL, opacity=0.12)
            .encode(y="y_lo:Q", y2="y_hi:Q")
        )

        rule_zero = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color="#333", strokeDash=[4, 4], strokeWidth=1)
            .encode(y="y:Q")
        )

        chart3 = alt.layer(
            err_tolerance, line_err, rule_zero,
            make_selector(data), make_vrule(data), tooltip_err,
        ).properties(width=700, height=180, title="Regelabweichung (e = Sollwert - Istwert)")

        # Kombination
        combined = (
            alt.vconcat(chart1, chart2, chart3)
            .resolve_scale(x="shared")
            .configure_title(fontSize=14, anchor="start")
        )

        return mo.vstack([combined])

    return (simulate,)


@app.cell
def _(mo):
    mo.md("""
    # Fuellstandsregelung fuer ein Rundbecken
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Vorgaben
    - **Maximaler Fuellstand:** 6 m
    - **Mindestfuellstand:** 0,5 m
    - **Regelgroesse:** Wasserstand im Becken
    - **Stellgroesse:** Zulaufvolumenstrom (Pumpe)
    - **Toleranzband:** +/-5 % um den Sollwert (rot hinterlegt)
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
    slw = mo.ui.slider(1.0, 5.0, 0.1, value=3.0, label="Sollwert (m)")
    slh0 = mo.ui.slider(0.5, 5.0, 0.1, value=1.5, label="Startwert (m)")
    return slh0, slw


@app.cell
def _(mo, slh0, slw):
    vstack_w = mo.vstack([
        mo.md("### Sollwert Fuellstand"), slw, mo.md(f"**{slw.value} m**"),
    ])
    vstack_h0 = mo.vstack([
        mo.md("### Startwert Fuellstand"), slh0, mo.md(f"**{slh0.value} m**"),
    ])
    return vstack_h0, vstack_w


@app.cell
def _(mo, vstack_h0, vstack_w):
    mo.hstack([vstack_w, vstack_h0])
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
    slk_d = mo.ui.slider(0.0, 1.0, 0.05, value=0.0, label="K_D")
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
    slte = mo.ui.slider(1, 10, 0.5, value=5.0, label="Dauer (h)")
    return (slte,)


@app.cell
def _(mo, slte):
    mo.vstack([mo.md("### Simulationsdauer"), slte, mo.md(f"**{slte.value} h**")])
    return


@app.cell
def _(simulate, slk_d, slk_i, slk_p, slte, slw):
    chart = simulate(
        k_p=slk_p.value,
        k_i=slk_i.value,
        k_d=slk_d.value,
        k=1.0,
        k_Q=20.0,
        tau=5.0,
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
    #| **$K_P$ erhoehen** | Schnellere Reaktion, aber mehr Ueberschwingen und #Schwingneigung |
    #| **$K_I$ erhoehen** | Beseitigt bleibende Regelabweichung, kann Ueberschwingen #verstaerken |
    #| **$K_D$ erhoehen** | Daempft Ueberschwingen und Schwingungen (Derivative on #Measurement) |
    #
    #**Tipp:** Zuerst nur $K_P$ einstellen bis der Fuellstand in die Naehe des Sollwerts #kommt,
    #dann $K_I$ langsam erhoehen um die bleibende Abweichung zu beseitigen,
    #und zuletzt $K_D$ vorsichtig erhoehen um Schwingungen zu daempfen.#
    #
    #**Toleranzband:** Der rot hinterlegte Bereich zeigt das +/-5 %-Band um den Sollwert.
    #Die Einschwingzeit ist die Zeit, bis der Fuellstand dauerhaft in diesem Band bleibt.
    #""")
    return


if __name__ == "__main__":
    app.run()
