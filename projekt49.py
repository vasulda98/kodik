"""Projekt 49 - Pruchodnost pasoveho dopravniku"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # pro zobrazeni oken odstrante radek (nebo pouzijte plt.show())
import matplotlib.pyplot as plt
from pathlib import Path

VSTUP = Path("data_projekt49.csv")
VYSTUP = Path("vysledek.csv")
PRAH = 0.70   # omezeni: merený průtok < 70 % teoretického
DT = 1.0      # vzorkování [s]

# ---------------------------------------------------------------- načtení
df = pd.read_csv(VSTUP)
t = df["cas_s"].to_numpy(dtype=float)
v = df["rychlost_m_s"].to_numpy(dtype=float)
rho = df["hustota_ks_m"].to_numpy(dtype=float)
q_mer = df["mereny_prutok_ks_s"].to_numpy(dtype=float)

# ---------------------------------------------------------------- Úkol 2: model průtoku
q_model = v * rho                      # [m/s * ks/m = ks/s]

# ---------------------------------------------------------------- Úkol 3: omezení
omezeni = (q_model > 0) & (q_mer < PRAH * q_model)


def souvisle_useky(maska):
    """Vrátí seznam (začátek_idx, konec_idx) maximálních souvislých úseků True."""
    useky, start = [], None
    for i, m in enumerate(maska):
        if m and start is None:
            start = i
        elif not m and start is not None:
            useky.append((start, i - 1))
            start = None
    if start is not None:
        useky.append((start, len(maska) - 1))
    return useky


useky = souvisle_useky(omezeni)

# ---------------------------------------------------------------- Úkol 4: ztracená produkce
ztrata = np.maximum(0.0, q_model - q_mer) * DT          # ztráta v každé sekundě [ks]
ztrata_celkem = ztrata.sum()
ztrata_useky = [ztrata[a:b + 1].sum() for a, b in useky]
ztrata_detekovane = float(sum(ztrata_useky))             # = součet ztrát přes omezené vzorky
ztrata_mimo = ztrata_celkem - ztrata_detekovane
# kumulovaná ztráta pro graf (celý záznam)
ztrata_kum = np.cumsum(ztrata)

# ---------------------------------------------------------------- Výpis výsledků
print("=" * 60)
print("PROJEKT 49 - výsledky")
print("=" * 60)
print(f"Počet vzorků: {len(t)}, nalezeno úseků omezení: {len(useky)}")
print(f"{'#':>2} {'první [s]':>10} {'poslední [s]':>13} {'délka [vzorků]':>15} {'ztráta [ks]':>12}")
for k, ((a, b), z) in enumerate(zip(useky, ztrata_useky), start=1):
    print(f"{k:>2} {t[a]:>10.0f} {t[b]:>13.0f} {b - a + 1:>15d} {z:>12.1f}")
print("-" * 60)
print(f"Ztráta v detekovaných úsecích:       {ztrata_detekovane:10.1f} ks")
print(f"Ztráta mimo detekované úseky:        {ztrata_mimo:10.1f} ks")
print(f"Ztráta za celý záznam (všechny s):   {ztrata_celkem:10.1f} ks")
print("(Součet za celý záznam se liší od součtu v úsecích o ztráty v sekundách,")
print(" kdy měřený průtok sice zaostával za modelem, ale neklesl pod 70 %.)")

# ---------------------------------------------------------------- Úkol 5: export
vysledek = pd.DataFrame({
    "cas_s": df["cas_s"],
    "prutok_model": q_model,
    "prutok_mereny": q_mer,
    "omezeni": omezeni.astype(int),
    "ztrata_ks": ztrata,
})
vysledek.to_csv(VYSTUP, index=False, float_format="%.5f")
print(f"\nUloženo do {VYSTUP}")

# ---------------------------------------------------------------- Úkol 1 + 2: grafy
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
plt.rcParams["axes.grid"] = True

ax[0, 0].plot(t, v, color="tab:blue", lw=1.2)
ax[0, 0].set_title("Rychlost dopravníku")

ax[0, 1].plot(t, rho, color="tab:blue", lw=1.2)
ax[0, 1].set_title("Hustota balíků")

ax[1, 0].plot(t, q_model, color="tab:blue", lw=1.2, label="Model")
ax[1, 0].plot(t, q_mer, color="tab:orange", lw=1.2, label="Měření")
for a, b in useky:   # vyznačení úseků omezení
    ax[1, 0].axvspan(t[a], t[b], color="red", alpha=0.15, lw=0)
ax[1, 0].set_title("Průchodnost")
ax[1, 0].legend(loc="lower left")

ax[1, 1].plot(t, ztrata_kum, color="tab:blue", lw=1.2)
ax[1, 1].set_title("Kumulovaná ztráta produkce")

for i in range(2):
    for j in range(2):
        ax[i, j].set_xlabel("")
        ax[i, j].grid(alpha=0.3)
        ax[i, j].set_xlim(-30, 630)
fig.tight_layout()
fig.savefig("projekt49_grafy.png", dpi=150)
# plt.show()   # odkomentujte pro zobrazení okna
