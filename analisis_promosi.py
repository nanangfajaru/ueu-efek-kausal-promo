"""
Estimasi Efek Kausal Program Promosi terhadap Pendapatan Bersih Perusahaan
==========================================================================
Program ini menghasilkan seluruh angka pada Bab 4 tesis:
  4.1  Gambaran umum pendapatan dan diskon
  4.2  Pola pemberian program promosi
  4.3  Ilustrasi perhitungan program promosi #8
  4.4  Hasil per program promosi (Tabel 4.2)
  4.5  Efek rata-rata gabungan
  4.6  Tambahan biaya diskon (Tabel 4.3)

Data  : dunnhumby "The Complete Journey" (file CSV)
Cara  : python analisis_promosi.py --data FOLDER_DATA --output FOLDER_HASIL

Dua cara hitung yang dibandingkan:
  1) Konvensional  : rata-rata belanja penerima - rata-rata belanja pembanding
                     selama periode promosi.
  2) Terkoreksi    : Difference-in-Differences doubly robust dengan bobot
                     entropy balancing (Sant'Anna & Zhao, 2020;
                     Chattopadhyay dkk., 2020).
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Pengaturan
# ---------------------------------------------------------------------------
KARAKTERISTIK = [            # karakteristik pelanggan sebelum promosi (Bab 3.3)
    "log_belanja_total",     # 1. monetary
    "log_jumlah_kunjungan",  # 2. frequency
    "hari_sejak_kunjungan",  # 3. recency
    "log_belanja_56hari",    # 4. belanja 56 hari terakhir
    "proporsi_diskon",       # 5. kecenderungan memakai diskon
    "jumlah_promosi_lalu",   # 6. jumlah promosi yang pernah diterima
]
MIN_PENERIMA = 5             # promosi dengan penerima < 5 tidak dihitung
ULANGAN_BOOTSTRAP = 1000     # jumlah ulangan bootstrap untuk standard error
SEED = 2026                  # agar hasil selalu sama setiap dijalankan
PROMOSI_ILUSTRASI = 8        # promosi yang dipakai untuk ilustrasi Bab 4.3


# ---------------------------------------------------------------------------
# 1. Membaca data
# ---------------------------------------------------------------------------
def baca_data(folder):
    def cari(nama):
        for n in (nama, nama.replace(".csv", "_1.csv")):
            p = os.path.join(folder, n)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"File {nama} tidak ditemukan di folder {folder}")

    trx = pd.read_csv(
        cari("transaction_data.csv"),
        usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE",
                 "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"],
    )
    # Diskon dicatat negatif di data -> dibuat positif
    trx["DISKON"] = -(trx.RETAIL_DISC + trx.COUPON_MATCH_DISC)   # ditanggung perusahaan
    trx["KUPON_PRODUSEN"] = -trx.COUPON_DISC                     # ditanggung produsen
    trx["PENJUALAN_KOTOR"] = trx.SALES_VALUE + trx.DISKON
    penerima = pd.read_csv(cari("campaign_table.csv"))
    promosi = pd.read_csv(cari("campaign_desc.csv"))
    return trx, penerima, promosi


# ---------------------------------------------------------------------------
# 2. Karakteristik pelanggan sebelum suatu hari (cutoff)
# ---------------------------------------------------------------------------
def karakteristik_pelanggan(trx, penerima, promosi, cutoff):
    sebelum = trx[trx.DAY < cutoff]
    g = sebelum.groupby("household_key")
    f = pd.DataFrame({
        "belanja_total": g.SALES_VALUE.sum(),
        "jumlah_kunjungan": g.BASKET_ID.nunique(),
        "kunjungan_terakhir": g.DAY.max(),
        "kotor": g.PENJUALAN_KOTOR.sum(),
        "diskon": g.DISKON.sum(),
    })
    f["belanja_56hari"] = (sebelum[sebelum.DAY >= cutoff - 56]
                           .groupby("household_key").SALES_VALUE.sum())
    f["belanja_56hari"] = f["belanja_56hari"].fillna(0)
    f["hari_sejak_kunjungan"] = cutoff - f.kunjungan_terakhir
    f["proporsi_diskon"] = (f.diskon / f.kotor.replace(0, np.nan)).fillna(0)
    lalu = penerima.merge(promosi[["CAMPAIGN", "START_DAY"]], on="CAMPAIGN")
    lalu = lalu[lalu.START_DAY < cutoff].groupby("household_key").CAMPAIGN.nunique()
    f["jumlah_promosi_lalu"] = lalu.reindex(f.index).fillna(0)
    f["log_belanja_total"] = np.log1p(f.belanja_total.clip(lower=0))
    f["log_jumlah_kunjungan"] = np.log1p(f.jumlah_kunjungan)
    f["log_belanja_56hari"] = np.log1p(f.belanja_56hari.clip(lower=0))
    return f


def jumlah_per_pelanggan(trx, hari_awal, hari_akhir, kolom, index):
    w = trx[(trx.DAY >= hari_awal) & (trx.DAY <= hari_akhir)]
    return w.groupby("household_key")[kolom].sum().reindex(index).fillna(0).values


# ---------------------------------------------------------------------------
# 3. Metode terkoreksi: entropy balancing + DiD + regresi (doubly robust)
# ---------------------------------------------------------------------------
def bobot_entropy_balancing(X_pembanding, rata2_penerima):
    """Langkah 1: bobot pembanding agar rata-rata karakteristiknya sama
    persis dengan penerima, sedekat mungkin dengan bobot seragam."""
    Z = X_pembanding - rata2_penerima

    def fungsi(l):
        a = Z @ l
        m = a.max()
        e = np.exp(a - m)
        return m + np.log(e.sum()), (e[:, None] * Z).sum(0) / e.sum()

    hasil = minimize(fungsi, np.zeros(Z.shape[1]), jac=True, method="BFGS",
                     options={"maxiter": 2000, "gtol": 1e-10})
    a = Z @ hasil.x
    w = np.exp(a - a.max())
    w = w / w.sum()
    return w if np.all(np.isfinite(w)) else None


def efek_terkoreksi(X, D, dY):
    """Langkah 1-3. X = karakteristik, D = 1 penerima / 0 pembanding,
    dY = perubahan belanja (periode promosi - periode sebelum)."""
    sd = X.std(0)
    Xs = (X - X.mean(0)) / np.where(sd == 0, 1, sd)
    Xt, Xc = Xs[D == 1], Xs[D == 0]
    w = bobot_entropy_balancing(Xc, Xt.mean(0))
    if w is None:
        return np.nan, None
    reg = LinearRegression().fit(Xc, dY[D == 0], sample_weight=w)   # Langkah 3
    efek = (dY[D == 1] - reg.predict(Xt)).mean() - np.sum(w * (dY[D == 0] - reg.predict(Xc)))
    return efek, w


def standard_error_bootstrap(X, D, dY, rng):
    idx_t, idx_c = np.where(D == 1)[0], np.where(D == 0)[0]
    hasil = []
    for _ in range(ULANGAN_BOOTSTRAP):
        ii = np.concatenate([rng.choice(idx_t, len(idx_t)), rng.choice(idx_c, len(idx_c))])
        hasil.append(efek_terkoreksi(X[ii], D[ii], dY[ii])[0])
    hasil = np.array(hasil)
    hasil = hasil[np.isfinite(hasil)]
    return float(np.std(hasil, ddof=1)) if len(hasil) > min(20, ULANGAN_BOOTSTRAP // 2) else np.nan


# ---------------------------------------------------------------------------
# 4. Perhitungan per program promosi
# ---------------------------------------------------------------------------
def hitung_promosi(no, trx, penerima, promosi):
    info = promosi[promosi.CAMPAIGN == no].iloc[0]
    s, e = int(info.START_DAY), int(info.END_DAY)
    L = e - s + 1
    # Promosi lain yang berjalan bersamaan
    bersamaan = promosi[(promosi.START_DAY <= e) & (promosi.END_DAY >= s)].CAMPAIGN.tolist()
    set_penerima = set(penerima[penerima.CAMPAIGN == no].household_key)
    set_sibuk = set(penerima[penerima.CAMPAIGN.isin(bersamaan)].household_key)

    # Karakteristik diukur sebelum periode sebelum dimulai (hari s - L)
    f = karakteristik_pelanggan(trx, penerima, promosi, s - L)
    # Kelompok pembanding = pelanggan yang tidak menerima promosi yang berjalan bersamaan
    f = f[f.index.isin(set_penerima) | ~f.index.isin(set_sibuk)].copy()
    D = f.index.isin(set_penerima).astype(int)

    sesudah = jumlah_per_pelanggan(trx, s, e, "SALES_VALUE", f.index)
    sebelum = jumlah_per_pelanggan(trx, s - L, s - 1, "SALES_VALUE", f.index)
    hasil = dict(
        promosi=no, tipe=info.DESCRIPTION.replace("Type", ""), hari_mulai=s, hari_akhir=e,
        durasi=L, pelanggan_penerima=int(D.sum()), kelompok_pembanding=int((1 - D).sum()),
        belanja_sebelum_penerima=sebelum[D == 1].mean(),
        belanja_selama_penerima=sesudah[D == 1].mean(),
        belanja_sebelum_pembanding=sebelum[D == 0].mean(),
        belanja_selama_pembanding=sesudah[D == 0].mean(),
    )
    # Cara hitung konvensional
    hasil["konvensional"] = sesudah[D == 1].mean() - sesudah[D == 0].mean()

    if D.sum() < MIN_PENERIMA:
        return hasil
    X = f[KARAKTERISTIK].values
    dY = sesudah - sebelum
    efek, w = efek_terkoreksi(X, D, dY)
    if w is None or not np.isfinite(efek):
        return hasil            # tidak ada pembanding yang mirip (mis. promosi #3)

    rng = np.random.default_rng(SEED + no)
    se = standard_error_bootstrap(X, D, dY, rng)
    diskon = (jumlah_per_pelanggan(trx, s, e, "DISKON", f.index)
              - jumlah_per_pelanggan(trx, s - L, s - 1, "DISKON", f.index))
    efek_diskon, _ = efek_terkoreksi(X, D, diskon)
    se_diskon = standard_error_bootstrap(X, D, diskon, rng)

    hasil.update(
        jumlah_pembanding_efektif=1 / np.sum(w ** 2),
        setelah_koreksi=efek, se=se,
        ik95_bawah=efek - 1.96 * se, ik95_atas=efek + 1.96 * se,
        tambahan_diskon=efek_diskon, se_diskon=se_diskon,
        belanja_sebelum_pembanding_dibobot=np.sum(w * sebelum[D == 0]),
        belanja_selama_pembanding_dibobot=np.sum(w * sesudah[D == 0]),
    )
    return hasil


def gabungkan(efek, se):
    """Rata-rata pembobotan (bobot = 1 / SE^2) dan statistik Q."""
    w = 1 / se ** 2
    m = np.sum(w * efek) / np.sum(w)
    s = np.sqrt(1 / np.sum(w))
    Q = np.sum(w * (efek - m) ** 2)
    return m, s, Q, len(efek) - 1, chi2.sf(Q, len(efek) - 1)



# ---------------------------------------------------------------------------
# 6. Gambar (diagram) untuk Bab 4
# ---------------------------------------------------------------------------
WARNA = {"biru": "#2a78d6", "oranye": "#eb6834", "aqua": "#1baf7a",
         "teks": "#0b0b0b", "teks2": "#52514e", "grid": "#e4e3df", "latar": "#ffffff"}


def _rp(x, k=1):
    """Format angka gaya Indonesia: 1.234,5"""
    s = f"{abs(x):,.{k}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("−" if x < 0 else "") + s


def _gaya(ax):
    ax.set_facecolor(WARNA["latar"])
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    for sisi in ("left", "bottom"):
        ax.spines[sisi].set_color(WARNA["grid"])
    ax.tick_params(colors=WARNA["teks2"], labelsize=9)
    ax.grid(axis="x", color=WARNA["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def buat_gambar(df, ringkas, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": WARNA["latar"]})
    simpan = lambda fig, nama: (fig.savefig(os.path.join(folder, nama), dpi=200, bbox_inches="tight"),
                                plt.close(fig))

    # Gambar 4.1 — pola pemberian promosi
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    _gaya(ax)
    label = ["Pernah menerima promosi", "Tidak pernah menerima promosi"]
    nilai = [ringkas["belanja_penerima"], ringkas["belanja_bukan"]]
    ax.barh(label, nilai, height=0.45, color=WARNA["biru"])
    for i, v in enumerate(nilai):
        ax.text(v + max(nilai) * 0.01, i, f"${_rp(v, 0)}", va="center", fontsize=9, color=WARNA["teks"])
    ax.invert_yaxis()
    ax.set_xlabel("Rata-rata total belanja per pelanggan selama 711 hari ($)", fontsize=9, color=WARNA["teks2"])
    ax.set_title("Gambar 4.1 Rata-rata belanja pelanggan penerima dan bukan penerima promosi",
                 fontsize=10, color=WARNA["teks"], loc="left")
    ax.set_xlim(0, max(nilai) * 1.15)
    simpan(fig, "gambar_4_1_pola_pemberian_promosi.png")

    # Gambar 4.2 — ilustrasi promosi #8 (sebelum vs selama)
    il = df[df.promosi == PROMOSI_ILUSTRASI].iloc[0]
    seri = [("Pelanggan penerima promosi", il.belanja_sebelum_penerima, il.belanja_selama_penerima, WARNA["biru"]),
            ("Pembanding tanpa bobot", il.belanja_sebelum_pembanding, il.belanja_selama_pembanding, WARNA["oranye"]),
            ("Pembanding setelah bobot", il.belanja_sebelum_pembanding_dibobot,
             il.belanja_selama_pembanding_dibobot, WARNA["aqua"])]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    _gaya(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=WARNA["grid"], linewidth=0.8)
    for nama, a, b, w in seri:
        ax.plot([0, 1], [a, b], color=w, linewidth=2, marker="o", markersize=7)
        ax.text(1.04, b, f"{nama}: {_rp(a)} → {_rp(b)}", va="center", fontsize=8.5, color=WARNA["teks"])
    ax.set_xticks([0, 1], ["Periode sebelum", "Selama promosi"])
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(0, max(max(s[1], s[2]) for s in seri) * 1.15)
    ax.set_ylabel("Rata-rata belanja per pelanggan ($)", fontsize=9, color=WARNA["teks2"])
    ax.set_title(f"Gambar 4.2 Ilustrasi perhitungan promosi #{PROMOSI_ILUSTRASI}: "
                 f"konvensional {_rp(il.konvensional)} vs setelah koreksi {_rp(il.setelah_koreksi)}",
                 fontsize=10, color=WARNA["teks"], loc="left")
    simpan(fig, "gambar_4_2_ilustrasi_promosi_8.png")

    # Gambar 4.3 — konvensional vs setelah koreksi per promosi
    d = df.sort_values("promosi", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 8))
    _gaya(ax)
    y = np.arange(len(d))
    ok = d.setelah_koreksi.notna()
    ax.hlines(y[ok], d.ik95_bawah[ok], d.ik95_atas[ok], color=WARNA["biru"], linewidth=1.5, alpha=0.6)
    ax.scatter(d.setelah_koreksi[ok], y[ok], s=36, color=WARNA["biru"], zorder=3,
               edgecolor=WARNA["latar"], linewidth=1.5, label="Setelah koreksi (dengan interval kepercayaan 95%)")
    ax.scatter(d.konvensional, y, s=36, color=WARNA["oranye"], zorder=3, marker="D",
               edgecolor=WARNA["latar"], linewidth=1.5, label="Konvensional")
    ax.axvline(0, color=WARNA["teks2"], linewidth=0.8)
    ax.set_yticks(y, [f"#{n}" for n in d.promosi])
    ax.set_xlabel("Efek terhadap pendapatan bersih per pelanggan ($)", fontsize=9, color=WARNA["teks2"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("Gambar 4.3 Efek tiap promosi: konvensional vs setelah koreksi",
                 fontsize=10, color=WARNA["teks"], loc="left")
    simpan(fig, "gambar_4_3_konvensional_vs_koreksi.png")

    # Gambar 4.4 — efek setelah koreksi per promosi + efek gabungan (forest plot)
    d2 = d[ok].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 8))
    _gaya(ax)
    y = np.arange(len(d2)) + 2
    ax.hlines(y, d2.ik95_bawah, d2.ik95_atas, color=WARNA["biru"], linewidth=1.5)
    ax.scatter(d2.setelah_koreksi, y, s=30, color=WARNA["biru"], zorder=3)
    m, s_ = ringkas["efek_gabungan"], ringkas["se_gabungan"]
    ax.fill([m - 1.96 * s_, m, m + 1.96 * s_, m], [0, 0.35, 0, -0.35], color=WARNA["teks"])
    ax.axvline(0, color=WARNA["teks2"], linewidth=0.8)
    ax.set_yticks(list(y) + [0], [f"#{n}" for n in d2.promosi] + ["Gabungan"])
    ax.text(m + 1.96 * s_ + 8, 0, f"{_rp(m)} [{_rp(m - 1.96 * s_)}; {_rp(m + 1.96 * s_)}]",
            va="center", fontsize=8.5, color=WARNA["teks"])
    ax.set_xlabel("Efek setelah koreksi per pelanggan ($), dengan interval kepercayaan 95%",
                  fontsize=9, color=WARNA["teks2"])
    ax.set_title("Gambar 4.4 Efek setelah koreksi tiap promosi dan efek rata-rata gabungan",
                 fontsize=10, color=WARNA["teks"], loc="left")
    simpan(fig, "gambar_4_4_efek_setelah_koreksi.png")

    # Gambar 4.5 — ringkasan: konvensional vs setelah koreksi vs tambahan diskon
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    _gaya(ax)
    label = ["Konvensional (rata-rata)", "Setelah koreksi (gabungan)", "Tambahan diskon perusahaan"]
    nilai = [ringkas["konvensional"], ringkas["efek_gabungan"], ringkas["diskon_gabungan"]]
    ax.barh(label, nilai, height=0.45, color=WARNA["biru"])
    for i, v in enumerate(nilai):
        ax.text(v + max(nilai) * 0.01, i, f"${_rp(v)}", va="center", fontsize=9, color=WARNA["teks"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(nilai) * 1.15)
    ax.set_xlabel("$ per pelanggan per periode promosi", fontsize=9, color=WARNA["teks2"])
    ax.set_title("Gambar 4.5 Ringkasan: efek konvensional, efek setelah koreksi, dan tambahan diskon",
                 fontsize=10, color=WARNA["teks"], loc="left")
    simpan(fig, "gambar_4_5_ringkasan.png")


# ---------------------------------------------------------------------------
# 5. Program utama
# ---------------------------------------------------------------------------
def main():
    global ULANGAN_BOOTSTRAP, SEED
    ap = argparse.ArgumentParser(description="Analisis efek program promosi (Bab 4 tesis)")
    ap.add_argument("--data", default="data", help="folder berisi file CSV dunnhumby")
    ap.add_argument("--output", default="hasil", help="folder untuk menyimpan hasil")
    ap.add_argument("--bootstrap", type=int, default=ULANGAN_BOOTSTRAP,
                    help="jumlah ulangan bootstrap (default %(default)s)")
    ap.add_argument("--seed", type=int, default=SEED, help="seed angka acak (default %(default)s)")
    args = ap.parse_args()
    ULANGAN_BOOTSTRAP, SEED = args.bootstrap, args.seed
    os.makedirs(args.output, exist_ok=True)
    mulai = time.time()

    print("Membaca data ...", flush=True)
    trx, penerima, promosi = baca_data(args.data)
    lap = []

    # ---- 4.1 Gambaran umum
    bersih = trx.SALES_VALUE.sum()
    diskon = trx.DISKON.sum()
    kotor = bersih + diskon
    lap += ["=== 4.1 Gambaran umum pendapatan dan diskon ===",
            f"Jumlah pelanggan                 : {trx.household_key.nunique():,}",
            f"Pendapatan bersih                : ${bersih:,.0f}",
            f"Diskon ditanggung perusahaan     : ${diskon:,.0f}",
            f"Penjualan kotor                  : ${kotor:,.0f}",
            f"Diskon / penjualan kotor         : {100 * diskon / kotor:.1f}%",
            f"Kupon produsen (tidak dihitung)  : ${trx.KUPON_PRODUSEN.sum():,.0f} "
            f"({100 * trx.KUPON_PRODUSEN.sum() / (diskon + trx.KUPON_PRODUSEN.sum()):.1f}% dari total potongan)",
            f"Baris transaksi berdiskon        : {100 * (trx.RETAIL_DISC < 0).mean():.1f}%", ""]

    # ---- 4.2 Pola pemberian promosi
    total = trx.groupby("household_key").SALES_VALUE.sum()
    jml = penerima.groupby("household_key").CAMPAIGN.nunique().reindex(total.index).fillna(0)
    lap += ["=== 4.2 Pola pemberian program promosi ===",
            f"Pelanggan pernah menerima promosi: {(jml > 0).sum():,} ({100 * (jml > 0).mean():.1f}%)",
            f"Rata-rata belanja penerima       : ${total[jml > 0].mean():,.0f}",
            f"Rata-rata belanja bukan penerima : ${total[jml == 0].mean():,.0f}",
            f"Korelasi jumlah promosi-belanja  : {np.corrcoef(jml, total)[0, 1]:.2f}", ""]

    # ---- 4.4 Per program promosi
    baris = []
    for no in sorted(promosi.CAMPAIGN):
        t0 = time.time()
        h = hitung_promosi(no, trx, penerima, promosi)
        baris.append(h)
        txt = (f"{h['setelah_koreksi']:8.1f} (SE {h['se']:.1f})"
               if "setelah_koreksi" in h else "tidak dapat dihitung")
        print(f"  Promosi #{no:2d}: konvensional {h['konvensional']:7.1f} | setelah koreksi {txt}"
              f"  [{time.time() - t0:.0f} detik]", flush=True)
    df = pd.DataFrame(baris)

    # ---- 4.3 Ilustrasi
    il = df[df.promosi == PROMOSI_ILUSTRASI].iloc[0]
    lap += [f"=== 4.3 Ilustrasi perhitungan promosi #{PROMOSI_ILUSTRASI} ===",
            f"{'Kelompok':40s}{'Sebelum':>10s}{'Selama':>10s}{'Perubahan':>11s}",
            f"{'Pelanggan penerima promosi':40s}{il.belanja_sebelum_penerima:10.1f}"
            f"{il.belanja_selama_penerima:10.1f}{il.belanja_selama_penerima - il.belanja_sebelum_penerima:11.1f}",
            f"{'Kelompok pembanding tanpa bobot':40s}{il.belanja_sebelum_pembanding:10.1f}"
            f"{il.belanja_selama_pembanding:10.1f}{il.belanja_selama_pembanding - il.belanja_sebelum_pembanding:11.1f}",
            f"{'Kelompok pembanding setelah bobot':40s}{il.belanja_sebelum_pembanding_dibobot:10.1f}"
            f"{il.belanja_selama_pembanding_dibobot:10.1f}"
            f"{il.belanja_selama_pembanding_dibobot - il.belanja_sebelum_pembanding_dibobot:11.1f}",
            f"Konvensional    : {il.konvensional:.1f}",
            f"Setelah koreksi : {il.setelah_koreksi:.1f}", ""]

    # ---- 4.5 dan 4.6 Gabungan
    ok = df.dropna(subset=["setelah_koreksi", "se"])
    m, s, Q, dof, p = gabungkan(ok.setelah_koreksi.values, ok.se.values)
    okd = ok.dropna(subset=["se_diskon"])
    md, sd_, *_ = gabungkan(okd.tambahan_diskon.values, okd.se_diskon.values)
    konv = ok.konvensional.mean()
    lap += ["=== 4.4 Ringkasan per program promosi ===",
            f"Konvensional positif             : {(df.konvensional > 0).sum()} dari {len(df)} promosi",
            f"Konvensional (min - maks, median): {df.konvensional.min():.1f} - {df.konvensional.max():.1f}, "
            f"median {df.konvensional.median():.1f}",
            f"Dapat dihitung setelah koreksi   : {len(ok)} promosi",
            f"Signifikan positif               : {list(ok[ok.ik95_bawah > 0].promosi)}", "",
            f"=== 4.5 Efek rata-rata gabungan ({len(ok)} promosi) ===",
            f"Efek setelah koreksi             : ${m:.1f} (SE {s:.1f}; IK95% {m - 1.96 * s:.1f} s.d. {m + 1.96 * s:.1f})",
            f"Rata-rata konvensional           : ${konv:.1f}",
            f"Efek / konvensional              : {100 * m / konv:.1f}%",
            f"Statistik Q                      : {Q:.1f} (derajat bebas {dof}, p = {p:.2f})", "",
            "=== 4.6 Tambahan biaya diskon ===",
            f"Tambahan diskon perusahaan       : ${md:.2f} (SE {sd_:.2f}; IK95% {md - 1.96 * sd_:.2f} s.d. {md + 1.96 * sd_:.2f})",
            f"Efek terhadap penjualan kotor    : ${m + md:.1f}; porsi diskon {100 * md / (m + md):.0f}%", ""]

    # ---- Gambar
    ringkas = dict(belanja_penerima=total[jml > 0].mean(), belanja_bukan=total[jml == 0].mean(),
                   efek_gabungan=m, se_gabungan=s, diskon_gabungan=md, konvensional=konv)
    print("Membuat gambar ...", flush=True)
    buat_gambar(df, ringkas, os.path.join(args.output, "gambar"))

    # ---- Simpan
    kolom = ["promosi", "tipe", "durasi", "pelanggan_penerima", "kelompok_pembanding",
             "jumlah_pembanding_efektif", "konvensional", "setelah_koreksi", "se",
             "ik95_bawah", "ik95_atas", "tambahan_diskon", "se_diskon"]
    tabel = df.reindex(columns=kolom).round(2)
    tabel.to_csv(os.path.join(args.output, "tabel_hasil_per_promosi.csv"), index=False)
    try:
        tabel.to_excel(os.path.join(args.output, "tabel_hasil_per_promosi.xlsx"), index=False)
    except ImportError:
        pass
    teks = "\n".join(lap)
    with open(os.path.join(args.output, "ringkasan_hasil.txt"), "w", encoding="utf-8") as fh:
        fh.write(teks)
    print("\n" + teks)
    print(f"Selesai dalam {(time.time() - mulai) / 60:.1f} menit. Hasil tersimpan di folder '{args.output}'.")


if __name__ == "__main__":
    main()
