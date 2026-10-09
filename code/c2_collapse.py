"""C2b — law collapse: predicted effect from  G_hat = g2*G + (1-g2)*G_op,  G_op = (G F + H)/(F - C H)  (F = -d, H = disturbance path),
against the effects learned by OE and GRU ensembles in every synthetic cell (c1_results_gru_sub.json, c1b_results.json)."""
import json, numpy as np, scipy.signal as ss
import c1_synthetic as c
from c2_mixlaw import freqresp_C, coherence_fun

numG, denG = ss.ss2tf(c.Ad, c.Bd[:, :1], np.array([[1.0, 0.0]]), np.zeros((1, 1)))
numH, denH = ss.ss2tf(c.Ad, c.Bd[:, 1:], np.array([[1.0, 0.0]]), np.zeros((1, 1)))

def law_effect(kp, d, g2fun, du=c.U_DES - c.U_OP, n=8192):
    w = 2 * np.pi * np.fft.rfftfreq(n); w[0] = 1e-6
    G = ss.freqz(numG[0], denG, worN=w)[1]; Hq = ss.freqz(numH[0], denH, worN=w)[1]; C = freqresp_C(w, kp); F = -d
    Gop = (G * F + Hq) / (F - C * Hq)
    g2 = g2fun(w); Gm = g2 * G + (1 - g2) * Gop
    x = np.zeros(n); x[:len(du)] = du
    return np.fft.irfft(np.fft.rfft(x) * Gm, n)[:len(du)], np.fft.irfft(np.fft.rfft(x) * Gop, n)[:len(du)]

if __name__ == "__main__":
    sc = np.sqrt(np.mean(c.EFF_TRUE ** 2)); rows = []
    cells = json.load(open('c1_results_gru_sub.json'))
    ctrl = [r for r in json.load(open('c1b_results.json')) if r['regime'] == 'closed_sr1']
    for r in cells + ctrl:
        kp, d, sr, N = r['kp'], r['d'], r['sigma_r'], r['N']; s = r.get('seed', 0)
        seed = 1000 * s + 7 if 'seed' in r else 7
        Y, U, R, Q = c.simulate(N, kp, d, sr, 0.01, seed=seed)
        g2 = (lambda w: np.zeros_like(w)) if sr == 0 else coherence_fun(U, R)
        pred, gop = law_effect(kp, d, g2)
        gru = np.array(r['eff_mean']); idx = np.arange(0, c.H, 10)
        oe = c.fit_oe(Y, U)
        e = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / sc)
        row = dict(kp=kp, d=d, sr=sr, N=N, gru_vs_truth=e(gru, c.EFF_TRUE[idx]), gru_vs_law=e(gru, pred[idx]),
                   oe_vs_truth=e(oe, c.EFF_TRUE), oe_vs_law=e(oe, pred), pred=pred[idx].tolist(), gru=gru.tolist(), oe=oe[idx].tolist())
        rows.append(row)
        print(f"kp {kp} d {d} sr {sr} N {N:6d} | GRU: vs truth {row['gru_vs_truth']:.2f} vs law {row['gru_vs_law']:.2f} | "
              f"OE: vs truth {row['oe_vs_truth']:.2f} vs law {row['oe_vs_law']:.2f}", flush=True)
    json.dump(rows, open('c2_collapse.json', 'w'))
    import pandas as pd; df = pd.DataFrame(rows)
    print(df[['gru_vs_truth', 'gru_vs_law', 'oe_vs_truth', 'oe_vs_law']].median().round(2))
