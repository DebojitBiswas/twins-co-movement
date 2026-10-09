"""C2 check: does a simulation-error-trained simulator follow G_hat = g2*G + (1-g2)*(-1/C), g2 = coherence(u, r)^2 ?"""
import numpy as np, scipy.signal as ss
import c1_synthetic as c

numG, denG = ss.ss2tf(c.Ad, c.Bd[:, :1], np.array([[1.0, 0.0]]), np.zeros((1, 1)))
numG = numG[0]

def freqresp_G(w): return ss.freqz(numG, denG, worN=w)[1]
def freqresp_C(w, kp):  # PI: u = kp e + (kp/20) * sum e  (integ updated after use -> ki/(z-1))
    z = np.exp(1j * w); return kp + (kp / 20.0) / (z - 1)

def predicted_effect(kp, g2fun, du=c.U_DES - c.U_OP, pad=8192):
    n = pad; w = 2 * np.pi * np.fft.rfftfreq(n); w[0] = 1e-6
    G = freqresp_G(w); C = freqresp_C(w, kp); g2 = g2fun(w)
    Gm = g2 * G + (1 - g2) * (-1.0 / C)
    x = np.zeros(n); x[:len(du)] = du
    return np.fft.irfft(np.fft.rfft(x) * Gm, n)[:len(du)]

def coherence_fun(U, R):
    f, Cxy = ss.coherence(U, R, nperseg=int(min(4096, len(U) // 4)))
    return lambda w: np.interp(w / (2 * np.pi), f, Cxy)

if __name__ == "__main__":
    for kp in (0.5, 2.0, 8.0):
        for sr in (0.0, 0.2, 1.0):
            Y, U, R, Q = c.simulate(200000, kp, 0.0, sr, 0.01, seed=7)
            g2 = (lambda w: np.zeros_like(w)) if sr == 0 else coherence_fun(U, R)
            pred = predicted_effect(kp, g2)
            oe = c.fit_oe(Y, U)
            # nonparametric (Wiener) learned response straight from the log, for reference
            f, Pyu = ss.csd(U, Y, nperseg=4096); _, Puu = ss.welch(U, nperseg=4096)
            Gw = lambda w: np.interp(w / (2 * np.pi), f, (Pyu / Puu).real) + 1j * np.interp(w / (2 * np.pi), f, (Pyu / Puu).imag)
            print(f"kp {kp} sr {sr}: g2(slow)={g2(np.array([2*np.pi/300]))[0]:.2f} | effect@50/200/599 true {c.EFF_TRUE[[50,200,599]].round(2)} "
                  f"pred {pred[[50,200,599]].round(2)} OE {oe[[50,200,599]].round(2)} | relerr pred-vs-OE "
                  f"{np.sqrt(np.mean((pred-oe)**2))/np.sqrt(np.mean(c.EFF_TRUE**2)):.2f}")
