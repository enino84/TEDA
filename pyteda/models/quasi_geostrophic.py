import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from .model import Model

class QGModel(Model):
    """Barotropic QG model with periodic BCs and state vector [q, psi]."""

    def __init__(self, N=64, L=1.0, dt=0.001, F=1600.0,
                 r=1e-2, rkb=1e-1, rkh=1e-5, rkh2=1e-8):
        self.N = N
        self.L = L
        self.dt = dt
        self.F = F
        self.r = r
        self.rkb = rkb
        self.rkh = rkh
        self.rkh2 = rkh2
        self.n = 2 * N * N  # state = [q, psi]
        self._L = None

        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)
        self.dx = self.dy = L / N

        kx = fftfreq(N, d=self.dx) * 2 * np.pi
        ky = fftfreq(N, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1e-10
        self.K4 = self.K2**2

    def laplacian(self, psi):
        return ifft2(-self.K2 * fft2(psi)).real

    def biharmonic(self, psi):
        return ifft2(self.K4 * fft2(psi)).real

    def arakawa_jacobian(self, psi, q):
        def jp(f): return np.roll(f, -1, axis=0)
        def jm(f): return np.roll(f, 1, axis=0)
        def kp(f): return np.roll(f, -1, axis=1)
        def km(f): return np.roll(f, 1, axis=1)

        j1 = (jp(psi) - jm(psi)) * (kp(q) - km(q)) - (kp(psi) - km(psi)) * (jp(q) - jm(q))
        j2 = kp(psi)*(jp(q)-q) - km(psi)*(q-jm(q)) - jp(psi)*(kp(q)-q) + jm(psi)*(q-km(q))
        j3 = jp(psi)*(kp(q)-km(q)) - jm(psi)*(kp(q)-km(q)) - kp(psi)*(jp(q)-jm(q)) + km(psi)*(jp(q)-jm(q))
        return (j1 + j2 + j3) / (12 * self.dx * self.dy)

    def forcing(self):
        return 2 * np.pi * np.sin(2 * np.pi * self.Y)

    def propagate(self, x0, T, just_final_state=True):
        q = x0[:self.N**2].reshape((self.N, self.N)).copy()
        psi_hat = fft2(q) / (self.K2 + self.F)
        psi_hat[0, 0] = 0.0
        psi = ifft2(psi_hat).real

        nt = len(T) - 1
        for step in range(nt):
            zeta = self.laplacian(psi)
            q = zeta - self.F * psi
            j = self.arakawa_jacobian(psi, q)
            dqdt = -psi.copy()
            dqdt += -self.r * j
            dqdt += -self.rkb * zeta
            dqdt += self.rkh * self.laplacian(zeta)
            dqdt += -self.rkh2 * self.biharmonic(zeta)
            dqdt += self.forcing()

            q += self.dt * dqdt
            psi_hat = fft2(q) / (self.K2 + self.F)
            psi_hat[0, 0] = 0.0
            psi = ifft2(psi_hat).real

        if just_final_state:
            return np.concatenate([q.flatten(), psi.flatten()])
        return None

    def get_initial_condition(self, seed=42, T=np.arange(0, 25, 0.001)):
        np.random.seed(seed)
        psi0 = 1e-4 * np.random.randn(self.N, self.N)
        q0 = self.laplacian(psi0) - self.F * psi0
        return self.propagate(np.concatenate([q0.flatten(), psi0.flatten()]), T)

    def get_number_of_variables(self):
        return self.n

    def get_ngb(self, i, r, cross=False):
        N = self.N
        n_grid = N * N

        var_index = 0 if i < n_grid else 1  # 0 = q, 1 = psi
        i_local = i % n_grid
        iy, ix = divmod(i_local, N)

        neighbors = []
        for dy in range(-r, r + 1):
            y = (iy + dy) % N
            for dx in range(-r, r + 1):
                x = (ix + dx) % N
                base = y * N + x

                # Add same variable neighbor
                neighbors.append(base + var_index * n_grid)

                if cross:
                    # Add cross-variable neighbor
                    neighbors.append(base + (1 - var_index) * n_grid)

        return np.array(sorted(set(neighbors)), dtype=int)  # Remove duplicates if cross=True


    def create_decorrelation_matrix(self, r, cross=False, cross_scale=1.0):
        N = self.N
        n = N * N
        total_n = 2 * n
        L = np.zeros((total_n, total_n))

        for i in range(total_n):
            iy1, ix1 = divmod(i % n, N)
            var1 = 0 if i < n else 1  # 0: q, 1: psi

            for j in range(i, total_n):
                iy2, ix2 = divmod(j % n, N)
                var2 = 0 if j < n else 1  # 0: q, 1: psi

                dx = min(abs(ix1 - ix2), N - abs(ix1 - ix2))
                dy = min(abs(iy1 - iy2), N - abs(iy1 - iy2))
                dij2 = dx**2 + dy**2

                if var1 == var2:
                    scale = 1.0
                elif cross:
                    scale = cross_scale
                else:
                    scale = 0.0

                val = np.exp(-dij2 / (2 * r**2)) * scale
                L[i, j] = L[j, i] = val

        self._L = L


    def get_decorrelation_matrix(self):
        return self._L
    
    def get_pre(self, i, r, cross=False):
        N = self.N
        n = N * N
        total_n = 2 * n

        if i < n:
            offset = 0  # q
        else:
            offset = n  # psi
            i_base = i - n
        i_base = i % n

        iy, ix = divmod(i_base, N)
        neighbors = []

        for dy in range(-r, r + 1):
            y = (iy + dy) % N
            for dx in range(-r, r + 1):
                x = (ix + dx) % N
                j_base = y * N + x

                for block_offset in [0, n]:  # q and psi blocks
                    j = j_base + block_offset
                    if j < i:
                        if cross or block_offset == offset:
                            neighbors.append(j)

        return np.array(neighbors, dtype=int)
