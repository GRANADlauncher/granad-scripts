import matplotlib.pyplot as plt
import kwant
import numpy as np
import pickle
from dataclasses import dataclass, field

# ---- 2D Bravais types we’ll sample (canonical representatives) ----
BRAVAIS_NAMES = ["square", "rectangular", "hexagonal", "oblique", "centered_rect"]

@dataclass
class Config:
    lattices: list = field(default_factory=lambda: BRAVAIS_NAMES)
    orbitals = ["p"]              # keep it simple: p_z-like single-orbital per site
    lattice_constant = 1.0

    min_atoms_uc = 1
    max_atoms_uc = 3
    max_fractional_coordinate = 0.9
    min_fractional_coordinate = 0.0
    min_distance = 0.1            # in lattice coords (rough pre-screen)

    # NOTE: keep fields as-is, but sample positive decay so exp(-alpha * r) decays
    max_decay_rate = -1.0
    min_decay_rate = -3.0
    hopping_cutoff = 1e-2         # stop adding hoppings once |t(r)| < cutoff

    num_moments = 1024

    max_slater_koster = 3.0
    min_slater_koster = 0.5

    max_radius = 100.0            # circle radius in units of a
    min_radius = 10.0

    max_angle = np.pi / 3
    min_angle = 0.0

    # ---- samplers ---------------------------------------------------
    def get_decay_rate(self):
        # sample positive alpha in [1, 3] given your negative bounds
        return np.random.uniform(-self.max_decay_rate, -self.min_decay_rate)

    def get_radius(self):
        # log-uniform radius helps cover sizes better
        rmin, rmax = self.min_radius, self.max_radius
        return np.exp(np.random.uniform(np.log(rmin), np.log(rmax)))

    def get_angle(self):
        return np.random.uniform(self.min_angle, self.max_angle)

    def get_lattice_type(self):
        return np.random.randint(0, len(self.lattices))

    # sample (pp sigma) within limits with random signs
    def get_slater_koster(self):
        mag_sigma = np.random.uniform(self.min_slater_koster, self.max_slater_koster)
        sgn_sigma = np.random.choice([-1.0, 1.0])
        return [sgn_sigma * mag_sigma]  # [ppσ]

    # put orbitals at random fractional coordinates in the unit cell
    # returns dict like {"p": [(u1, v1), (u2, v2), ...]}
    def get_orbitals(self):
        n = np.random.randint(self.min_atoms_uc, self.max_atoms_uc + 1)
        pts = []
        tries = 0
        while len(pts) < n and tries < 1000:
            u = np.random.uniform(self.min_fractional_coordinate, self.max_fractional_coordinate)
            v = np.random.uniform(self.min_fractional_coordinate, self.max_fractional_coordinate)
            # crude separation in fractional coords
            ok = True
            for (uu, vv) in pts:
                if np.hypot(u - uu, v - vv) < self.min_distance:
                    ok = False
                    break
            if ok:
                pts.append((u, v))
            tries += 1
        return {"p": pts}

    def new_params(self):
        return Params(
            decay_rate=self.get_decay_rate(),
            radius=self.get_radius(),
            angle=self.get_angle(),
            lattice_type=self.get_lattice_type(),
            slater_koster=self.get_slater_koster(),
            orbitals=self.get_orbitals(),
            hopping_cutoff=self.hopping_cutoff,
            num_moments=self.num_moments,
            lattice_constant=self.lattice_constant,
        )

    def get_params(self, database):
        p = self.new_params()
        while p._hash() in database:
            p = self.new_params()
        return p

def circle_func(r):
    R2 = float(r) ** 2
    def circle(pos):
        x, y = pos
        return (x * x + y * y) <= R2
    return circle

@dataclass
class Params:
    decay_rate : float
    radius : float
    angle : float
    lattice_type : int
    slater_koster : list = None     # [pp_sigma, pp_pi]
    orbitals : dict = None          # {"p": [(u, v), ...]}
    hopping_cutoff : float = 1e-2
    num_moments : int = 1024
    lattice_constant : float = 1.0

    moments_matrix : np.ndarray = None

    def _hash(self):
        hashes = [hash(round(float(self.decay_rate), 6)),
                  hash(round(float(self.radius), 6)),
                  hash(round(float(self.angle), 6)),
                  hash(int(self.lattice_type)),
                  hash(tuple(np.round(self.slater_koster, 6))),
                  hash(str(np.round(self.get_basis(), 6).tolist()))]
        return ''.join(map(str, hashes))

    # rotation matrix
    def _R(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([[c, -s], [s, c]])

    # primitive lattice vectors (canonical by type) rotated by angle
    def get_prim_vecs(self):
        a = self.lattice_constant
        name = BRAVAIS_NAMES[self.lattice_type]
        if name == "square":
            v1, v2 = np.array([a, 0.0]), np.array([0.0, a])
        elif name == "rectangular":
            v1, v2 = np.array([a, 0.0]), np.array([0.0, 1.3 * a])
        elif name == "hexagonal":
            v1 = np.array([a, 0.0])
            v2 = np.array([0.5 * a, np.sqrt(3) / 2 * a])
        elif name == "oblique":
            gamma = np.deg2rad(80.0)    # canonical oblique
            v1 = np.array([a, 0.0])
            v2 = np.array([a * np.cos(gamma), a * np.sin(gamma)])
        elif name == "centered_rect":
            # centered rectangular (rhombic) with a!=b but 60° equivalent primitive choice
            v1 = np.array([a, 0.0])
            v2 = np.array([0.5 * a, 1.2 * a])  # mild skew
        else:
            v1, v2 = np.array([a, 0.0]), np.array([0.0, a])

        R = self._R()
        return [R @ v1, R @ v2]

    # atomic basis in Cartesian coords from fractional positions
    def get_basis(self):
        prim = self.get_prim_vecs()
        a1, a2 = prim[0], prim[1]

        # flatten all orbital positions; we keep one sublattice per position
        frac = []
        for pts in self.orbitals.values():
            frac.extend(list(pts))
        # unique (tolerant)
        uniq = []
        for (u, v) in frac:
            p = u * a1 + v * a2
            # dedup by distance
            if all(np.linalg.norm(p - q) > 1e-6 for q in uniq):
                uniq.append(p)
        return np.array(uniq)

    # for completeness; not used in kwant.lattice.general path
    def get_orbs(self):
        # 1 orbital per basis position (pz-like)
        return [1] * len(self.get_basis())

    def get_circle_func(self):
        return circle_func(self.radius)

    # enumerate long-range hoppings up to cutoff
    # returns list of (HoppingKind, value)
    def get_neighbors_hopping(self):
        a1, a2 = self.get_prim_vecs()
        basis = self.get_basis()
        n_basis = len(basis)
        alpha = float(self.decay_rate)
        V_pi = self.slater_koster[0]

        # conservative r_max from cutoff (use max SK magnitude)
        fmax = V_pi
        if fmax <= 0:
            return []
        r_max = -np.log(self.hopping_cutoff / fmax) / max(alpha, 1e-12)
        r_max = float(r_max)

        # integer search window for lattice translations
        minlen = min(np.linalg.norm(a1), np.linalg.norm(a2))
        M = int(np.ceil(r_max / max(minlen, 1e-6))) + 1

        return [V_sigma  ]


        hops = []

        # simple pp mixing model with direction cosine; for pz in-plane: cos=0 → pure π
        def hop_value_from_disp(dvec):
            r = np.linalg.norm(dvec)
            if r < 1e-12 or r > r_max:
                return 0.0
            return V_sigma * np.exp(-alpha * r)

        seen = set()
        for i in range(n_basis):
            for j in range(n_basis):
                for m1 in range(-M, M + 1):
                    for m2 in range(-M, M + 1):
                        # displacement from (i) to (j + m1 a1 + m2 a2)
                        dvec = (basis[j] + m1 * a1 + m2 * a2) - basis[i]
                        r = np.linalg.norm(dvec)
                        if r < 1e-9 or r > r_max:
                            continue
                        # avoid duplicates: identify by (i,j,m1,m2)
                        key = (i, j, m1, m2)
                        if key in seen:
                            continue
                        seen.add(key)
                        tval = hop_value_from_disp(dvec)
                        if abs(tval) >= self.hopping_cutoff:
                            HK = kwant.builder.HoppingKind((m1, m2), subs[j], subs[i])
                            hops.append((HK, float(tval)))
        return hops


    def get_hoppings(self, dist):
        return np.exp(-dist) * self.slater_koster[0]
    
def get_distances_system(syst, lat, n):
    syst[lat.shape(circle_func(50), (0, 0))] = 0.
    fsyst = syst.finalized()
    max_pos = min(100, fsyst.graph.num_nodes)    
    positions = np.array([fsyst.pos(i) for i in range(max_pos)])
    distances = np.unique(np.round(np.linalg.norm(positions - positions[:, None], axis = -1), 4))
    max_dist = int(n*(n-1) / 2 + 3)
    
    return distances[:max_dist]
    
# ---- define the system & compute KPM conductivity moments ----------
def save_moments_matrix(params: Params):
    syst = kwant.Builder()

    # lattice with given primitive vectors and basis
    lat = kwant.lattice.general(params.get_prim_vecs(), basis=params.get_basis(), name='L')
    syst[lat.shape(params.get_circle_func(), (0, 0))] = 0.

    distances = get_distances_system(syst, lat, params.get_basis().shape[0])
    for i, val in enumerate(params.get_hoppings(distances)):
        print(distances[i], i, val)
        syst[lat.neighbors(i+1)] = val
                  
    syst.eradicate_dangling()
    fsyst = syst.finalized()

    # KPM conductivity moments (xx)
    cond = kwant.kpm.conductivity(fsyst, alpha='x', beta='x', num_moments=params.num_moments)
    params.moments_matrix = cond.moments_matrix

    print(fsyst.graph.num_nodes, cond(mu=0, temperature=0.00))

    # save features + target
    payload = dict(
        decay_rate=params.decay_rate,
        radius=params.radius,
        angle=params.angle,
        lattice_type=params.lattice_type,
        slater_koster=params.slater_koster,
        basis=params.get_basis(),
        prim_vecs=params.get_prim_vecs(),
        moments_matrix=params.moments_matrix,
    )
    with open(f"data/{params._hash()}.pkl", "wb") as f:
        pickle.dump(payload, f)


def generate_data(n_samples=40000, seed=42):
    config = Config()
    np.random.seed(seed)
    database = set()

    for i in range(n_samples):
        params = config.get_params(database)
        database.add(params._hash())
        save_moments_matrix(params)


if __name__ == '__main__':
    generate_data()
