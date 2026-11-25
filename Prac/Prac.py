

"""
=====================================================================================
  APEX COAST GP  -  Procedural 3D coastal circuit racer  (Python + Ursina Engine)
=====================================================================================

SETUP (Windows / Visual Studio / VS Code)
-----------------------------------------
    1.  Install Python 3.10 - 3.12 (64-bit)
    2.  pip install ursina
    3.  python apex_coast_gp.py          (or press F5 in VS Code)

No external assets are used: track, scenery, cars, sky, HUD and effects are all
generated procedurally at start-up (loading takes a few seconds).

CONTROLS
--------
    W / UP ............ Throttle
    S / DOWN .......... Brake  (hold when stopped = reverse)
    A,D / LEFT,RIGHT .. Steer  (speed sensitive + counter-steer assist)
    SPACE ............. Handbrake (hairpins / drifts)
    LEFT SHIFT ........ Nitro / DRS boost
    C ................. Camera: chase cam  <->  hood cam
    R ................. Reset to track
    ENTER ............. Restart race
    ESC ............... Quit

TECHNICAL NOTES
---------------
 * Vehicle model: bicycle model with load-dependent tyre curves (slip angle based),
   friction-circle traction, weight transfer, downforce, 6-speed automatic gearbox,
   rev limiter, handbrake, off-road grip loss. Fixed 120 Hz physics sub-stepping.
 * Rendering: the whole world is baked into a handful of static vertex-coloured
   meshes (road, curbs, rails, terrain, trees...) with one custom GLSL shader
   (procedural asphalt grain, wet sun sheen, atmospheric haze). Particles are pooled,
   skid marks use one dynamically rebuilt mesh.
 * If your GPU/driver rejects the custom shader, set USE_CUSTOM_SHADER = False.
"""

import math
import random as rnd
from collections import deque

from ursina import *

# =====================================================================================
#  CONFIG
# =====================================================================================
TITLE = 'APEX COAST GP'
USE_CUSTOM_SHADER = True
TOTAL_LAPS = 3
NUM_BOTS = 6

ROAD_HALF = 8.0           # half road width (m)
CURB_W = 1.4              # curb width (m)
WALL_LAT = ROAD_HALF + CURB_W + 4.0   # guard-rail lateral distance from centreline
STEP = 8.0                # target distance between centreline samples (m)

FOG_NEAR = 140.0
FOG_FAR = 2000.0
FOG_COL = (0.97, 0.74, 0.54)          # golden-hour haze (also sky horizon colour)

C4 = color.Color


def clampf(x, a, b):
    return a if x < a else (b if x > b else x)


def lerpf(a, b, t):
    return a + (b - a) * t


def wrap_pi(a):
    while a > math.pi:
        a -= math.tau
    while a < -math.pi:
        a += math.tau
    return a


def norm3(v):
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / l, v[1] / l, v[2] / l)


def mix3(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


SUN_DIR = norm3((0.63, 0.21, 0.75))       # direction TOWARDS the sun (low, golden hour)
CAR_LIGHT = norm3((-0.35, 0.85, 0.45))


def shade(rgb, n, L=SUN_DIR, a=1.0):
    """Bake simple warm-sun / cool-sky lighting into a vertex colour."""
    d = max(0.0, n[0] * L[0] + n[1] * L[1] + n[2] * L[2])
    h = 0.5 + 0.5 * n[1]
    amb = 0.30 + 0.28 * h
    r = rgb[0] * (amb * 0.80 + d * 1.10)
    g = rgb[1] * (amb * 0.92 + d * 0.92)
    b = rgb[2] * (amb * 1.12 + d * 0.70)
    return C4(min(r, 1.0), min(g, 1.0), min(b, 1.0), a)


# =====================================================================================
#  MESH BUILDER (batched geometry with vertex colours)
# =====================================================================================
class MB:
    def __init__(self):
        self.v = []
        self.c = []

    def tri(self, a, b, c, col):
        self.v += [a, b, c]
        self.c += [col, col, col]

    def quad(self, a, b, c, d, col):
        self.v += [a, b, c, a, c, d]
        self.c += [col] * 6

    def quadc(self, a, b, c, d, ca, cb, cc, cd):
        self.v += [a, b, c, a, c, d]
        self.c += [ca, cb, cc, ca, cc, cd]

    def tri_lit(self, a, b, c, rgb, center, L=SUN_DIR):
        """Triangle lit with a normal flipped to point away from `center`."""
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        n = norm3(n)
        mx = (a[0] + b[0] + c[0]) / 3.0 - center[0]
        my = (a[1] + b[1] + c[1]) / 3.0 - center[1]
        mz = (a[2] + b[2] + c[2]) / 3.0 - center[2]
        if n[0] * mx + n[1] * my + n[2] * mz < 0:
            n = (-n[0], -n[1], -n[2])
        self.tri(a, b, c, shade(rgb, n, L))

    def box(self, cx, cy, cz, sx, sy, sz, rgb, yaw=0.0, L=SUN_DIR, flat=False):
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
        cs, sn = math.cos(yaw), math.sin(yaw)

        def P(lx, ly, lz):
            return (cx + lx * cs + lz * sn, cy + ly, cz - lx * sn + lz * cs)

        def N(nx, ny, nz):
            return (nx * cs + nz * sn, ny, -nx * sn + nz * cs)

        faces = (
            ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
            ((-1, 0, 0), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
            ((0, 1, 0), ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
            ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
            ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
            ((0, 0, -1), ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))),
        )
        for n, cr in faces:
            col = C4(rgb[0], rgb[1], rgb[2], 1.0) if flat else shade(rgb, N(*n), L)
            pts = [P(hx * k[0], hy * k[1], hz * k[2]) for k in cr]
            self.quad(pts[0], pts[1], pts[2], pts[3], col)

    def build(self):
        return Mesh(vertices=self.v, colors=self.c, mode='triangle')


def cone(mb, cx, cy, cz, radius, height, sides, rgb, L=SUN_DIR, jitter=0.0):
    apex = (cx, cy + height, cz)
    ring = []
    for k in range(sides):
        a = math.tau * k / sides
        r = radius * (1.0 + rnd.uniform(-jitter, jitter))
        ring.append((cx + math.cos(a) * r, cy, cz + math.sin(a) * r))
    ctr = (cx, cy + height * 0.4, cz)
    for k in range(sides):
        mb.tri_lit(ring[k], ring[(k + 1) % sides], apex, rgb, ctr, L)


# =====================================================================================
#  CAR MESHES
# =====================================================================================
def loft(mb, st, side_rgb, top_rgbs, bottom_rgb, L=CAR_LIGHT):
    for i in range(len(st) - 1):
        z0, w0, a0, b0 = st[i]
        z1, w1, a1, b1 = st[i + 1]
        n_top = norm3((0.0, z1 - z0, -(b1 - b0)))
        mb.quad((-w0, b0, z0), (-w1, b1, z1), (w1, b1, z1), (w0, b0, z0), shade(top_rgbs[i], n_top, L))
        mb.quad((-w0, a0, z0), (w0, a0, z0), (w1, a1, z1), (-w1, a1, z1), shade(bottom_rgb, (0, -1, 0), L))
        mb.quad((w0, a0, z0), (w0, b0, z0), (w1, b1, z1), (w1, a1, z1), shade(side_rgb, (1, 0, 0), L))
        mb.quad((-w0, a0, z0), (-w1, a1, z1), (-w1, b1, z1), (-w0, b0, z0), shade(side_rgb, (-1, 0, 0), L))
    z, w, a, b = st[0]
    mb.quad((-w, a, z), (w, a, z), (w, b, z), (-w, b, z), shade(side_rgb, (0, 0, -1), L))
    z, w, a, b = st[-1]
    mb.quad((-w, a, z), (-w, b, z), (w, b, z), (w, a, z), shade(side_rgb, (0, 0, 1), L))


def build_car_mesh(paint, stripe):
    mb = MB()
    L = CAR_LIGHT
    glass = (0.07, 0.11, 0.16)
    carbon = (0.05, 0.05, 0.06)
    body = [(-2.15, 0.80, 0.30, 0.56), (-1.90, 0.94, 0.26, 0.64), (-1.10, 0.97, 0.24, 0.68),
            (0.30, 0.97, 0.24, 0.66), (1.20, 0.93, 0.26, 0.58), (1.85, 0.86, 0.28, 0.47),
            (2.15, 0.72, 0.32, 0.40)]
    loft(mb, body, paint, [paint] * 6, (0.03, 0.03, 0.035), L)
    cabin = [(-1.30, 0.60, 0.66, 0.86), (-0.85, 0.69, 0.66, 1.10), (-0.25, 0.71, 0.66, 1.17),
             (0.35, 0.67, 0.66, 1.02), (0.85, 0.60, 0.66, 0.76)]
    loft(mb, cabin, glass, [glass, paint, paint, glass], (0.03, 0.03, 0.035), L)
    # racing stripe along bonnet / roof / boot
    for st in (body, cabin):
        for i in range(len(st) - 1):
            z0, w0, a0, b0 = st[i]
            z1, w1, a1, b1 = st[i + 1]
            if st is cabin and i in (0, 3):
                continue
            n = norm3((0.0, z1 - z0, -(b1 - b0)))
            mb.quad((-0.13, b0 + 0.004, z0), (-0.13, b1 + 0.004, z1), (0.13, b1 + 0.004, z1),
                    (0.13, b0 + 0.004, z0), shade(stripe, n, L))
    # aero + details
    mb.box(0, 0.98, -2.00, 1.80, 0.05, 0.42, carbon, L=L)
    mb.box(-0.55, 0.78, -1.95, 0.06, 0.42, 0.08, carbon, L=L)
    mb.box(0.55, 0.78, -1.95, 0.06, 0.42, 0.08, carbon, L=L)
    mb.box(0, 0.27, 2.12, 1.95, 0.04, 0.35, carbon, L=L)
    mb.box(-0.62, 0.52, 2.13, 0.30, 0.08, 0.06, (1.0, 0.95, 0.75), flat=True)
    mb.box(0.62, 0.52, 2.13, 0.30, 0.08, 0.06, (1.0, 0.95, 0.75), flat=True)
    mb.box(0, 0.60, -2.17, 1.50, 0.07, 0.05, (1.0, 0.06, 0.04), flat=True)
    mb.box(-0.97, 0.88, 0.30, 0.12, 0.08, 0.16, paint, L=L)
    mb.box(0.97, 0.88, 0.30, 0.12, 0.08, 0.16, paint, L=L)
    mb.box(-0.45, 0.32, -2.18, 0.12, 0.10, 0.10, (0.25, 0.25, 0.27), L=L)
    mb.box(0.45, 0.32, -2.18, 0.12, 0.10, 0.10, (0.25, 0.25, 0.27), L=L)
    return mb.build()


def build_wheel_mesh(width):
    mb = MB()
    R = 0.34
    sides = 12
    hw = width / 2.0
    for k in range(sides):
        a0 = math.tau * k / sides
        a1 = math.tau * (k + 1) / sides
        y0, z0 = R * math.cos(a0), R * math.sin(a0)
        y1, z1 = R * math.cos(a1), R * math.sin(a1)
        t = 0.045 if k % 2 == 0 else 0.075
        mb.quad((-hw, y0, z0), (hw, y0, z0), (hw, y1, z1), (-hw, y1, z1), C4(t, t, t + 0.005, 1))
        rim = (0.72, 0.72, 0.78) if k % 2 == 0 else (0.22, 0.22, 0.25)
        rc = C4(rim[0], rim[1], rim[2], 1)
        for sx in (-hw, hw):
            mb.tri((sx, 0, 0), (sx, y0 * 0.82, z0 * 0.82), (sx, y1 * 0.82, z1 * 0.82), rc)
            mb.quad((sx, y0 * 0.82, z0 * 0.82), (sx, y0, z0), (sx, y1, z1), (sx, y1 * 0.82, z1 * 0.82),
                    C4(0.03, 0.03, 0.035, 1))
    return mb.build()


# =====================================================================================
#  CUSTOM GLSL SHADER (vertex colour + procedural grain + wet sun sheen + haze)
# =====================================================================================
VERT = '''
#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
in vec4 p3d_Vertex;
in vec4 p3d_Color;
out vec4 v_col;
out vec3 v_view;
out vec3 v_pos;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_col = p3d_Color;
    v_view = (p3d_ModelViewMatrix * p3d_Vertex).xyz;
    v_pos = p3d_Vertex.xyz;
}
'''

FRAG = '''
#version 140
uniform vec4 p3d_ColorScale;
uniform vec4 fog_data;
uniform vec3 fog_color;
uniform vec3 sun_dir;
uniform vec3 cam_pos;
in vec4 v_col;
in vec3 v_view;
in vec3 v_pos;
out vec4 frag;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
void main() {
    float d = length(v_view);
    vec3 col = v_col.rgb * p3d_ColorScale.rgb;
    float grain_k = fog_data.z;
    float wet = fog_data.w;
    if (grain_k > 0.0) {
        float lod = clamp(1.0 - d / 140.0, 0.0, 1.0);
        float n = vnoise(v_pos.xz * 2.7) * 0.55 + vnoise(v_pos.xz * 13.0) * 0.45;
        col *= 1.0 + (n - 0.5) * grain_k * (0.25 + 0.75 * lod);
    }
    if (wet > 0.0) {
        vec3 V = normalize(cam_pos - v_pos);
        vec3 R = reflect(-sun_dir, vec3(0.0, 1.0, 0.0));
        float s = pow(max(dot(R, V), 0.0), 14.0);
        float pm = 0.55 + 0.9 * vnoise(v_pos.xz * 0.045);
        col += vec3(1.0, 0.74, 0.46) * s * wet * pm;
    }
    float f = clamp((d - fog_data.x) / (fog_data.y - fog_data.x), 0.0, 1.0);
    f = f * f * (3.0 - 2.0 * f);
    col = mix(col, fog_color, f);
    frag = vec4(col, v_col.a * p3d_ColorScale.a);
}
'''


# =====================================================================================
#  TRACK
# =====================================================================================
PROFILE = [  # (offset beyond curb edge, vertical drop, rgb) - terrain cross section
    (0.0, 0.00, (0.30, 0.36, 0.17)),
    (5.0, -0.08, (0.33, 0.40, 0.18)),
    (22.0, -0.50, (0.28, 0.36, 0.16)),
    (55.0, -9.0, (0.24, 0.30, 0.15)),
    (95.0, -26.0, (0.38, 0.33, 0.26)),
]


def ground_drop(off):
    """Terrain drop at `off` metres beyond the curb edge (for tree placement)."""
    for k in range(len(PROFILE) - 1):
        o0, d0, _ = PROFILE[k]
        o1, d1, _ = PROFILE[k + 1]
        if off <= o1:
            return lerpf(d0, d1, (off - o0) / (o1 - o0))
    return PROFILE[-1][1]


def catmull(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return tuple(0.5 * ((2 * p1[i]) + (-p0[i] + p2[i]) * t + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2
                        + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3) for i in range(3))


class Track:
    LAT_A = 10.5      # AI lateral acceleration budget (m/s^2)
    BRAKE_A = 11.0    # AI braking budget (m/s^2)

    def __init__(self):
        sc = 1.6
        ctrl = [  # x, z, elevation
            (0, 0, 0), (110, -6, 2), (190, 24, 5), (260, -12, 10), (335, 28, 16), (410, 115, 24),
            (430, 250, 30), (365, 385, 26), (245, 440, 18), (150, 385, 10), (60, 300, 8),
            (-40, 330, 14), (-150, 425, 22), (-290, 405, 18), (-385, 305, 9), (-345, 175, 4),
            (-225, 110, 2), (-250, 15, 1), (-145, -62, 0)]
        pts = [(x * sc, y * 1.3, z * sc) for (x, z, y) in ctrl]
        n = len(pts)
        dense = []
        for i in range(n):
            p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
            for k in range(48):
                dense.append(catmull(p0, p1, p2, p3, k / 48.0))
        m = len(dense)
        cum = [0.0]
        for i in range(m):
            a, b = dense[i], dense[(i + 1) % m]
            cum.append(cum[-1] + math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))
        total = cum[-1]
        N = int(total // STEP)
        sp = total / N
        self.N, self.spacing, self.length = N, sp, total
        self.CX, self.CY, self.CZ = [], [], []
        j = 0
        for i in range(N):
            s = i * sp
            while cum[j + 1] < s:
                j += 1
            t = (s - cum[j]) / max(cum[j + 1] - cum[j], 1e-6)
            a, b = dense[j], dense[(j + 1) % m]
            self.CX.append(lerpf(a[0], b[0], t))
            self.CY.append(lerpf(a[1], b[1], t))
            self.CZ.append(lerpf(a[2], b[2], t))
        self.TX, self.TZ, self.RX, self.RZ, self.H, self.SL = [], [], [], [], [], []
        for i in range(N):
            a, b = (i - 1) % N, (i + 1) % N
            dx, dz = self.CX[b] - self.CX[a], self.CZ[b] - self.CZ[a]
            l = math.hypot(dx, dz)
            tx, tz = dx / l, dz / l
            self.TX.append(tx)
            self.TZ.append(tz)
            self.RX.append(tz)
            self.RZ.append(-tx)
            self.H.append(math.atan2(tx, tz))
            self.SL.append((self.CY[b] - self.CY[a]) / (2.0 * sp))
        K = [abs(wrap_pi(self.H[(i + 1) % N] - self.H[(i - 1) % N])) / (2.0 * sp) for i in range(N)]
        self.K = [max(K[(i + d) % N] for d in range(-3, 4)) for i in range(N)]
        prof = [min(82.0, math.sqrt(self.LAT_A / max(k, 1e-4))) for k in self.K]
        for _ in range(3):
            for i in range(N - 1, -1, -1):
                nxt = prof[(i + 1) % N]
                prof[i] = min(prof[i], math.sqrt(nxt * nxt + 2.0 * self.BRAKE_A * sp))
        self.prof = prof
        self.cx = sum(self.CX) / N
        self.cz = sum(self.CZ) / N

    def pt(self, i, off, dy=0.0):
        i %= self.N
        return (self.CX[i] + self.RX[i] * off, self.CY[i] + dy, self.CZ[i] + self.RZ[i] * off)

    def pt_t(self, i, t, off, dy=0.0):
        a = self.pt(i, off, dy)
        b = self.pt(i + 1, off, dy)
        return (lerpf(a[0], b[0], t), lerpf(a[1], b[1], t), lerpf(a[2], b[2], t))

    def sample(self, s):
        s %= self.length
        f = s / self.spacing
        i = int(f) % self.N
        j = (i + 1) % self.N
        t = f - int(f)
        x = lerpf(self.CX[i], self.CX[j], t)
        y = lerpf(self.CY[i], self.CY[j], t)
        z = lerpf(self.CZ[i], self.CZ[j], t)
        tx, tz = lerpf(self.TX[i], self.TX[j], t), lerpf(self.TZ[i], self.TZ[j], t)
        l = math.hypot(tx, tz) or 1.0
        return x, y, z, tx / l, tz / l

    def nearest(self, x, z):
        best, bi = 1e18, 0
        for i in range(self.N):
            d = (x - self.CX[i]) ** 2 + (z - self.CZ[i]) ** 2
            if d < best:
                best, bi = d, i
        return bi


# =====================================================================================
#  WORLD GEOMETRY BUILDERS
# =====================================================================================
def build_road(T):
    road, curbs, lines = MB(), MB(), MB()
    N = T.N
    H = ROAD_HALF
    for i in range(N):
        j = (i + 1) % N
        sh = 0.94 + 0.06 * ((i // 2) % 2)
        ac = C4(0.20 * sh, 0.20 * sh, 0.215 * sh, 1)
        road.quad(T.pt(i, -H), T.pt(i, H), T.pt(j, H), T.pt(j, -H), ac)
        # curbs (2 m red / white stripes)
        for side in (-1, 1):
            for k in range(4):
                t0, t1 = k / 4.0, (k + 1) / 4.0
                red = ((i * 4 + k) % 2 == 0)
                cc = C4(0.85, 0.08, 0.06, 1) if red else C4(0.92, 0.92, 0.9, 1)
                o0, o1 = side * H, side * (H + CURB_W)
                curbs.quad(T.pt_t(i, t0, o0, 0.03), T.pt_t(i, t0, o1, 0.03),
                           T.pt_t(i, t1, o1, 0.03), T.pt_t(i, t1, o0, 0.03), cc)
        # edge lines + dashed centre
        wl = C4(0.90, 0.90, 0.88, 1)
        for side in (-1, 1):
            lines.quad(T.pt(i, side * (H - 0.75), 0.02), T.pt(i, side * (H - 0.45), 0.02),
                       T.pt(j, side * (H - 0.45), 0.02), T.pt(j, side * (H - 0.75), 0.02), wl)
        if i % 2 == 0:
            lines.quad(T.pt(i, -0.12, 0.02), T.pt(i, 0.12, 0.02), T.pt_t(i, 0.5, 0.12, 0.02),
                       T.pt_t(i, 0.5, -0.12, 0.02), wl)
    # start / finish chequered line (2 rows x 16 columns across the road)
    for r in range(2):
        for c in range(16):
            o0 = -H + c * (2 * H / 16.0)
            o1 = o0 + 2 * H / 16.0
            t0, t1 = r * 0.125, (r + 1) * 0.125
            col = C4(0.95, 0.95, 0.95, 1) if (r + c) % 2 == 0 else C4(0.04, 0.04, 0.04, 1)
            lines.quad(T.pt_t(0, t0, o0, 0.03), T.pt_t(0, t0, o1, 0.03),
                       T.pt_t(0, t1, o1, 0.03), T.pt_t(0, t1, o0, 0.03), col)
    return road, curbs, lines


def build_terrain(T):
    mb = MB()
    E = ROAD_HALF + CURB_W
    tint = (1.08, 1.0, 0.82)
    for side in (-1, 1):
        for i in range(T.N):
            j = (i + 1) % T.N
            v = 0.90 + 0.20 * (((i * 7919) % 13) / 13.0)
            for k in range(len(PROFILE) - 1):
                o0, d0, c0 = PROFILE[k]
                o1, d1, c1 = PROFILE[k + 1]
                ca = C4(min(c0[0] * v * tint[0], 1), min(c0[1] * v * tint[1], 1), min(c0[2] * v * tint[2], 1), 1)
                cb = C4(min(c1[0] * v * tint[0], 1), min(c1[1] * v * tint[1], 1), min(c1[2] * v * tint[2], 1), 1)
                mb.quadc(T.pt(i, side * (E + o0), d0), T.pt(i, side * (E + o1), d1),
                         T.pt(j, side * (E + o1), d1), T.pt(j, side * (E + o0), d0), ca, cb, cb, ca)
    return mb


def build_rails(T):
    mb = MB()
    for side in (-1, 1):
        for i in range(T.N):
            j = (i + 1) % T.N
            pan = C4(0.80, 0.80, 0.84, 1) if i % 2 == 0 else C4(0.70, 0.72, 0.78, 1)
            lo, hi = -0.02, 0.95
            mb.quad(T.pt(i, side * WALL_LAT, lo + 0.30), T.pt(i, side * WALL_LAT, hi),
                    T.pt(j, side * WALL_LAT, hi), T.pt(j, side * WALL_LAT, lo + 0.30), pan)
            top = C4(0.85, 0.10, 0.08, 1) if i % 2 == 0 else C4(0.93, 0.93, 0.93, 1)
            mb.quad(T.pt(i, side * WALL_LAT, hi), T.pt(i, side * (WALL_LAT + 0.3), hi + 0.04),
                    T.pt(j, side * (WALL_LAT + 0.3), hi + 0.04), T.pt(j, side * WALL_LAT, hi), top)
            if i % 2 == 0:
                x, y, z = T.pt(i, side * (WALL_LAT + 0.2))
                mb.box(x, y + 0.45, z, 0.16, 0.95, 0.16, (0.25, 0.25, 0.28), yaw=T.H[i])
    return mb


def build_scenery(T):
    mb = MB()
    E = ROAD_HALF + CURB_W
    rnd.seed(7)
    for i in range(0, T.N, 2):
        for side in (-1, 1):
            if rnd.random() < 0.35:
                continue
            for _ in range(rnd.choice((1, 2, 2))):
                off = rnd.uniform(9.0, 52.0)
                lat = side * (E + off)
                x, y, z = T.pt(i, lat, ground_drop(off))
                x += rnd.uniform(-3, 3)
                z += rnd.uniform(-3, 3)
                h = rnd.uniform(6.0, 13.0)
                g = rnd.uniform(0.75, 1.15)
                grn = (0.10 * g, 0.26 * g, 0.12 * g)
                mb.box(x, y + 0.7, z, 0.45, 1.4, 0.45, (0.30, 0.20, 0.12))
                cone(mb, x, y + 1.0, z, h * 0.33, h * 0.55, 7, grn, jitter=0.12)
                cone(mb, x, y + 1.0 + h * 0.30, z, h * 0.25, h * 0.55, 7, grn, jitter=0.12)
    for _ in range(70):  # boulders
        i = rnd.randrange(T.N)
        side = rnd.choice((-1, 1))
        off = rnd.uniform(6.0, 40.0)
        x, y, z = T.pt(i, side * (E + off), ground_drop(off))
        r = rnd.uniform(0.8, 2.4)
        ctr = (x, y + r * 0.4, z)
        top = (x + rnd.uniform(-.3, .3), y + r * 0.9, z + rnd.uniform(-.3, .3))
        ring = [(x + math.cos(a) * r * rnd.uniform(.8, 1.2), y, z + math.sin(a) * r * rnd.uniform(.8, 1.2))
                for a in [math.tau * k / 6 for k in range(6)]]
        rock = rnd.uniform(0.42, 0.55)
        for k in range(6):
            mb.tri_lit(ring[k], ring[(k + 1) % 6], top, (rock, rock * 0.94, rock * 0.88), ctr)
    return mb


def build_landmarks(T):
    mb = MB()
    H = ROAD_HALF + CURB_W
    yaw = T.H[0]
    x0, y0, z0 = T.CX[0], T.CY[0], T.CZ[0]
    rx, rz = T.RX[0], T.RZ[0]
    # start / finish gantry
    for side in (-1, 1):
        mb.box(x0 + rx * side * (H + 1.8), y0 + 4.5, z0 + rz * side * (H + 1.8), 1.2, 9.0, 1.2,
               (0.16, 0.16, 0.2), yaw=yaw)
    mb.box(x0, y0 + 8.6, z0, 2 * (H + 2.4), 1.5, 1.3, (0.10, 0.10, 0.13), yaw=yaw)
    cells = 28
    for c in range(cells):
        for r in range(2):
            lo = -(H + 2.4) + c * (2 * (H + 2.4) / cells)
            hi = lo + 2 * (H + 2.4) / cells
            col = C4(0.95, 0.95, 0.95, 1) if (c + r) % 2 == 0 else C4(0.04, 0.04, 0.05, 1)
            yb, yt = y0 + 7.95 + r * 0.7, y0 + 8.65 + r * 0.7
            for fz in (-0.68, 0.68):
                ax, az = x0 + rx * lo + T.TX[0] * fz, z0 + rz * lo + T.TZ[0] * fz
                bx, bz = x0 + rx * hi + T.TX[0] * fz, z0 + rz * hi + T.TZ[0] * fz
                mb.quad((ax, yb, az), (bx, yb, bz), (bx, yt, bz), (ax, yt, az), col)
    # grandstand on the left of the start straight
    for k in range(-7, 8):
        i = k % T.N
        for tier in range(4):
            x, y, z = T.pt(i, -(H + 22.0 + tier * 3.2))
            mb.box(x, y + 1.2 + tier * 1.5, z, 3.0, 2.4 + tier * 3.0, 8.0, (0.55, 0.55, 0.6), yaw=T.H[i])
            for s in range(3):
                cc = rnd.choice(((0.9, 0.2, 0.15), (0.2, 0.45, 0.9), (0.95, 0.85, 0.2), (0.9, 0.9, 0.9), (0.2, 0.7, 0.35)))
                xx, yy, zz = T.pt(i, -(H + 22.0 + tier * 3.2) + 1.52, 0.0)
                mb.box(xx, y + 2.6 + tier * 3.0 * 0.5 + 0.0, zz + (s - 1) * 2.3 * T.TZ[i], 0.2, 0.9, 1.6, cc, yaw=T.H[i], flat=True)
    return mb


def build_mountains(T):
    mb = MB()
    rnd.seed(21)
    for k in range(46):
        a = math.tau * k / 46 + rnd.uniform(-0.05, 0.05)
        d = rnd.uniform(1150.0, 1650.0)
        cx, cz = T.cx + math.cos(a) * d, T.cz + math.sin(a) * d
        r = rnd.uniform(220.0, 420.0)
        h = rnd.uniform(160.0, 360.0)
        base = -36.0
        apex = (cx + rnd.uniform(-40, 40), base + h, cz + rnd.uniform(-40, 40))
        sides = 6
        ring = [(cx + math.cos(t) * r * rnd.uniform(.85, 1.15), base, cz + math.sin(t) * r * rnd.uniform(.85, 1.15))
                for t in [math.tau * q / sides for q in range(sides)]]
        tone = rnd.uniform(0.85, 1.1)
        for q in range(sides):
            mb.tri_lit(ring[q], ring[(q + 1) % sides], apex, (0.50 * tone, 0.42 * tone, 0.46 * tone),
                       (cx, base + h * 0.3, cz))
    return mb


def build_sky(radius=2500.0):
    mb = MB()
    rings = [-18, -6, -2, 0, 2, 5, 9, 14, 22, 34, 50, 70, 90]
    segs = 48
    hor, mid, zen = FOG_COL, (0.55, 0.66, 0.88), (0.16, 0.30, 0.62)

    def sky_col(el, az):
        e = math.radians(el)
        d = (math.cos(e) * math.sin(az), math.sin(e), math.cos(e) * math.cos(az))
        t = max(0.0, d[1])
        if t < 0.35:
            c = mix3(hor, mid, (t / 0.35) ** 0.6)
        else:
            c = mix3(mid, zen, ((t - 0.35) / 0.65) ** 0.8)
        dot = max(0.0, d[0] * SUN_DIR[0] + d[1] * SUN_DIR[1] + d[2] * SUN_DIR[2])
        glow = dot ** 9 * 0.55 + dot ** 60 * 0.6
        return C4(min(c[0] + 1.0 * glow, 1.0), min(c[1] + 0.62 * glow, 1.0), min(c[2] + 0.30 * glow, 1.0), 1)

    def P(el, az):
        e = math.radians(el)
        return (radius * math.cos(e) * math.sin(az), radius * math.sin(e), radius * math.cos(e) * math.cos(az))

    for r in range(len(rings) - 1):
        for s in range(segs):
            a0, a1 = math.tau * s / segs, math.tau * (s + 1) / segs
            e0, e1 = rings[r], rings[r + 1]
            mb.quadc(P(e0, a0), P(e0, a1), P(e1, a1), P(e1, a0),
                     sky_col(e0, a0), sky_col(e0, a1), sky_col(e1, a1), sky_col(e1, a0))
    return mb


def build_sea():
    mb = MB()
    s = 5200.0
    c = C4(0.06, 0.25, 0.38, 1)
    mb.quad((-s, -30.0, -s), (-s, -30.0, s), (s, -30.0, s), (s, -30.0, -s), c)
    return mb


# =====================================================================================
#  PARTICLES (pooled) & SKID MARKS (single dynamic mesh)
# =====================================================================================
SMOKE, SPARK, FLAME = 0, 1, 2


class Particles:
    def __init__(self, n=160):
        self.ents = []
        for _ in range(n):
            e = Entity(model='circle', billboard=True, enabled=False, color=C4(1, 1, 1, 0))
            self.ents.append(e)
        self.data = [None] * n
        self.free = list(range(n - 1, -1, -1))
        self.active = []

    def emit(self, kind, x, y, z, vx, vy, vz, life, s0, s1, rgb, a0=1.0):
        if not self.free:
            return
        i = self.free.pop()
        e = self.ents[i]
        e.position = (x, y, z)
        e.scale = s0
        e.color = C4(rgb[0], rgb[1], rgb[2], a0)
        e.enabled = True
        self.data[i] = [life, life, vx, vy, vz, s0, s1, rgb, a0, kind, x, y, z]
        self.active.append(i)

    def update(self, dt):
        keep = []
        for i in self.active:
            d = self.data[i]
            d[0] -= dt
            e = self.ents[i]
            if d[0] <= 0.0:
                e.enabled = False
                self.free.append(i)
                continue
            t = 1.0 - d[0] / d[1]
            kind = d[9]
            if kind == SPARK:
                d[3] -= 16.0 * dt
            elif kind == SMOKE:
                k = max(0.0, 1.0 - 1.6 * dt)
                d[2] *= k
                d[4] *= k
                d[3] = d[3] * k + 0.5 * dt
            d[10] += d[2] * dt
            d[11] += d[3] * dt
            d[12] += d[4] * dt
            e.position = (d[10], d[11], d[12])
            e.scale = d[5] + (d[6] - d[5]) * t
            rgb = d[7]
            e.color = C4(rgb[0], rgb[1], rgb[2], d[8] * (1.0 - t) ** (1.0 if kind == SMOKE else 1.6))
            keep.append(i)
        self.active = keep


class Skids:
    MAXQ = 450

    def __init__(self):
        self.q = deque(maxlen=self.MAXQ)
        self.dirty = False
        self.timer = 0.0
        pad = [(0.0, -999.0, 0.0)] * (self.MAXQ * 6)
        self.mesh = Mesh(vertices=pad, mode='triangle')
        self.ent = Entity(model=self.mesh, color=C4(0.02, 0.02, 0.02, 0.62))
        self.ent.set_two_sided(True)
        self.ent.set_depth_offset(3)

    def add(self, a, b, c, d):
        self.q.append((a, b, c, d))
        self.dirty = True

    def update(self, dt):
        self.timer -= dt
        if self.dirty and self.timer <= 0.0:
            v = []
            for (a, b, c, d) in self.q:
                v += [a, b, c, a, c, d]
            v += [(0.0, -999.0, 0.0)] * (self.MAXQ * 6 - len(v))
            self.mesh.vertices = v
            self.mesh.generate()
            self.dirty = False
            self.timer = 0.15


# =====================================================================================
#  VEHICLE PHYSICS CONSTANTS
# =====================================================================================
MASS = 1300.0
IZ = 2300.0
A_F = 1.30           # CG -> front axle
B_R = 1.40           # CG -> rear axle
WB = A_F + B_R
HCG = 0.52
WR = 0.33
FINAL = 3.7
GEARS = [3.40, 2.35, 1.78, 1.40, 1.15, 0.96]
IDLE = 1000.0
MAXRPM = 8000.0
PEAK_T = 450.0
MU_F = 1.52
MU_R = 1.48
CT = 1.4
BT = 9.5
G = 9.81


def engine_torque(rpm):
    r = rpm / 1000.0
    if r < 1.0:
        t = 0.55
    elif r < 5.5:
        t = 0.55 + 0.45 * math.sin((r - 1.0) / 4.5 * math.pi / 2.0)
    else:
        t = 1.0 - 0.38 * ((r - 5.5) / 2.5) ** 1.4
    return max(0.0, t) * PEAK_T


def max_steer(v):
    return 0.62 / (1.0 + (v / 22.0) ** 1.6)


def fmt_time(t):
    if t is None or t >= 1e8:
        return '--:--.---'
    m = int(t // 60)
    s = t - m * 60
    return '%d:%06.3f' % (m, s)


# =====================================================================================
#  CAR
# =====================================================================================
class Car:
    def __init__(self, game, name, paint, stripe, is_player, skill, lane_bias):
        self.g = game
        self.name = name
        self.paint = paint
        self.is_player = is_player
        self.skill = skill
        self.lane_bias = lane_bias
        self.ui_col = C4(paint[0], paint[1], paint[2], 1)

        self.root = Entity()
        self.body = Entity(parent=self.root, model=build_car_mesh(paint, stripe))
        game.skin(self.body)
        self.body.set_two_sided(True)
        self.pivots, self.wheels = [], []
        for sx in (-1, 1):
            for lz, front in ((1.38, True), (-1.32, False)):
                piv = Entity(parent=self.body, position=(sx * 0.90, 0.34, lz))
                w = Entity(parent=piv, model=build_wheel_mesh(0.28 if front else 0.33))
                game.skin(w)
                w.set_two_sided(True)
                self.pivots.append((piv, front))
                self.wheels.append(w)
        self.shadow = Entity(parent=self.root, model='quad', rotation_x=90, scale=(2.4, 5.0), y=0.05,
                             color=C4(0, 0, 0, 0.42))
        self.shadow.set_depth_offset(2)
        self.reset_state()

    # ------------------------------------------------------------------ state
    def reset_state(self):
        self.x = self.z = self.h = 0.0
        self.y = 0.0
        self.vx = self.vz = 0.0
        self.w = 0.0
        self.steer = 0.0
        self.steer_in = 0.0
        self.steer_target = 0.0
        self.in_thr = self.in_brk = self.in_rev = self.in_hand = 0.0
        self.boosting = False
        self.nitro = 1.0
        self.gear = 1
        self.rpm = IDLE
        self.shift_t = 0.0
        self.ax_s = self.ay_s = 0.0
        self.vlong = self.vlat = self.speed = 0.0
        self.slip = 0.0
        self.wheelspin = 0.0
        self.spin = 0.0
        self.idx = 0
        self.progress = 0
        self.lat = self.along = 0.0
        self.on_road = True
        self.laps_done = 0
        self.lap_start = 0.0
        self.lap_times = []
        self.best = 1e9
        self.finished = False
        self.finish_time = None
        self.lane = 0.0
        self.lane_t = 0.0
        self.lane_timer = 0.0
        self.stuck_t = 0.0
        self.pitch_road = 0.0
        self.prev_thr = 0.0
        self.last_skid = [None, None]
        self.rev_mode = False

    def place_on_grid(self, slot):
        T = self.g.track
        row, col = slot // 2, slot % 2
        back = 12.0 + row * 9.5 + col * 4.5
        x, y, z, tx, tz = T.sample(T.length - back)
        lat = -3.6 if col == 0 else 3.6
        self.x = x + tz * lat
        self.z = z - tx * lat
        self.h = math.atan2(tx, tz)
        self.vx = self.vz = self.w = 0.0
        self.idx = T.nearest(self.x, self.z)
        self.progress = -int(round(back / T.spacing))
        self.update_track()
        self.y = y
        self.sync_visual(1 / 60.0, True)

    def reset_to_track(self):
        T = self.g.track
        i = T.nearest(self.x, self.z)
        self.x, self.z = T.CX[i], T.CZ[i]
        self.h = T.H[i]
        self.vx = self.vz = self.w = 0.0
        self.steer = self.steer_in = 0.0
        self.gear = 1
        self.rpm = IDLE
        self.idx = i
        self.update_track()
        self.sync_visual(1 / 60.0, True)

    # ------------------------------------------------------------------ track tracking
    def update_track(self):
        T = self.g.track
        N = T.N
        best, bi = 1e18, self.idx
        for k in range(-6, 13):
            j = (self.idx + k) % N
            d = (self.x - T.CX[j]) ** 2 + (self.z - T.CZ[j]) ** 2
            if d < best:
                best, bi = d, j
        if best > 3600.0:
            bi = T.nearest(self.x, self.z)
        delta = bi - self.idx
        if delta > N // 2:
            delta -= N
        elif delta < -N // 2:
            delta += N
        self.progress += delta
        self.idx = bi
        dx, dz = self.x - T.CX[bi], self.z - T.CZ[bi]
        self.lat = dx * T.RX[bi] + dz * T.RZ[bi]
        self.along = dx * T.TX[bi] + dz * T.TZ[bi]
        self.y = T.CY[bi] + T.SL[bi] * self.along
        self.on_road = abs(self.lat) < ROAD_HALF + CURB_W - 0.2

    def wall_collision(self):
        T = self.g.track
        lim = WALL_LAT - 1.0
        if abs(self.lat) <= lim:
            return
        i = self.idx
        sgn = 1.0 if self.lat > 0 else -1.0
        over = abs(self.lat) - lim
        rx, rz = T.RX[i], T.RZ[i]
        self.x -= sgn * rx * over
        self.z -= sgn * rz * over
        nx, nz = -sgn * rx, -sgn * rz
        vn = self.vx * nx + self.vz * nz
        if vn < 0.0:
            imp = -vn
            self.vx -= 1.28 * vn * nx
            self.vz -= 1.28 * vn * nz
            damp = 1.0 - min(0.30, imp * 0.018)
            self.vx *= damp
            self.vz *= damp
            self.w += -sgn * (1.0 if self.vlong >= 0 else -1.0) * imp * 0.035
            self.g.on_impact(self, imp, self.x + sgn * rx * 1.0, self.y + 0.5, self.z + sgn * rz * 1.0, nx, nz)

    # ------------------------------------------------------------------ controllers
    def player_controls(self, dt, revonly=False):
        k = held_keys
        up = k['w'] or k['up arrow']
        dn = k['s'] or k['down arrow']
        lf = k['a'] or k['left arrow']
        rt = k['d'] or k['right arrow']
        thr_t = 1.0 if up else 0.0
        brk_t, rev_t = 0.0, 0.0
        if dn:
            if self.vlong > 1.2:
                brk_t = 1.0
            else:
                rev_t = 1.0
        if up and self.vlong < -1.2:
            brk_t, thr_t = 1.0, 0.0
        self.in_thr += clampf(thr_t - self.in_thr, -9.0 * dt, 6.0 * dt)
        self.in_brk = brk_t
        self.in_rev = rev_t
        self.in_hand = 1.0 if k['space'] else 0.0
        tgt = (1.0 if rt else 0.0) - (1.0 if lf else 0.0)
        rate = 5.5 if tgt != 0.0 else 9.0
        self.steer_in += clampf(tgt - self.steer_in, -rate * dt, rate * dt)
        self.steer_target = self.steer_in * max_steer(self.speed)
        if not revonly:
            want_boost = k['left shift'] and self.nitro > 0.02 and self.in_thr > 0.2
            self.boosting = bool(want_boost)

    def ai_control(self, dt):
        g = self.g
        T = g.track
        N = T.N
        i = self.idx
        spd = self.speed
        la = max(3, int((12.0 + spd * 0.55) / T.spacing))
        self.lane_timer -= dt
        if self.lane_timer <= 0.0:
            self.lane_timer = 0.25
            dh = wrap_pi(T.H[(i + 6) % N] - T.H[(i + 1) % N])
            base = clampf(dh * 9.0, -4.2, 4.2) + self.lane_bias
            for o in g.cars:
                if o is self:
                    continue
                dm = (o.progress - self.progress) * T.spacing
                if 1.0 < dm < 34.0 and abs(o.lat - self.lat) < 3.2:
                    side = 1.0 if o.lat < self.lat else -1.0
                    if abs(o.lat - self.lat) < 0.6:
                        side = -1.0 if self.lat > 0 else 1.0
                    base = clampf(o.lat + side * 4.0, -5.5, 5.5)
                    break
            self.lane_t = base
        self.lane += (self.lane_t - self.lane) * min(1.0, dt * 2.2)
        j = (i + la) % N
        tx = T.CX[j] + T.RX[j] * self.lane
        tz = T.CZ[j] + T.RZ[j] * self.lane
        err = wrap_pi(math.atan2(tx - self.x, tz - self.z) - self.h)
        ms = max_steer(spd)
        self.steer_target = clampf(err * 1.1, -ms, ms)
        # target speed
        vt = T.prof[(i + 3) % N] * self.skill
        if g.player is not None and not self.is_player:
            delta = (g.player.progress - self.progress) * T.spacing
            vt *= 1.0 + clampf(delta * 0.0004, -0.07, 0.05)
        if not self.on_road:
            vt = min(vt, 32.0)
        for o in g.cars:
            if o is self:
                continue
            dm = (o.progress - self.progress) * T.spacing
            if 0.5 < dm < 14.0 and abs(o.lat - self.lat) < 2.3:
                vt = min(vt, o.speed + 2.0)
        if spd < vt:
            self.in_thr = clampf((vt - spd) / 6.0 + 0.2, 0.0, 1.0)
            self.in_brk = 0.0
        else:
            self.in_thr = 0.0
            self.in_brk = clampf((spd - vt) / 7.0, 0.0, 1.0)
        if self.slip > 0.6:
            self.in_thr *= 0.5
        self.in_rev = 0.0
        self.in_hand = 0.0
        self.boosting = bool(vt > 58.0 and spd > 45.0 and self.nitro > 0.35 and T.K[i] < 0.002)
        # stuck recovery
        if spd < 2.0 and g.state == 'racing':
            self.stuck_t += dt
            if self.stuck_t > 2.5:
                self.stuck_t = 0.0
                self.reset_to_track()
        else:
            self.stuck_t = 0.0

    def apply_steer(self, dt):
        tgt = self.steer_target
        if self.speed > 9.0 and self.vlong > 0.0:
            beta = math.atan2(self.vlat, max(self.vlong, 1.0))
            assist = clampf(beta * 0.9, -0.30, 0.30)
            tgt = clampf(tgt + assist * (1.0 - 0.5 * min(1.0, abs(self.steer_in))), -0.55, 0.55)
        self.steer += (tgt - self.steer) * min(1.0, dt * 9.0)

    def nitro_logic(self, dt):
        if self.boosting and self.nitro > 0.0:
            self.nitro = max(0.0, self.nitro - 0.22 * dt)
            if self.nitro <= 0.0:
                self.boosting = False
        else:
            self.boosting = False
            self.nitro = min(1.0, self.nitro + (0.035 + 0.07 * min(1.0, self.slip)) * dt)

    def countdown_revs(self, dt):
        tgt = IDLE + self.in_thr * 4800.0
        self.rpm += (tgt - self.rpm) * min(1.0, dt * 8.0)
        self.speed = 0.0
        self.update_track()

    # ------------------------------------------------------------------ physics (one sub-step)
    def physics(self, dt):
        h = self.h
        sh, ch = math.sin(h), math.cos(h)
        vl = self.vx * sh + self.vz * ch
        vt = self.vx * ch - self.vz * sh
        speed = math.hypot(self.vx, self.vz)
        surf = 1.0 if self.on_road else 0.62

        fz_tot = MASS * G + 0.9 * vl * vl
        fzf = max(fz_tot * B_R / WB - MASS * self.ax_s * HCG / WB, 2000.0)
        fzr = max(fz_tot * A_F / WB + MASS * self.ax_s * HCG / WB, 2000.0)

        thr, brk, rev, hb = self.in_thr, self.in_brk, self.in_rev, self.in_hand

        # ---- gearbox / engine
        gear = self.gear
        ratio = GEARS[gear - 1] * FINAL
        rpm_w = abs(vl) / WR * ratio * 9.5493
        tgt = max(IDLE, rpm_w)
        if thr > 0.05:
            tgt = max(tgt, IDLE + thr * 3600.0 * (1.0 - min(1.0, abs(vl) / 10.0)))
        self.rpm += (tgt - self.rpm) * min(1.0, dt * 16.0)
        if self.shift_t > 0.0:
            self.shift_t -= dt
        elif vl > 1.0:
            up_rpm = 5200.0 + thr * 1900.0
            down_rpm = 2400.0 + brk * 900.0
            if self.rpm > up_rpm and gear < 6:
                self.gear += 1
                self.shift_t = 0.17
                self.g.on_shift(self)
            elif gear > 1 and (self.rpm < down_rpm or (thr > 0.9 and self.rpm < 3800.0)):
                self.gear -= 1
                self.shift_t = 0.05
        elif gear > 1 and vl < 3.0:
            self.gear = 1
        limiter = MAXRPM - 100.0 + (450.0 if self.boosting else 0.0)
        tq = engine_torque(self.rpm) if (self.shift_t <= 0.0 and self.rpm < limiter) else 0.0
        boost = 1.38 if self.boosting else 1.0
        fdrive = thr * tq * ratio * 0.88 / WR * boost
        if rev > 0.0:
            fdrive = -rev * 4800.0 * (1.0 if vl > -14.0 else 0.0)

        # ---- traction (friction circle at the driven rear axle)
        fmax_r = MU_R * fzr * surf
        fd = clampf(fdrive, -fmax_r, fmax_r)
        used_r = min(1.0, abs(fd) / max(fmax_r, 1.0))
        wheelspin = max(0.0, abs(fdrive) - fmax_r) / max(fmax_r, 1.0) if abs(vl) < 40.0 else 0.0

        # ---- braking
        fb_cmd = brk * 17500.0
        fb_max = (MU_F * fzf * 0.95 + MU_R * fzr * 0.85) * surf
        fb = min(fb_cmd, fb_max)
        used_f = min(1.0, fb * 0.64 / max(MU_F * fzf * surf, 1.0))
        used_rb = min(1.0, fb * 0.36 / max(MU_R * fzr * surf, 1.0))
        lat_f = max(0.40, math.sqrt(max(0.0, 1.0 - 0.85 * used_f * used_f)))
        used_rt = min(1.0, math.sqrt(used_r * used_r + used_rb * used_rb))
        lat_r = max(0.45, math.sqrt(max(0.0, 1.0 - 0.85 * used_rt * used_rt)))
        fhand = hb * MASS * 6.5 if abs(vl) > 0.5 else 0.0
        sgn = 1.0 if vl > 0.0 else -1.0
        fx_brake = -sgn * min(fb + fhand, MASS * abs(vl) / dt * 0.95)

        # ---- resistance
        cd = 0.43 * (0.84 if self.boosting else 1.0)
        fdrag = -cd * vl * abs(vl)
        frr = 0.0
        if abs(vl) > 0.05:
            frr = -sgn * min(0.016 * MASS * G * (1.0 if self.on_road else 6.0), MASS * abs(vl) / dt * 0.5)

        # ---- tyre forces (slip-angle based)
        vden = max(abs(vl), 3.0)
        af = math.atan2(vt + A_F * self.w - vl * math.tan(self.steer), vden)
        ar = math.atan2(vt - B_R * self.w, vden)
        hbf = 0.38 if hb > 0.5 else 1.0
        fyf = -fzf * MU_F * surf * lat_f * math.sin(CT * math.atan(BT * af))
        fyr = -fzr * MU_R * surf * lat_r * hbf * math.sin(CT * math.atan(BT * ar))
        fyx = -MASS * vt * clampf((4.0 - speed) / 4.0, 0.0, 1.0) * 8.0

        fx = fd + fx_brake + fdrag + frr - fyf * math.sin(self.steer)
        fy = fyf * math.cos(self.steer) + fyr + fyx
        axb, ayb = fx / MASS, fy / MASS
        self.vx += (axb * sh + ayb * ch) * dt
        self.vz += (axb * ch - ayb * sh) * dt

        torque = A_F * fyf * math.cos(self.steer) - B_R * fyr - self.w * 2200.0
        if speed < 3.0:
            torque -= self.w * IZ * 5.0
        self.w = clampf(self.w + torque / IZ * dt, -3.5, 3.5)
        self.h += self.w * dt
        self.x += self.vx * dt
        self.z += self.vz * dt

        k = min(1.0, dt * 7.0)
        self.ax_s += (axb - self.ax_s) * k
        self.ay_s += (ayb - self.ay_s) * k
        self.vlong, self.vlat = vl, vt
        self.speed = math.hypot(self.vx, self.vz)
        self.wheelspin = wheelspin
        slip = max(abs(ar) - 0.10, 0.0) * clampf(speed / 12.0, 0.0, 1.0) * 4.0
        slip += wheelspin * 0.9 * (1.0 - clampf(speed / 40.0, 0.0, 1.0))
        slip += hb * clampf(speed / 10.0, 0.0, 1.0) * 0.8
        slip += max(0.0, fb - 0.9 * fb_max) / max(fb_max, 1.0) * 2.0 * clampf(speed / 20.0, 0.0, 1.0)
        self.slip = slip

    # ------------------------------------------------------------------ visuals
    def wheel_world(self, lx, lz):
        c, s = math.cos(self.h), math.sin(self.h)
        return (self.x + lx * c + lz * s, self.z - lx * s + lz * c)

    def sync_visual(self, dt, snap=False):
        T = self.g.track
        self.root.position = (self.x, self.y, self.z)
        self.root.rotation_y = math.degrees(self.h)
        i = self.idx
        along = T.TX[i] * math.sin(self.h) + T.TZ[i] * math.cos(self.h)
        tgt = -math.degrees(math.atan(T.SL[i] * along))
        self.pitch_road = tgt if snap else self.pitch_road + (tgt - self.pitch_road) * min(1.0, dt * 10.0)
        self.root.rotation_x = self.pitch_road
        self.body.rotation_x = clampf(-self.ax_s * 0.32, -3.5, 4.0)
        self.body.rotation_z = clampf(self.ay_s * 0.45, -5.0, 5.0)
        cp = camera.world_position
        near = (cp.x - self.x) ** 2 + (cp.z - self.z) ** 2 < 140.0 ** 2
        if near:
            self.spin += math.degrees((self.vlong / WR + self.wheelspin * 40.0) * dt)
            for piv, front in self.pivots:
                if front:
                    piv.rotation_y = math.degrees(self.steer)
            for w in self.wheels:
                w.rotation_x = self.spin

    def effects(self, dt):
        g = self.g
        P = g.particles
        near = (camera.world_position.x - self.x) ** 2 + (camera.world_position.z - self.z) ** 2 < 220.0 ** 2
        if not near:
            return
        c, s = math.cos(self.h), math.sin(self.h)
        # tyre smoke + skid marks
        if self.slip > 0.18 and self.on_road:
            for k, lx in enumerate((-0.85, 0.85)):
                wx, wz = self.wheel_world(lx, -1.32)
                wy = self.y + 0.06
                if rnd.random() < min(1.0, self.slip * 0.9) * (1.0 if self.is_player else 0.4):
                    P.emit(SMOKE, wx, wy + 0.2, wz, self.vx * 0.15 + rnd.uniform(-1, 1), rnd.uniform(0.3, 1.0),
                           self.vz * 0.15 + rnd.uniform(-1, 1), rnd.uniform(0.9, 1.6), 0.9, rnd.uniform(3.0, 5.0),
                           (0.90, 0.86, 0.82), 0.55)
                if self.is_player or self.slip > 0.7:
                    last = self.last_skid[k]
                    if last is not None:
                        dx, dz = wx - last[0], wz - last[1]
                        d = math.hypot(dx, dz)
                        if 0.35 < d < 4.0:
                            px, pz = -dz / d * 0.15, dx / d * 0.15
                            g.skids.add((last[0] + px, wy, last[1] + pz), (last[0] - px, wy, last[1] - pz),
                                        (wx - px, wy, wz - pz), (wx + px, wy, wz + pz))
                            self.last_skid[k] = (wx, wz)
                    else:
                        self.last_skid[k] = (wx, wz)
        else:
            self.last_skid = [None, None]
        # off-road dust
        if not self.on_road and self.speed > 10.0 and rnd.random() < 0.5:
            P.emit(SMOKE, self.x + rnd.uniform(-1, 1), self.y + 0.3, self.z + rnd.uniform(-1, 1),
                   rnd.uniform(-1, 1), 1.0, rnd.uniform(-1, 1), 1.0, 1.0, 4.0, (0.55, 0.45, 0.30), 0.4)
        # backfire on lift-off / nitro flames
        ex = self.wheel_world(0.45, -2.25)
        if self.prev_thr > 0.7 and self.in_thr < 0.2 and self.rpm > 5500.0 and rnd.random() < 0.6:
            for lx in (-0.45, 0.45):
                fx, fz = self.wheel_world(lx, -2.25)
                P.emit(FLAME, fx, self.y + 0.35, fz, -s * 3.0, 0.5, -c * 3.0, 0.16, 0.5, 0.15, (1.0, 0.55, 0.15), 1.0)
        if self.boosting:
            for lx in (-0.45, 0.45):
                fx, fz = self.wheel_world(lx, -2.3)
                P.emit(FLAME, fx, self.y + 0.35, fz, -s * 8.0 + self.vx, 0.2, -c * 8.0 + self.vz, 0.14, 0.6, 0.2,
                       (0.45, 0.70, 1.0), 0.9)
        self.prev_thr = self.in_thr


# =====================================================================================
#  GAME
# =====================================================================================
class Game:
    def __init__(self):
        Text.default_resolution = 1080 * Text.size
        self.shader = Shader(language=Shader.GLSL, vertex=VERT, fragment=FRAG) if USE_CUSTOM_SHADER else None
        self.sheen = []
        self.track = Track()
        self.build_world()
        self.particles = Particles()
        self.skids = Skids()
        self.cars = []
        self.player = None
        self.state = 'countdown'
        self.count = 3.99
        self.race_time = 0.0
        self.finish_order = []
        self.order = []
        self.order_timer = 0.0
        self.cam_mode = 0
        self.cam_pos = Vec3(0, 5, 0)
        self.cam_vel = Vec3(0, 0, 0)
        self.cam_h = 0.0
        self.cam_roll = 0.0
        self.shake = 0.0
        self.shake_dir = (0.0, 0.0)
        self.chroma = 0.0
        self.msg_t = 0.0
        self.clock = 0.0
        self.fx_tick = 0
        self.hud_cache = {}
        self.build_cars()
        self.build_hud()
        self.build_fx()
        camera.clip_plane_far = 6000
        camera.clip_plane_near = 0.3
        self.start_race()

    # ------------------------------------------------------------------ world
    def skin(self, e, grain=0.0, wet=0.0, fogged=True):
        if self.shader is None:
            return
        e.shader = self.shader
        e.set_shader_input('fog_data', Vec4(FOG_NEAR if fogged else 1e8, FOG_FAR if fogged else 2e8, grain, wet))
        e.set_shader_input('fog_color', Vec3(*FOG_COL))
        e.set_shader_input('sun_dir', Vec3(*SUN_DIR))
        e.set_shader_input('cam_pos', Vec3(0, 0, 0))

    def static(self, mb, grain=0.0, wet=0.0, fogged=True, offset=0, sheen=False):
        e = Entity(model=mb.build())
        self.skin(e, grain, wet, fogged)
        e.set_two_sided(True)
        if offset:
            e.set_depth_offset(offset)
        if sheen and self.shader is not None:
            self.sheen.append(e)
        return e

    def build_world(self):
        T = self.track
        window.color = C4(FOG_COL[0], FOG_COL[1], FOG_COL[2], 1)
        road, curbs, lines = build_road(T)
        self.static(road, grain=0.55, wet=0.85, sheen=True)
        self.static(curbs, grain=0.25, wet=0.35, offset=1, sheen=True)
        self.static(lines, grain=0.2, wet=0.6, offset=2, sheen=True)
        self.static(build_terrain(T), grain=0.35)
        self.static(build_rails(T))
        self.static(build_scenery(T))
        self.static(build_landmarks(T))
        self.static(build_mountains(T))
        self.static(build_sea(), wet=1.4, sheen=True)
        self.sky = self.static(build_sky(), fogged=False)
        self.sky.set_bin('background', 0)
        self.sky.set_depth_write(False)
        # sun disc + glow (billboards)
        self.sun_glow = Entity(model='circle', billboard=True, scale=520, color=C4(1.0, 0.72, 0.40, 0.22))
        self.sun_disc = Entity(model='circle', billboard=True, scale=110, color=C4(1.0, 0.95, 0.80, 1.0))

    def build_cars(self):
        specs = [('YOU', (0.05, 0.34, 0.95), (0.95, 0.95, 0.97), True, 1.0, 0.0),
                 ('Apex', (0.88, 0.08, 0.08), (0.95, 0.95, 0.95), False, 0.985, 0.8),
                 ('Viper', (0.98, 0.80, 0.08), (0.08, 0.08, 0.1), False, 0.97, -0.9),
                 ('Nova', (0.05, 0.75, 0.85), (0.95, 0.95, 0.95), False, 0.96, 1.2),
                 ('Rogue', (0.10, 0.62, 0.25), (0.95, 0.95, 0.95), False, 0.95, -1.3),
                 ('Blaze', (0.98, 0.45, 0.06), (0.08, 0.08, 0.1), False, 0.94, 0.4),
                 ('Titan', (0.52, 0.22, 0.80), (0.95, 0.95, 0.95), False, 0.93, -0.5)]
        for name, paint, stripe, isp, skill, bias in specs[:NUM_BOTS + 1]:
            c = Car(self, name, paint, stripe, isp, skill, bias)
            self.cars.append(c)
            if isp:
                self.player = c

    # ------------------------------------------------------------------ HUD
    def build_hud(self):
        ui = camera.ui
        gold = C4(1.0, 0.82, 0.25, 1)
        white = C4(0.95, 0.96, 1.0, 1)
        # top-left race info
        Entity(parent=ui, model='quad', color=C4(0, 0, 0, 0.35), scale=(0.40, 0.25), position=(-0.66, 0.37), z=0.4)
        self.t_pos = Text(text='POS 1/7', origin=(-.5, .5), position=(-0.85, 0.485), scale=3.0, color=gold)
        self.t_lap = Text(text='LAP 1/3', origin=(-.5, .5), position=(-0.85, 0.405), scale=1.7, color=white)
        self.t_time = Text(text='TIME 0:00.000', origin=(-.5, .5), position=(-0.85, 0.360), scale=1.5, color=white)
        self.t_best = Text(text='BEST --:--.---', origin=(-.5, .5), position=(-0.85, 0.322), scale=1.2, color=C4(0.7, 1, 0.7, 1))
        self.t_last = Text(text='LAST --:--.---', origin=(-.5, .5), position=(-0.85, 0.290), scale=1.2, color=C4(0.8, 0.8, 0.9, 1))
        # centre messages
        self.t_msg = Text(text='', origin=(0, 0), position=(0, 0.20), scale=6.0, color=gold)
        self.t_sub = Text(text='', origin=(0, 0), position=(0, 0.11), scale=1.6, color=white)
        # speed / tach cluster
        AC, R = (0.64, -0.29), 0.17
        Entity(parent=ui, model='circle', color=C4(0, 0, 0, 0.40), scale=0.44, position=AC, z=0.4)
        self.segs = []
        for i in range(30):
            ang = 210.0 - i * (240.0 / 29.0)
            px = AC[0] + math.cos(math.radians(ang)) * R
            py = AC[1] + math.sin(math.radians(ang)) * R
            e = Entity(parent=ui, model='quad', scale=(0.011, 0.034), position=(px, py), rotation_z=ang - 90.0,
                       color=C4(0.15, 0.15, 0.18, 0.8))
            self.segs.append(e)
        self.seg_key = None
        self.t_speed = Text(text='0', origin=(0, 0), position=(AC[0], AC[1] - 0.01), scale=4.2, color=white)
        self.t_kmh = Text(text='KM/H', origin=(0, 0), position=(AC[0], AC[1] - 0.075), scale=1.1, color=C4(0.7, 0.8, 1, 1))
        self.t_gear = Text(text='1', origin=(0, 0), position=(AC[0], AC[1] + 0.075), scale=3.4, color=gold)
        self.t_shift = Text(text='', origin=(0, 0), position=(AC[0], AC[1] + 0.205), scale=1.5, color=C4(1, 0.2, 0.2, 1))
        # nitro bar
        Entity(parent=ui, model='quad', color=C4(0, 0, 0, 0.45), scale=(0.44, 0.032), position=(0, -0.455), z=0.4)
        self.nitro_bar = Entity(parent=ui, model='quad', origin=(-.5, 0), scale=(0.42, 0.018),
                                position=(-0.21, -0.455), color=C4(0.2, 0.6, 1, 1))
        Text(text='NITRO', origin=(0.5, 0), position=(-0.235, -0.455), scale=1.1, color=C4(0.6, 0.8, 1, 1))
        self.t_hint = Text(text='W/S throttle-brake   A/D steer   SPACE handbrake   SHIFT nitro   C camera   R reset',
                           origin=(-.5, -.5), position=(-0.86, -0.485), scale=0.85, color=C4(1, 1, 1, 0.55))
        # minimap
        T = self.track
        mm_c = (0.72, 0.26)
        Entity(parent=ui, model='quad', color=C4(0, 0, 0, 0.35), scale=(0.34, 0.34), position=mm_c, z=0.4)
        xs = [T.CX[i] for i in range(T.N)]
        zs = [T.CZ[i] for i in range(T.N)]
        rng = max(max(xs) - min(xs), max(zs) - min(zs))
        self.mm_scale = 0.30 / rng
        self.mm_c = mm_c
        mb = MB()
        wcol = C4(0.85, 0.85, 0.9, 1)
        for i in range(0, T.N, 3):
            j = (i + 3) % T.N
            ax, ay = self.mm_xy(T.CX[i], T.CZ[i])
            bx, by = self.mm_xy(T.CX[j], T.CZ[j])
            dx, dy = bx - ax, by - ay
            l = math.hypot(dx, dy) or 1.0
            px, py = -dy / l * 0.0035, dx / l * 0.0035
            mb.quad((ax + px, ay + py, 0), (ax - px, ay - py, 0), (bx - px, by - py, 0), (bx + px, by + py, 0), wcol)
        Entity(parent=ui, model=mb.build(), z=0.3)
        self.dots = []
        for c in self.cars:
            d = Entity(parent=ui, model='circle', scale=0.016 if c.is_player else 0.011, color=c.ui_col, z=0.1 if c.is_player else 0.2)
            self.dots.append(d)
        # results overlay
        self.res_bg = Entity(parent=ui, model='quad', color=C4(0, 0, 0, 0.7), scale=(0.9, 0.7), z=0.05, enabled=False)
        self.res_txt = Text(text='', origin=(0, 0), position=(0, 0.0), scale=1.7, color=white, enabled=False)

    def mm_xy(self, x, z):
        return (self.mm_c[0] + (x - self.track.cx) * self.mm_scale, self.mm_c[1] + (z - self.track.cz) * self.mm_scale)

    def set_txt(self, key, ent, s):
        if self.hud_cache.get(key) != s:
            self.hud_cache[key] = s
            ent.text = s

    def set_msg(self, s, t):
        self.msg_text = s
        self.msg_t = t

    # ------------------------------------------------------------------ screen effects
    def build_fx(self):
        ui = camera.ui
        try:
            from PIL import Image
            size = 128
            px = []
            for y in range(size):
                for x in range(size):
                    dx = x / (size - 1) * 2 - 1
                    dy = y / (size - 1) * 2 - 1
                    d = min(1.0, math.hypot(dx, dy) / 1.25)
                    a = max(0.0, (d - 0.45) / 0.55)
                    a = a * a * (3 - 2 * a)
                    px.append((255, 255, 255, int(255 * a ** 1.2)))
            img = Image.new('RGBA', (size, size))
            img.putdata(px)
            vtex = Texture(img)
        except Exception:
            vtex = None
        kw = dict(parent=ui, model='quad', scale=(2.4, 1.4))
        if vtex is not None:
            self.vig = Entity(texture=vtex, color=C4(0, 0, 0, 0.1), z=1.5, **kw)
            self.boost_vig = Entity(texture=vtex, color=C4(0.15, 0.45, 1.0, 0.0), z=1.45, **kw)
            self.chroma_r = Entity(texture=vtex, color=C4(1, 0.1, 0.1, 0.0), z=1.4, **kw)
            self.chroma_c = Entity(texture=vtex, color=C4(0.1, 0.9, 1.0, 0.0), z=1.4, **kw)
        else:
            self.vig = self.boost_vig = self.chroma_r = self.chroma_c = Entity(color=C4(0, 0, 0, 0), z=1.5, **kw)
        self.flash = Entity(parent=ui, model='quad', scale=(2.4, 1.4), color=C4(1, 1, 1, 0), z=1.3)
        self.glare = Entity(parent=ui, model='quad', scale=(2.4, 1.4), color=C4(1.0, 0.78, 0.45, 0), z=1.35)
        self.lines = []
        for k in range(40):
            pv = Entity(parent=ui, z=1.6, rotation_z=k * 9.0 + rnd.uniform(-3, 3))
            ln = Entity(parent=pv, model='quad', origin=(0, -0.5), y=0.4, scale=(0.003, 0.1), color=C4(1, 1, 1, 0), enabled=False)
            self.lines.append(ln)
        self.flares = []
        for t, sz, col in ((0.0, 0.26, (1.0, 0.85, 0.55)), (0.35, 0.07, (1.0, 0.7, 0.3)), (0.7, 0.12, (0.6, 0.8, 1.0)),
                           (1.15, 0.05, (1.0, 0.6, 0.3)), (1.6, 0.16, (0.5, 1.0, 0.7)), (2.1, 0.09, (1.0, 0.5, 0.8))):
            e = Entity(parent=ui, model='circle', scale=sz, color=C4(col[0], col[1], col[2], 0), z=1.25)
            self.flares.append((e, t, col))

    def update_fx(self, dt):
        p = self.player
        spd = p.speed
        sf = clampf((spd - 28.0) / 50.0, 0.0, 1.0)
        boost = 1.0 if p.boosting else 0.0
        self.fx_tick += 1
        for k, ln in enumerate(self.lines):
            if (k + self.fx_tick) % 2:
                continue
            if sf <= 0.03:
                if ln.enabled:
                    ln.enabled = False
                continue
            ln.enabled = True
            ln.y = rnd.uniform(0.28, 0.55)
            ln.scale = (0.0022 + 0.002 * sf, rnd.uniform(0.06, 0.22) * (0.5 + sf))
            a = sf * rnd.uniform(0.10, 0.30) * (1.4 if boost else 1.0)
            ln.color = C4(0.7, 0.85, 1.0, a) if boost else C4(1.0, 0.95, 0.88, a)
        g_long = clampf(abs(p.ax_s) / 12.0, 0.0, 1.0)
        self.vig.color = C4(0, 0, 0, 0.10 + 0.42 * sf + 0.15 * g_long)
        self.boost_vig.color = C4(0.15, 0.45, 1.0, 0.35 * boost)
        self.chroma = max(0.0, self.chroma - dt * 2.4)
        c = self.chroma
        self.chroma_r.x = 0.014 * c
        self.chroma_c.x = -0.014 * c
        self.chroma_r.color = C4(1, 0.1, 0.1, 0.55 * c)
        self.chroma_c.color = C4(0.1, 0.9, 1.0, 0.55 * c)
        self.flash.color = C4(1, 1, 1, 0.22 * c * c)
        # sun glare + lens flare
        f, r, u = camera.forward, camera.right, camera.up
        zc = SUN_DIR[0] * f.x + SUN_DIR[1] * f.y + SUN_DIR[2] * f.z
        inten = 0.0
        if zc > 0.05:
            xc = SUN_DIR[0] * r.x + SUN_DIR[1] * r.y + SUN_DIR[2] * r.z
            yc = SUN_DIR[0] * u.x + SUN_DIR[1] * u.y + SUN_DIR[2] * u.z
            th = math.tan(math.radians(camera.fov) * 0.5)
            tv = th / window.aspect_ratio
            nx, ny = xc / (zc * th), yc / (zc * tv)
            sx, sy = nx * 0.5 * window.aspect_ratio, ny * 0.5
            edge = clampf(1.6 - max(abs(nx), abs(ny)), 0.0, 1.0)
            inten = (zc ** 10) * edge
            for e, t, col in self.flares:
                e.position = (sx * (1.0 - t), sy * (1.0 - t))
                e.color = C4(col[0], col[1], col[2], 0.30 * inten * (1.0 if t == 0.0 else 0.8))
        else:
            for e, t, col in self.flares:
                e.color = C4(col[0], col[1], col[2], 0.0)
        self.glare.color = C4(1.0, 0.78, 0.45, 0.20 * max(0.0, zc) ** 18 + 0.10 * inten)

    # ------------------------------------------------------------------ events
    def on_impact(self, car, imp, x, y, z, nx, nz):
        if imp < 1.5:
            return
        n = min(14, int(imp * 1.2))
        for _ in range(n):
            self.particles.emit(SPARK, x, y, z, rnd.uniform(-5, 5) + nx * 3, rnd.uniform(1, 6), rnd.uniform(-5, 5) + nz * 3,
                                rnd.uniform(0.25, 0.6), 0.22, 0.05, (1.0, 0.78, 0.30), 1.0)
        if car.is_player:
            self.shake = min(1.0, self.shake + imp * 0.045)
            self.shake_dir = (-nx, -nz)
            self.chroma = min(1.0, self.chroma + imp * 0.09)

    def on_shift(self, car):
        if car.is_player and rnd.random() < 0.6:
            for lx in (-0.45, 0.45):
                fx, fz = car.wheel_world(lx, -2.25)
                self.particles.emit(FLAME, fx, car.y + 0.35, fz, -math.sin(car.h) * 3, 0.4, -math.cos(car.h) * 3,
                                    0.12, 0.45, 0.12, (1.0, 0.6, 0.2), 1.0)

    def collide_cars(self):
        cars = self.cars
        R = 2.1
        for a in range(len(cars)):
            A = cars[a]
            for b in range(a + 1, len(cars)):
                B = cars[b]
                dx, dz = A.x - B.x, A.z - B.z
                if dx * dx + dz * dz > 36.0:
                    continue
                for oa in (1.25, -1.25):
                    ax_ = A.x + math.sin(A.h) * oa
                    az_ = A.z + math.cos(A.h) * oa
                    for ob in (1.25, -1.25):
                        bx = B.x + math.sin(B.h) * ob
                        bz = B.z + math.cos(B.h) * ob
                        ddx, ddz = ax_ - bx, az_ - bz
                        d2 = ddx * ddx + ddz * ddz
                        if d2 >= R * R:
                            continue
                        d = math.sqrt(d2) or 1e-3
                        nx, nz = ddx / d, ddz / d
                        pen = R - d
                        A.x += nx * pen * 0.5
                        A.z += nz * pen * 0.5
                        B.x -= nx * pen * 0.5
                        B.z -= nz * pen * 0.5
                        vn = (A.vx - B.vx) * nx + (A.vz - B.vz) * nz
                        if vn < 0.0:
                            j = -1.35 * vn * 0.5
                            A.vx += nx * j
                            A.vz += nz * j
                            B.vx -= nx * j
                            B.vz -= nz * j
                            rax, raz = ax_ - A.x, az_ - A.z
                            rbx, rbz = bx - B.x, bz - B.z
                            A.w = clampf(A.w + 0.55 * (nx * raz - nz * rax) * j, -3.5, 3.5)
                            B.w = clampf(B.w + 0.55 * (-nx * rbz + nz * rbx) * j, -3.5, 3.5)
                            if -vn > 1.5:
                                self.on_impact(A if A.is_player else B, -vn * 0.6,
                                               (ax_ + bx) / 2, A.y + 0.6, (az_ + bz) / 2, nx, nz)

    # ------------------------------------------------------------------ race flow
    def start_race(self):
        self.state = 'countdown'
        self.count = 3.99
        self.race_time = 0.0
        self.finish_order = []
        self.msg_text = ''
        self.msg_t = 0.0
        self.res_bg.enabled = False
        self.res_txt.enabled = False
        for slot, c in enumerate(self.cars):
            c.reset_state()
        # player on pole, bots behind in order
        for slot, c in enumerate(self.cars):
            c.place_on_grid(slot)
        self.cam_h = self.player.h
        p = self.player
        self.cam_pos = Vec3(p.x - math.sin(p.h) * 8, p.y + 3, p.z - math.cos(p.h) * 8)
        self.cam_vel = Vec3(0, 0, 0)

    def lap_logic(self, c):
        lap = max(0, int(c.progress // self.track.N))
        if lap > c.laps_done:
            c.laps_done = lap
            lt = self.race_time - c.lap_start
            c.lap_start = self.race_time
            c.lap_times.append(lt)
            fastest = lt < c.best
            if fastest:
                c.best = lt
            if c.is_player:
                if lap >= TOTAL_LAPS:
                    pass
                else:
                    self.set_msg('LAP %d' % (lap + 1), 1.6)
                    self.msg_sub = 'LAP TIME ' + fmt_time(lt) + ('   FASTEST!' if fastest else '')
            if lap >= TOTAL_LAPS and not c.finished:
                c.finished = True
                c.finish_time = self.race_time
                self.finish_order.append(c)
                if c.is_player:
                    self.state = 'finished'
                    self.set_msg('FINISH!', 4.0)

    def compute_order(self):
        T = self.track
        fin = sorted([c for c in self.cars if c.finished], key=lambda c: self.finish_order.index(c))
        rest = sorted([c for c in self.cars if not c.finished],
                      key=lambda c: -(c.progress + c.along / T.spacing))
        self.order = fin + rest

    # ------------------------------------------------------------------ camera
    def update_camera(self, dt):
        p = self.player
        spd = p.speed
        cf = clampf(spd / 80.0, 0.0, 1.0)
        boost = 1.0 if p.boosting else 0.0
        hfov = 65.0 + 30.0 * (cf ** 1.2) + 6.0 * boost
        camera.fov = clampf(hfov, 60.0, 102.0)
        sh, ch = math.sin(p.h), math.cos(p.h)
        # shake (impact + chassis vibration)
        self.shake = max(0.0, self.shake - dt * 2.2)
        vib = 0.004 + (cf ** 2) * 0.020 + boost * 0.012
        sx = rnd.uniform(-1, 1) * (vib + self.shake * 0.35) + self.shake_dir[0] * self.shake * 0.25
        sy = rnd.uniform(-1, 1) * (vib * 0.8 + self.shake * 0.25)
        sz = rnd.uniform(-1, 1) * (vib + self.shake * 0.35) + self.shake_dir[1] * self.shake * 0.25
        if self.cam_mode == 1:
            camera.clip_plane_near = 0.1
            camera.position = Vec3(p.x + sh * 0.95 + sx * 0.5, p.y + 1.12 + sy * 0.5, p.z + ch * 0.95 + sz * 0.5)
            camera.rotation = Vec3(p.root.rotation_x + p.body.rotation_x * 0.8, math.degrees(p.h), p.body.rotation_z * 0.9)
            self.cam_pos = Vec3(camera.x, camera.y, camera.z)
            self.cam_h = p.h
            self.sun_follow()
            return
        camera.clip_plane_near = 0.3
        vel_h = math.atan2(p.vx, p.vz) if spd > 5.0 else p.h
        diff = wrap_pi(vel_h - p.h)
        target_h = p.h + diff * 0.55 * clampf(spd / 20.0, 0.0, 1.0)
        self.cam_h += wrap_pi(target_h - self.cam_h) * min(1.0, dt * 4.5)
        back = 7.0 + clampf(p.ax_s, -14.0, 10.0) * 0.12 + boost * 1.8 + spd * 0.015
        height = 2.35 + spd * 0.008 + max(0.0, -p.ax_s - 3.0) * 0.02
        dx, dz = math.sin(self.cam_h), math.cos(self.cam_h)
        desired = Vec3(p.x - dx * back, p.y + height, p.z - dz * back)
        carv = Vec3(p.vx, 0, p.vz)
        k, c = 40.0, 10.0
        n = 2
        sdt = dt / n
        for _ in range(n):
            acc = (desired - self.cam_pos) * k + (carv - self.cam_vel) * c
            self.cam_vel += acc * sdt
            self.cam_pos += self.cam_vel * sdt
        camera.position = Vec3(self.cam_pos.x + sx, self.cam_pos.y + sy, self.cam_pos.z + sz)
        look = Vec3(p.x + dx * 10.0, p.y + 1.0, p.z + dz * 10.0)
        camera.look_at(look)
        tgt_roll = clampf(-p.ay_s * 0.18 - math.degrees(diff) * 0.05, -6.0, 6.0)
        self.cam_roll += (tgt_roll - self.cam_roll) * min(1.0, dt * 4.0)
        camera.rotation_z = self.cam_roll
        self.sun_follow()

    def sun_follow(self):
        cp = camera.world_position
        sp = Vec3(cp.x + SUN_DIR[0] * 2200.0, cp.y + SUN_DIR[1] * 2200.0, cp.z + SUN_DIR[2] * 2200.0)
        self.sun_disc.position = sp
        self.sun_glow.position = sp
        self.sky.position = cp
        if self.shader is not None:
            v = Vec3(cp.x, cp.y, cp.z)
            for e in self.sheen:
                e.set_shader_input('cam_pos', v)

    # ------------------------------------------------------------------ HUD update
    def update_hud(self, dt):
        p = self.player
        T = self.track
        self.order_timer -= dt
        if self.order_timer <= 0.0:
            self.order_timer = 0.2
            self.compute_order()
        pos = self.order.index(p) + 1 if p in self.order else 1
        self.set_txt('pos', self.t_pos, 'POS %d/%d' % (pos, len(self.cars)))
        lap = min(p.laps_done + 1, TOTAL_LAPS)
        self.set_txt('lap', self.t_lap, 'LAP %d/%d' % (lap, TOTAL_LAPS))
        if self.state == 'countdown':
            t = 0.0
        else:
            t = (p.finish_time if p.finished else self.race_time) - p.lap_start if not p.finished else p.finish_time
        if int(self.clock * 30) % 2 == 0:
            self.set_txt('time', self.t_time, 'TIME ' + fmt_time(max(0.0, t)) if not p.finished else 'TOTAL ' + fmt_time(p.finish_time))
        self.set_txt('best', self.t_best, 'BEST ' + fmt_time(p.best))
        self.set_txt('last', self.t_last, 'LAST ' + fmt_time(p.lap_times[-1] if p.lap_times else None))
        kmh = int(p.speed * 3.6)
        if int(self.clock * 20) % 2 == 0:
            self.set_txt('spd', self.t_speed, str(kmh))
        gtxt = 'R' if (p.vlong < -0.5 and p.in_rev > 0) else str(p.gear)
        self.set_txt('gear', self.t_gear, gtxt)
        # tachometer
        frac = clampf(p.rpm / MAXRPM, 0.0, 1.0)
        lit = int(frac * 30.0 + 0.5)
        flash = p.rpm > 7100.0 and int(self.clock * 14) % 2 == 0
        key = (lit, flash, p.boosting)
        if key != self.seg_key:
            self.seg_key = key
            for i, e in enumerate(self.segs):
                f = i / 29.0
                if i < lit:
                    if f < 0.6:
                        col = C4(0.25, 0.95, 0.45, 1)
                    elif f < 0.8:
                        col = C4(1.0, 0.85, 0.2, 1)
                    else:
                        col = C4(1.0, 0.18, 0.15, 1) if not flash else C4(1, 1, 1, 1)
                else:
                    col = C4(0.15, 0.15, 0.18, 0.8)
                e.color = col
        self.set_txt('shift', self.t_shift, 'SHIFT' if (p.rpm > 7100.0 and flash and p.gear < 6) else '')
        self.nitro_bar.scale_x = max(0.001, 0.42 * p.nitro)
        self.nitro_bar.color = C4(1.0, 0.7, 0.2, 1) if p.boosting else C4(0.2, 0.6, 1.0, 1)
        # minimap dots
        for c, d in zip(self.cars, self.dots):
            d.position = self.mm_xy(c.x, c.z)
        # centre messages
        if self.state == 'countdown':
            n = int(math.ceil(self.count))
            self.set_txt('msg', self.t_msg, str(max(1, min(3, n))))
            self.set_txt('sub', self.t_sub, 'GET READY')
        else:
            if self.msg_t > 0.0:
                self.msg_t -= dt
                self.set_txt('msg', self.t_msg, self.msg_text)
                self.set_txt('sub', self.t_sub, getattr(self, 'msg_sub', '') if self.msg_text.startswith('LAP') or self.msg_text == 'FINISH!' else '')
            else:
                self.set_txt('msg', self.t_msg, '')
                self.set_txt('sub', self.t_sub, '')
        if self.state == 'finished' and not self.res_bg.enabled:
            self.res_bg.enabled = True
            self.res_txt.enabled = True
        if self.res_bg.enabled:
            lines = ['RACE RESULTS', '']
            for k, c in enumerate(self.order):
                tm = fmt_time(c.finish_time) if c.finished else 'racing...'
                lines.append('%d.  %-7s  %s' % (k + 1, c.name, tm))
            lines += ['', 'ENTER = restart race']
            self.set_txt('res', self.res_txt, '\n'.join(lines))
        if self.t_hint.enabled and self.clock > 12.0:
            self.t_hint.enabled = False

    # ------------------------------------------------------------------ main update
    def update(self):
        dt = min(time.dt, 0.05)
        if dt <= 0.0:
            return
        self.clock += dt
        if self.state == 'countdown':
            self.count -= dt
            if self.count <= 0.0:
                self.state = 'racing'
                self.race_time = 0.0
                self.msg_text = 'GO!'
                self.msg_t = 1.0
                for c in self.cars:
                    c.lap_start = 0.0
        else:
            self.race_time += dt
        for c in self.cars:
            if self.state == 'countdown':
                if c.is_player:
                    c.player_controls(dt, revonly=True)
                else:
                    c.in_thr = 0.0
                c.countdown_revs(dt)
                continue
            if c.is_player and not c.finished:
                c.player_controls(dt)
            else:
                c.ai_control(dt)
            c.apply_steer(dt)
            c.nitro_logic(dt)
            n = max(1, int(math.ceil(dt * 120.0)))
            sub = dt / n
            for _ in range(n):
                c.physics(sub)
            c.update_track()
            c.wall_collision()
        self.collide_cars()
        for c in self.cars:
            if self.state != 'countdown':
                self.lap_logic(c)
            c.sync_visual(dt)
            c.effects(dt)
        self.particles.update(dt)
        self.skids.update(dt)
        self.update_camera(dt)
        self.update_fx(dt)
        self.update_hud(dt)

    def input(self, key):
        if key == 'c':
            self.cam_mode = 1 - self.cam_mode
            self.player.body.enabled = True
        elif key == 'r':
            self.player.reset_to_track()
            self.cam_pos = Vec3(self.player.x - math.sin(self.player.h) * 8, self.player.y + 3,
                                self.player.z - math.cos(self.player.h) * 8)
            self.cam_vel = Vec3(0, 0, 0)
            self.cam_h = self.player.h
        elif key == 'enter':
            self.start_race()
        elif key == 'escape':
            application.quit()


# =====================================================================================
#  ENTRY POINT
# =====================================================================================
game = None


def update():
    if game is not None:
        game.update()


def input(key):
    if game is not None:
        game.input(key)


if __name__ == '__main__':
    try:
        app = Ursina(title=TITLE, borderless=False, vsync=True, development_mode=False)
    except TypeError:
        app = Ursina()
    try:
        window.title = TITLE
        window.borderless = False
        window.exit_button.visible = False
        window.fps_counter.enabled = True
        window.vsync = True
    except Exception:
        pass
    game = Game()
    app.run()