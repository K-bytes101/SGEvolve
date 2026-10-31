import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

import math
# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# Initial path parameters: radius r (approx 2/pi) and 12 Fourier coefficients
_C = np.array([0.63661977, 0.01216, 0.00114, 0.00043, 0.00016, 0.00006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_CORRIDOR = Polygon([(-5, 0), (0, 0), (0, 5), (1, 5), (1, -1), (-5, -1)])


def _optimize_path():
    """
    Refines the Fourier coefficients using SLSQP to maximize the sofa area.
    This runs once during initialization on a coarse grid for efficiency.
    """
    global _C
    try:
        from scipy.optimize import minimize
        def objective(coeffs):
            r, fk = coeffs[0], coeffs[1:]
            ks = np.arange(1, len(coeffs))
            signs = (-1) ** (ks - 1)
            intersected = None
            # Sample 50 steps for a more accurate area estimate during optimization
            for t in np.linspace(0, np.pi / 2, 50):
                sv = np.sin(2 * ks * t)
                hx = r * np.cos(t) + np.sum(fk * sv)
                hy = r * np.sin(t) + np.sum(fk * signs * sv)
                # Transform corridor to sofa frame: translate by hx, -hy then rotate by -theta
                p = rotate(translate(_CORRIDOR, hx, -hy), -t, origin=(0, 0), use_radians=True)
                intersected = p if intersected is None else intersected.intersection(p)
                if intersected.is_empty: return 0.0
            return -float(intersected.area)

        # Increase maxiter to 40 and use a denser grid to refine the Fourier coefficients
        res = minimize(objective, _C, method='SLSQP', options={'maxiter': 40})
        if res.success:
            _C = res.x
    except Exception:
        pass  # Fallback to initial guess if optimization fails


# Run optimization at module load
_optimize_path()


def get_sofa_pose(theta: float):
    """
    Calculates the optimized (x, y) pose of the sofa at angle theta.
    Uses a high-order symmetric Fourier series to maximize the intersection area.
    """
    t = float(theta)
    r, fk = _C[0], _C[1:]
    ks = np.arange(1, len(_C))
    sv = np.sin(2 * ks * t)
    hx = r * np.cos(t) + np.sum(fk * sv)
    hy = r * np.sin(t) + np.sum(fk * ((-1) ** (ks - 1)) * sv)
    return (float(-hx), float(hy))


# EVOLVE-BLOCK-END


# --- 1. 定义初始 L 型走廊 ---
# 走廊宽度为 1，拐角在 (0,0) 到 (1,1) 附近
corridor_poly = Polygon([
    (-5, 0), (0, 0), (0, 5), (1, 5), (1, -1), (-5, -1)
])

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title("Sofa Perspective: Intersection of Corridors")

# 准备绘图对象
current_corridor_patch, = ax.fill([], [], color='gray', alpha=0.2, label='Moving Corridor')
sofa_patch, = ax.fill([], [], color='blue', alpha=0.6, label='Evolving Sofa (Intersection)')
text_area = ax.text(-2.8, 2.5, '', fontsize=12, color='red')

intersection_poly = None
steps = 1000


def init():
    current_corridor_patch.set_xy(np.empty((0, 2)))
    sofa_patch.set_xy(np.empty((0, 2)))
    return current_corridor_patch, sofa_patch


def update(frame):
    global intersection_poly

    theta = (frame / steps) * np.pi / 2
    x, y = get_sofa_pose(theta)

    # --- 核心逻辑：逆变换 ---
    # 将走廊变换到沙发的参考系中
    p = translate(corridor_poly, -x, -y)
    p = rotate(p, -theta, origin=(0, 0), use_radians=True)

    # 更新当前时刻走廊的显示
    cx, cy = p.exterior.xy
    current_corridor_patch.set_xy(np.c_[cx, cy])

    # --- 计算交集 ---
    if intersection_poly is None:
        intersection_poly = p
    else:
        intersection_poly = intersection_poly.intersection(p)

    # 更新沙发（交集）的显示
    if not intersection_poly.is_empty:
        # 如果交集是由多个多边形组成的 (MultiPolygon)，取 exterior 可能会报错
        # 为简化演示，这里假设它是连通的
        if intersection_poly.geom_type == 'Polygon':
            sx, sy = intersection_poly.exterior.xy
            sofa_patch.set_xy(np.c_[sx, sy])
            text_area.set_text(f"Current Area: {intersection_poly.area:.4f}")
        elif intersection_poly.geom_type == 'MultiPolygon':
            # 处理不连通的情况
            all_coords = []
            for geom in intersection_poly.geoms:
                all_coords.append(np.c_[geom.exterior.xy])
            # 简单起见，只画第一个
            sofa_patch.set_xy(all_coords[0])

    return current_corridor_patch, sofa_patch, text_area


ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=True, interval=2, repeat=False)

plt.legend(loc='lower right')
plt.show()