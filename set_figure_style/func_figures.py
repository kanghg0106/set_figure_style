from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["savefig.facecolor"] = "white"

TITLE_FS = 17
AXIS_FS = 15
TICK_FS = 13

ttfs, axfs, tkfs = TITLE_FS, AXIS_FS, TICK_FS


def _apply_tick_params(
    ax,
    axis_name: str,          # "x" or "y"
    *,
    major_fontsize: float,
    minor_fontsize: float,
    major_length: float,
    minor_length: float,
    major_width: float,
    minor_width: float,
    rotation: float = 0.0,
    is_minortick: bool = True,
    color=None,
    **extra,                 # top=True 같은 걸 넘길 때 사용
):
    """x 또는 y 에 대해 major/minor tick 스타일을 한 번에 적용."""
    # major
    major_kwargs = dict(
        axis=axis_name,
        which="major",
        direction="in",
        labelsize=major_fontsize,
        length=major_length,
        width=major_width,
        labelrotation=rotation,
    )
    if color is not None:
        major_kwargs["colors"] = color
    major_kwargs.update(extra)
    ax.tick_params(**major_kwargs)

    # minor
    if is_minortick:
        minor_kwargs = dict(
            axis=axis_name,
            which="minor",
            direction="in",
            labelsize=minor_fontsize,
            length=minor_length,
            width=minor_width,
            labelrotation=rotation,
        )
        if color is not None:
            minor_kwargs["colors"] = color
        minor_kwargs.update(extra)
        ax.tick_params(**minor_kwargs)


def set_frame_style(
    ax,
    title='',
    double_axis=False,
    xtick_fontsize=tkfs,
    ytick_fontsize=tkfs,
    title_fontsize=ttfs,
    xlabel='X',
    xlabel_fontsize=axfs,
    ylabel='Y',
    ylabel_fontsize=axfs,
    x_labelpad=10,
    y_labelpad=10,
    # 여기부터는 thick 스타일을 기본값으로 설정
    linewidth=1.5,
    is_minortick=True,
    tick_linewidth=1.5,
    minor_tick_linewidth=1.0,
    tick_length=5.0,
    minor_tick_length=3.0,
    xtick_rotation=0,
    ytick_rotation=0,
    **kwargs,
):
    """
    축 스타일 통일 함수.
    - style 파라미터는 제거했고, thick 스타일이 기본값.
    - double_axis=False  : 위/오른쪽까지 tick mirror (labels는 아래/왼쪽만)
    - double_axis=True   : twinx() 같은 보조축용 (mirror 안 함)
    """

    # minor locator 설정
    if is_minortick:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # === tick 설정 (중복 제거) ===
    for ax_name, fs, rot in [
        ('x', xtick_fontsize, xtick_rotation),
        ('y', ytick_fontsize, ytick_rotation),
    ]:
        _apply_tick_params(
            ax,
            ax_name,
            major_fontsize=fs,
            minor_fontsize=fs,
            major_length=tick_length,
            minor_length=minor_tick_length,
            major_width=tick_linewidth,
            minor_width=minor_tick_linewidth,
            rotation=rot,
            is_minortick=is_minortick,
        )

    # double_axis=False 이면 ticks 를 양쪽에 mirror
    if not double_axis:
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        # 라벨은 기본값(아래/왼쪽)만 살아있음 (기존 labeltop=False, labelright=False 와 시각적으로 동일)

    # 제목 / 라벨
    ax.set_title(f'{title}', fontsize=title_fontsize, y=1.05)
    ax.set_xlabel(f'{xlabel}', fontsize=xlabel_fontsize, labelpad=x_labelpad)
    ax.set_ylabel(f'{ylabel}', fontsize=ylabel_fontsize, labelpad=y_labelpad)

    # 스파인 두께
    for ax_i in ['top', 'bottom', 'left', 'right']:
        ax.spines[ax_i].set_linewidth(linewidth)

# ----------------------------------------------------------------------
# 2. 텍스트 박스 / 범례
# ----------------------------------------------------------------------


def set_textbox(
    ax,
    text: str,
    position: tuple[float, float],
    *,
    fontsize: float = 10,
    edge: bool = False,
    facecolor: str = "k",
    fontcolor: str = "k",
    edgecolor: str = "k",
    alpha: float = 0.0,
    rotation: float = 0.0,
    edgelinewidth: float = 0.8,
    zorder: int = 999,
    fontweight: str = "light",
):
    """
    ax 좌표계(0~1, 0~1)를 기준으로 위치 지정하는 텍스트 박스.
    """

    if not edge:
        props = dict(
            boxstyle="square",
            facecolor=facecolor,
            alpha=alpha,
            linewidth=0,
            edgecolor=edgecolor,
        )
    else:
        props = dict(
            boxstyle="square",
            facecolor=facecolor,
            alpha=alpha,
            linewidth=edgelinewidth,
            edgecolor=edgecolor,
        )

    ax.text(
        position[0],
        position[1],
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=fontcolor,
        rotation=rotation,
        zorder=zorder,
        fontweight=fontweight,
        va="center",
        ha="center",
        bbox=props,
    )


def set_legend_style(
    ax,
    *,
    ncols: int = 1,
    frame: bool = False,
    legend_title: str | None = "",
    position="best",
    legend_fontsize: float = TICK_FS - 1.5,
    legend_title_fontsize: float = TICK_FS - 1.5,
    frame_linewidth: float = 1.5,
    handles=None,
    labelspacing: float = 0.2,
    **kwargs,
):
    """
    범례 스타일 통일.

    position : str 또는 (x, y) tuple
        문자열이면 loc=position, tuple이면 bbox_to_anchor=position 로 동작.
    """

    position = kwargs.get("pos", position)
    legend_title = kwargs.get("lt", legend_title)
    legend_fontsize = kwargs.get("lfs", legend_fontsize)
    legend_title_fontsize = kwargs.get("ltfs", legend_title_fontsize)

    is_tuple_pos = isinstance(position, tuple)

    legend_kwargs = dict(
        fontsize=legend_fontsize,
        ncol=ncols,
        frameon=frame,
        framealpha=1,
        borderpad=0.02,
        labelspacing=labelspacing,
        handles=handles,
        handlelength=1.75,
        columnspacing=1,
    )

    if legend_title not in (None, ""):
        legend_kwargs["title"] = legend_title
        legend_kwargs["title_fontsize"] = legend_title_fontsize

    if is_tuple_pos:
        legend = ax.legend(bbox_to_anchor=position, **legend_kwargs)
    else:
        legend = ax.legend(loc=position, **legend_kwargs)

    legend_frame = legend.get_frame()
    legend_frame.set_linewidth(frame_linewidth)
    legend_frame.set_edgecolor("black")
    legend_frame.set_boxstyle("square")


# ----------------------------------------------------------------------
# 3. 간단 유틸리티
# ----------------------------------------------------------------------

def set_box(ax, x_dim,y_dim, *,
            coords: str = "data", edgecolor=None, facecolor=None, alpha: float = 1.0, hatch=None, lw: float = 1.0, fill: bool = True, ls: str = "-",zorder: int = 99,):
    """
    데이터 영역 또는 축 좌표계(0~1, 0~1) 기준으로 직사각형 패치를 추가.

    x_dim : (x0, x1)
    y_dim : (y0, y1)

    coords : {'data', 'ax'}
        'data'  : 데이터 좌표 (기존 동작과 동일, 기본값)
        'ax'  : 축 좌표 (0~1, 0~1) 기준
    """

    if coords == "ax":
        transform = ax.transAxes
    else:  # 'data'
        transform = ax.transData

    rect = patches.Rectangle(
        (x_dim[0], y_dim[0]),
        x_dim[1] - x_dim[0],
        y_dim[1] - y_dim[0],
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        fill=fill,
        hatch=hatch,
        zorder=zorder,
        lw=lw,
        ls=ls,
        transform=transform,
    )

    ax.add_patch(rect)
    return rect

def set_axis_color(axis,
    auxilary_axis,
    color,
    auxilary_color,
    *,
    axis_position="left",
    auxilary_position="right",
):
    """
    twinx() 등으로 만든 보조축과 기본축의 '색'만 맞출 때 사용.
    - 두께, tick 길이/위치는 건드리지 않는다.
    - set_frame_style_new 을 먼저 호출해 둔 상태에서 쓰는 것을 가정.
    """

    # --- spine 색만 변경 (두께는 그대로) ---
    axis.spines[axis_position].set_color(color)

    # 보조축 쪽 spine 색
    auxilary_axis.spines[axis_position].set_color(color)
    auxilary_axis.spines[auxilary_position].set_color(auxilary_color)

    # --- y축 라벨 색 ---
    axis.yaxis.label.set_color(color)
    auxilary_axis.yaxis.label.set_color(auxilary_color)

    # --- y축 tick 색 (길이/두께는 set_frame_style_new 값 유지) ---
    axis.tick_params(axis="y", colors=color)
    auxilary_axis.tick_params(axis="y", colors=auxilary_color)


def help_Gridspec():
    """간단한 GridSpec 사용 예시 출력."""
    print(
        "fig = plt.figure(dpi=400)\n"
        "gs = GridSpec(3, 3, figure=fig, hspace=0.5)\n"
        "ax0 = fig.add_subplot(gs[:, 2:])\n"
        "ax1 = fig.add_subplot(gs[:, :2])\n"
        "ax = [ax0, ax1]"
    )


# ----------------------------------------------------------------------
# 4. Curly Brace
# ----------------------------------------------------------------------
def set_CurlyBrace(ax, ll_corner, width: float, height: float, *, direction: str = "h", color: str = "k", lw: float = 1.0, ls: str = "-", coords: str = "ax", zorder: int = 99): 
    """
    중괄호 모양 patch 를 축에 추가.

    Parameters
    ----------
    ax : matplotlib.ax.Axes
    ll_corner : (x, y)
        왼쪽-아래 기준 좌표
        - coords='data'  : 데이터 좌표
        - coords='ax'  : 축 좌표 (0~1, 0~1)
    width, height : float
        중괄호 전체 크기
        - coords='data'  : 데이터 단위
        - coords='ax'  : 축 좌표 단위
    direction : {'h', 'v'}
        'h' : { } 가 가로로 눕는 형태
        'v' : { } 가 세로 형태
    coords : {'data', 'ax'}
        위치/크기를 어떤 좌표계 기준으로 줄지 선택.
    """

    Path = mpath.Path

    if direction == "h":
        verts = np.array([(0, 0), (0.5, 0), (0.5, 0.2), (0.5, 0.3), (0.5, 0.5), (1, 0.5), (0.5, 0.5), (0.5, 0.7), (0.5, 0.8), (0.5, 1), (0, 1),])
    else:  # 'v'
        verts = np.array([(0, 0),(0, 0.5),(0.2, 0.5),(0.3, 0.5),(0.5, 0.5),(0.5, 1),(0.5, 0.5),(0.7, 0.5),(0.8, 0.5),(1, 0.5),(1, 0)])

    # width / height 로 스케일링하고 왼쪽-아래 위치로 이동
    verts[:, 0] *= width
    verts[:, 1] *= height
    verts[:, 0] += ll_corner[0]
    verts[:, 1] += ll_corner[1]

    codes = [Path.MOVETO,Path.CURVE3,Path.CURVE3,Path.LINETO,Path.CURVE3,Path.CURVE3,Path.CURVE3,Path.CURVE3,Path.LINETO,Path.CURVE3,Path.CURVE3]

    # 좌표계 선택
    if coords == "ax":
        transform = ax.transAxes
    else:  # "data"
        transform = ax.transData

    cb_patch = mpatches.PathPatch(
        Path(verts, codes),
        fc="none",
        clip_on=False,
        transform=transform,
        color=color,
        lw=lw,
        ls=ls,
        zorder=zorder
    )

    ax.add_patch(cb_patch)


def set_arrow(
    ax,
    x,
    y,
    x0=None,
    x1=None,
    *,
    x_gap: float = 0.0,
    y_gap: float = 0.0,
    fit_order: int | None = None,
    arrow_position: str = "right",   # 'right' 또는 'left'
    color: str = "k",
    alpha: float = 1,
    arrow_head_size: float = 50.0,         # scatter s 값 (조금 키워둠, 필요시 조절)
    arrow_tail_linewidth: float = 2.0,
    linestyle: str = "-",
    zorder: int = 1000,
    deg_correction: float = 0.0,
):
    """
    주어진 데이터 (x, y) 위에 곡선을 그리고, 한쪽 끝에 화살표 머리를 붙인다.

    x0, x1 : 화살표를 그릴 x 구간 (None이면 전체 범위)
    fit_order : None 또는 0 이면 원 데이터 그대로, 1 이상이면 해당 차수로 polyfit
    arrow_position : {'right', 'left'}
    """

    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("set_arrow: x와 y의 shape가 같아야 합니다.")

    # ---------------- 1. 사용할 x 구간 선택 ----------------
    if x0 is None and x1 is None:
        x0_sel = x.min()
        x1_sel = x.max()
    else:
        x0_sel = x.min() if x0 is None else x0
        x1_sel = x.max() if x1 is None else x1
        if x0_sel > x1_sel:
            x0_sel, x1_sel = x1_sel, x0_sel

    mask = (x >= x0_sel) & (x <= x1_sel)
    if not np.any(mask):
        # 범위 안에 데이터가 없으면 중앙에 가까운 점 주변으로 대충 구성
        idx = np.argmin(np.abs(x - 0.5 * (x0_sel + x1_sel)))
        if idx == 0:
            idxs = [0, 1] if len(x) > 1 else [0]
        elif idx == len(x) - 1:
            idxs = [len(x) - 2, len(x) - 1]
        else:
            idxs = [idx - 1, idx, idx + 1]
        x_seg = x[idxs]
        y_seg = y[idxs]
    else:
        x_seg = x[mask]
        y_seg = y[mask]

    if x_seg.size < 2:
        return

    # 정렬
    sort_idx = np.argsort(x_seg)
    x_seg = x_seg[sort_idx]
    y_seg = y_seg[sort_idx]

    # ---------------- 2. 스무딩 (옵션) ----------------
    if fit_order is not None and fit_order > 0 and x_seg.size > fit_order:
        coef = np.polyfit(x_seg, y_seg, fit_order)
        x_fine = np.linspace(x_seg[0], x_seg[-1], max(50, (len(x_seg) - 1) * 20))
        y_fine = np.polyval(coef, x_fine)
    else:
        x_fine = x_seg
        y_fine = y_seg

    # ---------------- 3. 평행 이동 ----------------
    x_fine = x_fine + x_gap
    y_fine = y_fine + y_gap

    # ---------------- 4. 곡선 그리기 ----------------
    ax.plot(
        x_fine,
        y_fine,
        lw=arrow_tail_linewidth,
        ls=linestyle,
        zorder=zorder,
        color=color,
        alpha=alpha,
    )

    if x_fine.size < 2:
        return

    # ---------------- 5. 화살표 머리 (예전 각도 로직을 살짝 정리해서 사용) ----------------
    if arrow_position == "right":
        # 예전 코드처럼 '왼쪽 끝'의 기울기를 사용해서 각도 계산
        dx = x_fine[0] - x_fine[1]
        dy = y_fine[0] - y_fine[1]
        head_x, head_y = x_fine[-1], y_fine[-1]
        base_angle = np.degrees(np.arctan(dx / dy)) if dy != 0 else 0.0
        marker_angle = base_angle + deg_correction + 60.0
    elif arrow_position == "left":
        dx = x_fine[1] - x_fine[0]
        dy = y_fine[1] - y_fine[0]
        head_x, head_y = x_fine[0], y_fine[0]
        base_angle = np.degrees(np.arctan(dx / dy)) if dy != 0 else 0.0
        marker_angle = base_angle + deg_correction
    else:
        raise ValueError("set_arrow: arrow_position 은 'right' 또는 'left' 이어야 합니다.")

    ax.scatter(
        head_x,
        head_y,
        marker=(3, 0, marker_angle),
        s=arrow_head_size,
        zorder=zorder,
        color=color,
        alpha=alpha,
    )

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

# 색 보간 유틸
def _blend(c1, c2, t):
    c1 = np.array(c1); c2 = np.array(c2)
    return (1-t)*c1 + t*c2


def make_radial_gradient(
    n=128,
    base_color="red",
    highlight_color="white",
    center_brighten=0.4,
    edge_darken=0.1,
    highlight_strength=1.0,
    highlight_pos=(-0.35, -0.25),
    highlight_sigma=0.3,
    *,
    rim_width=0.15,
    rim_strength=1,
):
    y, x = np.ogrid[-1:1:n*1j, -1:1:n*1j]
    r_raw = np.sqrt(x * x + y * y)     # 원 바깥은 1보다 큼
    r = np.clip(r_raw, 0, 1)           # 색 보간용은 0~1로 제한

    base = np.array(to_rgba(base_color))
    white = np.array([1, 1, 1, 1])
    black = np.array([0, 0, 0, 1])

    inner = _blend(base, white, center_brighten)
    outer = _blend(base, black, edge_darken)

    img = inner * (1 - r)[..., None] + outer * r[..., None]
    img[..., 3] = 1.0

    # 하이라이트
    hx, hy = highlight_pos
    s = highlight_sigma
    spot = np.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * s * s))
    hi = np.array(to_rgba(highlight_color))
    w = (highlight_strength * spot)[..., None]
    img[..., :3] = img[..., :3] * (1 - w) + hi[:3] * w

    # 테두리 rim (r 기준, 0~1)
    if rim_width > 0 and rim_strength > 0:
        rim_start = 1.0 - rim_width
        t = np.clip((r - rim_start) / (1.0 - rim_start), 0, 1)
        rim_mask = (t > 0.05).astype(float)[..., None]
        img[..., :3] = img[..., :3] * (1 - rim_strength * rim_mask) \
                       + black[:3] * (rim_strength * rim_mask)

    # --- 알파: r_raw 기준으로 원 바깥은 0, 원 안은 1→0 ---
    alpha = 1.0 -  np.clip(r_raw, 0, 1) ** 1000
    alpha[r_raw > 1] = 0.0         # 원 바깥은 완전히 투명
    alpha = np.clip(alpha, 0, 1)

    img[..., 3] = alpha
    return img

def gradient_marker(ax, xy, img=None, marker_size=10, zorder=3):
    """
    픽셀(포인트) 단위 radius의 그라디언트 마커 + 깔끔한 테두리.
    - 축이 log여도 항상 원형/일정 크기
    """
    if img is None:
        img = make_radial_gradient()

    x, y = xy
        

    # 1) 채움(그라디언트)
    ab = AnnotationBbox(
        OffsetImage(img, zoom=marker_size/72, resample=True),
        (x, y), xycoords='data', frameon=False, pad=0, zorder=zorder
    )
    ax.add_artist(ab)
    # 축 경계로 클리핑
    ab.set_clip_on(True)
    ab.set_clip_path(ax.patch)   # 또는: ab.set_clip_box(ax.bbox)

def draw_gradient_marker(ax, x, y, *, base_color='red', highlight_color='w',
                        center_brighten=0.4, edge_darken=0.1,
                        highlight_strength=1.0, highlight_sigma=0.3, 
                        rim_width=.15, rim_strength=1,
                        marker_size=10, zorder=9999):
    """
    gradient marker를 여러 개 찍는 high-level API.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x와 y는 같은 길이여야 합니다.")

    img = make_radial_gradient(base_color=base_color, highlight_color=highlight_color,
                               center_brighten=center_brighten, edge_darken=edge_darken,
                               highlight_strength=highlight_strength, highlight_sigma=highlight_sigma, 
                               rim_width=rim_width, rim_strength=rim_strength)

    # 점 찍기
    for xi, yi in zip(x, y):
        gradient_marker(
            ax=ax,
            xy=(xi, yi),
            img=img,
            marker_size=marker_size,
            zorder=zorder,
        )


    return None


from typing import Iterable

import matplotlib.cm as mcm
from matplotlib.colors import to_hex, Colormap

# cmcrameri는 선택적(optional) 의존성으로 처리
try:
    from cmcrameri import cm as _cmcrameri
except ImportError:  # 패키지가 없으면 None으로 두고 넘어감
    _cmcrameri = None


def get_color_array(
    cmap: str | Colormap,
    data_length: int,
    *,
    c_start: float = 0.075,
    c_end: float = 1.0,
) -> np.ndarray:
    """
    주어진 colormap에서 `data_length`만큼 색을 샘플링하여 RGBA 배열 반환.

    Parameters
    ----------
    cmap : str 또는 Colormap
        - str 이면:
          1) cmcrameri.cm 안에서 먼저 찾고
          2) 없으면 matplotlib.cm.get_cmap 에서 찾는다.
        - Colormap 인스턴스면 그대로 사용.
    data_length : int
        샘플링할 색 개수.
    c_start, c_end : float
        colormap 상에서 사용할 구간 (0~1).

    Returns
    -------
    colors : (N, 4) ndarray
        RGBA 배열 (0~1 범위).
    """
    if isinstance(cmap, Colormap):
        cmap_obj = cmap
    elif isinstance(cmap, str):
        cmap_obj = None

        # 1) cmcrameri 우선
        if _cmcrameri is not None and hasattr(_cmcrameri, cmap):
            cmap_obj = getattr(_cmcrameri, cmap)
        else:
            # 2) matplotlib 기본 colormap
            cmap_obj = mcm.get_cmap(cmap)

    else:
        raise TypeError("cmap 은 str 또는 matplotlib.colors.Colormap 이어야 합니다.")

    values = np.linspace(c_start, c_end, data_length)
    return cmap_obj(values)


def rgb2hex(color: Iterable[float]) -> str:
    """
    (r, g, b) 또는 (r, g, b, a) 형태의 색을 16진수 문자열로 변환.
    입력 값은 0~1 범위 혹은 0~255 범위 모두 허용.

    Examples
    --------
    >>> rgb2hex((1.0, 0.5, 0.0))
    '#ff8000'
    """
    arr = np.array(list(color), dtype=float).ravel()
    if arr.size < 3:
        raise ValueError("rgb2hex: 최소 3개 값(r,g,b)이 필요합니다.")

    # 0~255 범위로 들어왔으면 0~1 로 정규화
    if arr.max() > 1.0:
        arr = arr / 255.0

    # 앞의 3개(r,g,b)만 사용, to_hex가 '#rrggbb' 반환
    return to_hex(arr[:3], keep_alpha=False)


# ----------------------------------------------------------------------
# 고정 팔레트 색상 (논문/프레젠용)
# ----------------------------------------------------------------------
c_red = "#FF2C43"
c_orange = "#FF9A47"
c_yellow = "#F8E448"
c_lime = "#98EE84"
c_green = "#1B9C21"
c_mint = "#76DDBE"
c_blue = "#49BCE6"
c_navy = "#4965E6"
c_purple = "#A588DB"
c_pink = "#FF7BCC"
