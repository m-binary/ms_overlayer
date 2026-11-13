"""Screenshot Overlay: legt ein transparentes PNG verlustfrei über einen Screenshot.

Neue Overlays: einfach ein PNG in den Ordner `overlays/` legen.
Der Dateiname wird zum Anzeigenamen (z. B. `fast_track.png` → „Fast Track“).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageCms, ImageOps, PngImagePlugin

OVERLAY_DIR = Path(__file__).parent / "overlays"

# Weicht das Seitenverhältnis des Overlays höchstens so stark ab, wird es minimal
# gestreckt (nichts geht verloren). Darüber wird es proportional hochskaliert,
# bis es den Screenshot komplett füllt, und mittig beschnitten.
MAX_STRETCH = 0.03

SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


# ---------------------------------------------------------------- Bildverarbeitung

def list_overlays() -> dict[str, Path]:
    return {p.stem.replace("_", " ").title(): p for p in sorted(OVERLAY_DIR.glob("*.png"))}


def _rgb_icc(img: Image.Image) -> bytes | None:
    """ICC-Profil des Bildes, sofern es ein gültiges RGB-Profil ist."""
    icc = img.info.get("icc_profile")
    if not icc:
        return None
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    except (OSError, ImageCms.PyCMSError):
        return None
    return icc if profile.profile.xcolor_space.strip() == "RGB" else None


def _profile(icc: bytes | None) -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(io.BytesIO(icc)) if icc else SRGB


def _profile_name(icc: bytes | None) -> str:
    if not icc:
        return "kein Farbprofil (sRGB)"
    return ImageCms.getProfileDescription(_profile(icc)).strip() or "eingebettetes Profil"


def _fill_box(src: tuple[int, int], dst: tuple[int, int]) -> tuple[float, float, float, float]:
    """Ausschnitt des Overlays (Subpixel-genau), der auf die Screenshot-Größe skaliert wird."""
    w, h = src
    src_ratio, dst_ratio = w / h, dst[0] / dst[1]
    if abs(src_ratio / dst_ratio - 1) <= MAX_STRETCH:
        return (0.0, 0.0, float(w), float(h))
    if src_ratio > dst_ratio:  # Overlay breiter: links und rechts beschneiden
        bw = h * dst_ratio
        return ((w - bw) / 2, 0.0, (w + bw) / 2, float(h))
    bh = w / dst_ratio  # Overlay höher: oben und unten beschneiden
    return (0.0, (h - bh) / 2, float(w), (h + bh) / 2)


def _premultiplied_overlay(overlay: Image.Image, target_icc: bytes | None,
                           size: tuple[int, int]) -> np.ndarray:
    """Overlay als Float-Array mit vormultipliziertem Alpha in Zielgröße und Zielfarbraum."""
    source_icc = _rgb_icc(overlay)
    rgba = overlay.convert("RGBA")

    # Farben in den Farbraum des Screenshots umrechnen (z. B. sRGB → Display P3),
    # damit das Overlay auf dem iPhone exakt so aussieht wie gestaltet.
    if source_icc != target_icc:
        rgb = ImageCms.profileToProfile(
            rgba.convert("RGB"), _profile(source_icc), _profile(target_icc),
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB",
        )
        rgb.putalpha(rgba.getchannel("A"))
        rgba = rgb

    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:4]
    if rgba.size == size:
        return arr

    # Lanczos-Skalierung je Kanal in 32-Bit-Float auf vormultiplizierten Werten:
    # keine dunklen Säume an transparenten Kanten, keine Rundungsverluste.
    box = _fill_box(rgba.size, size)
    channels = [
        np.asarray(Image.fromarray(np.ascontiguousarray(arr[..., c]))
                   .resize(size, Image.Resampling.LANCZOS, box=box))
        for c in range(4)
    ]
    out = np.stack(channels, axis=-1)
    alpha = np.clip(out[..., 3:4], 0.0, 1.0)
    out[..., :3] = np.clip(out[..., :3], 0.0, alpha)
    out[..., 3:4] = alpha
    return out


@st.cache_data(show_spinner=False, max_entries=16)
def render(screenshot: bytes, overlay_path: str, overlay_mtime: float) -> tuple[bytes, dict]:
    src = Image.open(io.BytesIO(screenshot))
    source_format = src.format or "Bild"
    src = ImageOps.exif_transpose(src)

    icc = _rgb_icc(src)
    xmp = src.info.get("XML:com.adobe.xmp") or src.info.get("xmp")
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", "ignore")

    base = src.convert("RGBA")
    opaque = base.getchannel("A").getextrema() == (255, 255)
    if opaque:
        base = base.convert("RGB")

    with Image.open(overlay_path) as overlay:
        ov = _premultiplied_overlay(overlay, icc, base.size)

    # Alpha-Compositing ("source over") in Float-Präzision. Wo das Overlay
    # transparent ist, bleiben die Pixel des Screenshots bitgenau erhalten.
    px = np.asarray(base, dtype=np.float32) / 255.0
    ov_a = ov[..., 3:4]
    if opaque:
        result = ov[..., :3] + px * (1.0 - ov_a)
    else:
        base_a = px[..., 3:4]
        out_a = ov_a + base_a * (1.0 - ov_a)
        premult = ov[..., :3] + px[..., :3] * base_a * (1.0 - ov_a)
        rgb = np.divide(premult, out_a, out=np.zeros_like(premult), where=out_a > 0)
        result = np.concatenate([rgb, out_a], axis=-1)
    image = Image.fromarray(np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8))

    # Verlustfrei als PNG speichern, mit Farbprofil und Metadaten des Originals.
    pnginfo = PngImagePlugin.PngInfo()
    if xmp:
        pnginfo.add_itxt("XML:com.adobe.xmp", xmp)
    options: dict = {"pnginfo": pnginfo, "compress_level": 6}
    if icc:
        options["icc_profile"] = icc
    if src.info.get("exif"):
        options["exif"] = src.info["exif"]
    if src.info.get("dpi"):
        options["dpi"] = src.info["dpi"]
    buffer = io.BytesIO()
    image.save(buffer, "PNG", **options)

    meta = {"width": image.width, "height": image.height,
            "format": source_format, "profile": _profile_name(icc)}
    return buffer.getvalue(), meta


# ---------------------------------------------------------------- Oberfläche

def main() -> None:
    st.set_page_config(page_title="Screenshot Overlay", page_icon=":material/layers:",
                       layout="centered")
    st.html("""<style>
      [data-testid="stMainBlockContainer"] { padding-top: 2.5rem; padding-bottom: 4rem; }
    </style>""")
    st.markdown("#### :material/layers: Screenshot Overlay")

    overlays = list_overlays()
    if not overlays:
        st.error("Im Ordner `overlays/` liegt kein PNG. Lege dort mindestens ein Overlay ab.",
                 icon=":material/error:")
        st.stop()

    choice = st.radio(":material/style: Overlay", list(overlays), horizontal=True)
    upload = st.file_uploader(":material/add_photo_alternate: Screenshot",
                              type=["png", "jpg", "jpeg", "webp"])
    if upload is None:
        st.caption(":material/info: Wähle einen Screenshot aus deiner Fotomediathek.")
        return

    path = overlays[choice]
    try:
        with st.spinner("Overlay wird angewendet …"):
            png, meta = render(upload.getvalue(), str(path), path.stat().st_mtime)
    except (OSError, ValueError, Image.DecompressionBombError):
        st.error("Diese Datei lässt sich nicht als Bild öffnen. Lade einen PNG- oder "
                 "JPEG-Screenshot hoch.", icon=":material/broken_image:")
        return

    st.caption(f":material/aspect_ratio: {meta['width']} × {meta['height']} px, "
               f"{meta['format']}, {meta['profile']}")

    # Ergebnis in Originalauflösung als echtes PNG einbetten (st.image würde es
    # ggf. neu kodieren). So kann es auf dem iPhone direkt in Fotos gesichert werden.
    b64 = base64.b64encode(png).decode("ascii")
    st.html(f'<img src="data:image/png;base64,{b64}" alt="Screenshot mit Overlay" '
            'style="display:block;width:100%;height:auto;border-radius:16px;'
            '-webkit-touch-callout:default">')
    st.caption(":material/touch_app: Bild gedrückt halten und „Zu Fotos hinzufügen“ wählen.")

    stem = Path(upload.name).stem
    st.download_button("PNG herunterladen", png, file_name=f"{stem}_{path.stem}.png",
                       mime="image/png", icon=":material/download:", type="primary",
                       width="stretch")


if __name__ == "__main__":
    main()
