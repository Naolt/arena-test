from pathlib import Path
from PIL import Image


def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    import fitz
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def pptx_to_images(pptx_path: str, output_dir: str = "/tmp/slides") -> list[Image.Image]:
    import subprocess
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(out), pptx_path],
        check=True,
    )
    pdf_path = out / (Path(pptx_path).stem + ".pdf")
    return pdf_to_images(str(pdf_path))


def video_to_keyframes(video_path: str, fps: float = 1.0, max_frames: int = 64) -> list[Image.Image]:
    """
    Extracts one frame per `fps` seconds (default: 1 frame/sec), capped at `max_frames`.
    For short marketing ads (< 60s) this gives ~1 frame per second.
    Raise `fps` for denser sampling or lower it for long videos.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(video_fps / fps))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames
