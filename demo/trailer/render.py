"""Render the trailer (index.html) frame by frame to video. Every frame is drawn by the page's own render(t), so the
output is exact regardless of machine speed.

    python demo/trailer/render.py                        # trailer.mp4 (1920x1080, 30 fps, with music) + trailer.gif (silent)
    python demo/trailer/render.py --stills 5 12 30       # PNG stills at those seconds into stills/, for checking

Needs playwright (python -m playwright install chromium) and ffmpeg on PATH.
"""
import argparse, asyncio, functools, http.server, subprocess, threading
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent


def serve():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    h = functools.partial(Quiet, directory=str(HERE))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


async def main(a):
    srv = serve()
    url = f"http://127.0.0.1:{srv.server_address[1]}/index.html?capture=1"
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=a.scale, color_scheme="dark")
        await pg.goto(url)
        await pg.evaluate("window.__ready")
        await pg.wait_for_timeout(300)
        if a.stills:
            out = Path(a.stills_dir)
            out.mkdir(exist_ok=True)
            for t in a.stills:
                await pg.evaluate(f"window.__seek({t})")
                await pg.screenshot(path=str(out / f"t{t:05.1f}.png"))
            print(f"{len(a.stills)} stills -> {out}")
            return
        dur = await pg.evaluate("window.__duration")
        n = int(dur * a.fps)
        w, h = int(1280 * a.scale), int(720 * a.scale)
        mp4 = HERE / "trailer.mp4"
        music = HERE / "media" / "living-voyage.mp3"      # already trimmed to the trailer and faded out
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(a.fps), "-c:v", "mjpeg", "-i", "-",
                               "-i", str(music), "-map", "0:v", "-map", "1:a", "-vf", f"scale={w}:{h}", "-c:v", "libx264",
                               "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
                               "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
        for i in range(n):
            await pg.evaluate(f"window.__seek({i / a.fps})")
            ff.stdin.write(await pg.screenshot(type="jpeg", quality=93))
            if i % (a.fps * 5) == 0:
                print(f"  {i / a.fps:5.1f}s / {dur}s", flush=True)
        ff.stdin.close()
        ff.wait()
        await b.close()
    print(f"{n} frames -> {mp4}")
    # README preview: smaller GIF with a per-video palette
    gif = HERE / "trailer.gif"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf",
                    f"fps={a.gif_fps},scale={a.gif_width}:-1:flags=lanczos,split[x][y];[x]palettegen=max_colors=96:stats_mode=diff[p];"
                    f"[y][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(gif)], check=True)
    print(f"-> {gif} ({gif.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.5, help="device scale factor (1.5 = 1920x1080)")
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--stills", type=float, nargs="*")
    ap.add_argument("--stills-dir", default=str(HERE / "stills"))
    asyncio.run(main(ap.parse_args()))
