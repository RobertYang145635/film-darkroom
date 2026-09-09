"""跨进程抓取真实 negative-lab 窗口画面（用户看到什么就抓什么）。"""

import ctypes
import ctypes.wintypes as wt
import sys

from PIL import Image

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


def find_window(title_part: str):
    result = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _l):
        n = user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if title_part in buf.value:
                rect = wt.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                if rect.right - rect.left > 50:
                    result.append((hwnd, buf.value, rect))
        return True

    user32.EnumWindows(cb, 0)
    return result


def capture(hwnd, rect, path):
    w, h = rect.right - rect.left, rect.bottom - rect.top
    hdc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mem, bmp)
    # PW_RENDERFULLCONTENT = 2：允许抓取 DirectComposition/硬件加速窗口
    ok = user32.PrintWindow(hwnd, mem, 2)
    bmi = ctypes.create_string_buffer(40)
    ctypes.memset(bmi, 0, 40)
    bmi[0:4] = (40).to_bytes(4, "little")
    bmi[4:8] = w.to_bytes(4, "little", signed=True)
    bmi[8:12] = (-h).to_bytes(4, "little", signed=True)
    bmi[12:14] = (1).to_bytes(2, "little")
    bmi[14:16] = (32).to_bytes(2, "little")
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, bmi, 0)
    img = Image.frombuffer("RGBX", (w, h), buf.raw, "raw", "RGBX", 0, 1)
    img.convert("RGB").save(path)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)
    return ok


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "negative-lab"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/v6_ui/diag/real_window.png"
    wins = find_window(part)
    if not wins:
        print("WINDOW NOT FOUND", part)
        sys.exit(1)
    hwnd, title, rect = wins[0]
    ok = capture(hwnd, rect, out)
    print(f"captured hwnd={hwnd} title={title!r} rect=({rect.right-rect.left}x"
          f"{rect.bottom-rect.top}) printwindow_ok={ok} -> {out}")
