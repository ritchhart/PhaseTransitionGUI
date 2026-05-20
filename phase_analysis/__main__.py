"""Entry point: python -m phase_analysis"""

import sys
import tkinter as tk
from phase_analysis.gui import TransitionAnalysisGUI


def get_terminal_position():
    """
    Attempt to get terminal position cross-platform.
    Returns (x, y) of terminal center, or None on failure.
    """
    if sys.platform.startswith('linux'):
        try:
            import subprocess
            win_id = subprocess.check_output(
                ['xdotool', 'getactivewindow'], stderr=subprocess.DEVNULL
            ).decode().strip()
            out = subprocess.check_output(
                ['xdotool', 'getwindowgeometry', '--shell', win_id],
                stderr=subprocess.DEVNULL
            ).decode()
            p = {k: int(v) for k, v in
                 (line.split('=') for line in out.strip().split('\n'))}
            return p['X'] + p['WIDTH'] // 2, p['Y'] + p['HEIGHT'] // 2
        except Exception:
            pass

    elif sys.platform == 'darwin':
        try:
            import subprocess
            script = '''
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
            end tell
            tell application frontApp
                set b to bounds of front window
                return item 1 of b & "," & item 2 of b & "," & item 3 of b & "," & item 4 of b
            end tell
            '''
            out = subprocess.check_output(
                ['osascript', '-e', script], stderr=subprocess.DEVNULL
            ).decode().strip()
            x1, y1, x2, y2 = map(int, out.split(','))
            return (x1 + x2) // 2, (y1 + y2) // 2
        except Exception:
            pass

    elif sys.platform == 'win32':
        try:
            import ctypes
            import ctypes.wintypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            cx = (rect.left + rect.right) // 2
            cy = (rect.top + rect.bottom) // 2
            return cx, cy
        except Exception:
            pass

    return None


def main():
    pos = get_terminal_position()

    root = tk.Tk()
    app = TransitionAnalysisGUI(root)

    win_w, win_h = 2000, 1500
    root.update_idletasks()

    if pos:
        cx, cy = pos
        x = cx - win_w // 2
        y = cy - win_h // 2
    else:
        x = (root.winfo_screenwidth() - win_w) // 2
        y = (root.winfo_screenheight() - win_h) // 2

    root.geometry(f"{win_w}x{win_h}+{x}+{y}")
    root.mainloop()


if __name__ == "__main__":
    main()
