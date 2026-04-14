# arducam-tui-visualizer

TUI depth rendering API.

Usage:
    from arducam_tui import LiveViewer

    viewer = LiveViewer(width=240, height=180)
    with viewer:
        while running:
            frame = cam.get_frame()
            viewer.draw(frame)