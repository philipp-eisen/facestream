import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB_HOST = "127.0.0.1"
STUB_PORT = 8123
STUB_BASE = f"http://{STUB_HOST}:{STUB_PORT}"


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture(scope="session")
def stub_server():
    """A local stand-in for the Modal deployment.

    Speaks the same websocket protocol and runs the real aiortc peer connection
    and the real ProcessFrameTrack, with a cheap CPU tint in place of the GPU
    face swap.
    """
    process = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "tests" / "stub_server.py")],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 60
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Stub server exited early:\n{process.stdout.read()}")
        if _port_open(STUB_HOST, STUB_PORT):
            break
        time.sleep(0.25)
    else:
        process.terminate()
        raise RuntimeError("Stub server did not start in time")

    yield STUB_BASE

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture(scope="session")
def browser():
    playwright = pytest.importorskip("playwright.sync_api").sync_playwright().start()

    # `playwright install chromium` puts the browser where launch() looks. Set
    # FACESTREAM_CHROMIUM to point at an existing build instead.
    options = {}
    if executable := os.environ.get("FACESTREAM_CHROMIUM"):
        options["executable_path"] = executable

    try:
        browser = playwright.chromium.launch(
            args=[
                "--no-sandbox",
                # The page needs a camera, and nothing here can click a
                # permission prompt -- the same flags OBS needs.
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
            ],
            **options,
        )
    except Exception:
        playwright.stop()
        raise

    yield browser

    browser.close()
    playwright.stop()


@pytest.fixture
def page(browser):
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    yield page
    page.close()


def visible(page, selector: str) -> bool:
    return page.eval_on_selector(
        selector,
        "el => { const s = getComputedStyle(el);"
        " return s.display !== 'none' && s.visibility !== 'hidden'; }",
    )


def wait_for_stream(page, timeout=30000):
    page.wait_for_function(
        "() => document.getElementById('remoteVideo').videoWidth > 0", timeout=timeout
    )
