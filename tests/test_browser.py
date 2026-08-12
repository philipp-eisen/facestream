"""The web client, in Chromium, against the stub backend.

The stub runs the real aiortc peer connection and the real ProcessFrameTrack,
so these exercise the whole signalling and media path -- everything except the
GPU swap itself.
"""

import pytest
from conftest import visible, wait_for_stream

pytestmark = pytest.mark.browser

FACE = "http://127.0.0.1:8123/face.jpg"
SERVER = "ws://127.0.0.1:8123"


def received(page):
    return page.evaluate("async () => await fetch('/received').then(r => r.json())")


@pytest.fixture
def obs_page(page, stub_server):
    page.goto(f"{stub_server}/?obs=1&face={FACE}&server={SERVER}&stats=1&res=720")
    wait_for_stream(page)
    return page


class TestObsMode:
    def test_hides_every_piece_of_chrome(self, obs_page):
        assert obs_page.evaluate("document.body.classList.contains('obs-mode')")
        for selector in [
            "#titleContainer",
            "#uploadContainer",
            "#localVideo",
            ".social-links",
        ]:
            assert not visible(obs_page, selector), selector
        assert visible(obs_page, "#remoteVideo")

    def test_video_fills_the_canvas(self, obs_page):
        fit = obs_page.eval_on_selector(
            "#remoteVideo", "el => getComputedStyle(el).objectFit"
        )
        assert fit == "cover"

        box = obs_page.eval_on_selector(
            "#remoteVideo",
            "el => { const r = el.getBoundingClientRect(); return [r.width, r.height]; }",
        )
        assert box == [1280, 720]

    def test_negotiates_the_requested_resolution(self, obs_page):
        """Without the SDP bitrate hint this starts at 640x360 and creeps up."""
        dimensions = obs_page.evaluate(
            "() => { const v = document.getElementById('remoteVideo');"
            " return [v.videoWidth, v.videoHeight]; }"
        )
        assert dimensions == [1280, 720]

    def test_starts_without_being_clicked(self, obs_page):
        messages = received(obs_page)["messages"]
        assert "upload_image" in messages
        assert "offer" in messages
        assert not visible(obs_page, "#errorBanner")

    def test_frames_flow(self, obs_page):
        obs_page.wait_for_timeout(2000)
        assert received(obs_page)["frames"] > 10

    def test_stats_overlay_reports_live_numbers(self, obs_page):
        obs_page.wait_for_timeout(2500)
        assert visible(obs_page, "#statsOverlay")
        assert "fps" in (obs_page.text_content("#statsOverlay") or "")

    def test_heartbeat_keeps_the_session_alive(self, obs_page):
        before = received(obs_page)["pings"]
        obs_page.wait_for_timeout(17000)  # the client heartbeats every 15s
        assert received(obs_page)["pings"] > before
        assert obs_page.evaluate(
            "document.getElementById('remoteVideo').videoWidth > 0"
        )


class TestOptions:
    def test_mirror_and_fit(self, page, stub_server):
        page.goto(
            f"{stub_server}/?obs=1&face=lucy&server={SERVER}&mirror=0&fit=contain"
        )
        page.wait_for_timeout(1500)

        assert page.evaluate("document.body.classList.contains('no-mirror')")
        transform = page.eval_on_selector(
            "#remoteVideo", "el => getComputedStyle(el).transform"
        )
        assert transform in ("none", "matrix(1, 0, 0, 1, 0, 0)")
        assert (
            page.eval_on_selector(
                "#remoteVideo", "el => getComputedStyle(el).objectFit"
            )
            == "contain"
        )

    def test_preset_keys_resolve(self, page, stub_server):
        page.goto(f"{stub_server}/?obs=1&server={SERVER}")
        page.wait_for_timeout(1000)

        assert "Lucy_Liu" in page.evaluate("resolveFaceUrl('lucy')")
        assert page.evaluate("resolveFaceUrl('nobody')") is None
        assert page.evaluate("resolveFaceUrl('https://example.com/a.jpg')") == (
            "https://example.com/a.jpg"
        )

    @pytest.mark.parametrize(
        "href,server,expected",
        [
            (
                "https://facestream.example.com/",
                None,
                "wss://facestream.example.com/ws",
            ),
            ("http://localhost:8000/", None, "ws://localhost:8000/ws"),
            (
                "https://x--facestream-main-web.modal.run/",
                None,
                "wss://x--facestream-main-web.modal.run/ws",
            ),
            ("https://pages.example.com/", "app.modal.run", "wss://app.modal.run/ws"),
            (
                "https://pages.example.com/",
                "ws://localhost:8000",
                "ws://localhost:8000/ws",
            ),
        ],
    )
    def test_backend_url_resolution(self, page, stub_server, href, server, expected):
        """A custom domain must reach its own backend, not the upstream demo."""
        page.goto(f"{stub_server}/?obs=1&server={SERVER}")
        page.wait_for_timeout(500)

        got = page.evaluate(
            """([href, server]) => {
              const url = new URL(href);
              const p = new URLSearchParams(server ? {server} : {});
              const override = p.get("server");
              if (override) {
                const withScheme = /^wss?:\\/\\//i.test(override)
                  ? override : `wss://${override}`;
                return withScheme.replace(/\\/+$/, "") + "/ws";
              }
              if (/^https?:$/.test(url.protocol)) {
                const scheme = url.protocol === "https:" ? "wss" : "ws";
                return `${scheme}://${url.host}/ws`;
              }
              return "wss://philipp-eisen--facestream-main-web.modal.run/ws";
            }""",
            [href, server],
        )
        assert got == expected

    def test_defaults_to_its_own_origin(self, page, stub_server):
        sockets = []
        page.on("websocket", lambda ws: sockets.append(ws.url))

        page.goto(f"{stub_server}/?obs=1&face={FACE}")  # note: no &server=
        wait_for_stream(page)

        assert sockets == ["ws://127.0.0.1:8123/ws"]
        assert page.evaluate("() => pc && pc.connectionState === 'connected'")


class TestErrors:
    def test_unknown_face_says_so(self, page, stub_server):
        page.goto(f"{stub_server}/?obs=1&face=nosuchperson&server={SERVER}")
        page.wait_for_timeout(1500)

        assert visible(page, "#errorBanner")
        assert "Unknown face" in page.text_content("#errorBanner")

    def test_missing_face_explains_what_to_add(self, page, stub_server):
        page.goto(f"{stub_server}/?obs=1&server={SERVER}")
        page.wait_for_timeout(1500)

        assert visible(page, "#errorBanner")
        assert "face=" in page.text_content("#errorBanner")

    def test_server_side_errors_surface(self, page, stub_server):
        page.goto(f"{stub_server}/?obs=1&server={SERVER}")
        page.wait_for_timeout(1200)
        page.evaluate(
            "() => ws.send(JSON.stringify({type:'upload_image', image:'x', fail:true}))"
        )
        page.wait_for_timeout(1200)

        assert "No face found" in page.text_content("#errorBanner")


class TestReconnect:
    def test_obs_mode_recovers_on_its_own(self, page, stub_server):
        dialogs = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))

        page.goto(f"{stub_server}/?obs=1&face={FACE}&server={SERVER}")
        wait_for_stream(page)
        page.evaluate("() => ws.send(JSON.stringify({type:'kill'}))")

        page.wait_for_function("() => !ws || ws.readyState !== 1", timeout=10000)
        assert "Reconnecting" in page.text_content("#errorBanner")
        assert not dialogs

        # The <video> holds its last frame, so wait for a genuinely new peer
        # connection rather than for videoWidth.
        page.wait_for_function(
            "() => serverReady === true && pc && pc.connectionState === 'connected'",
            timeout=40000,
        )
        page.wait_for_timeout(1000)

        assert page.evaluate("document.getElementById('remoteVideo').videoWidth > 0")
        assert not visible(page, "#errorBanner")
        assert not dialogs

    def test_normal_mode_still_asks_first(self, page, stub_server):
        """Upstream made reconnects deliberate so a forgotten tab can't wake GPUs."""
        dialogs = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))

        page.goto(f"{stub_server}/?server={SERVER}")
        page.wait_for_function("() => serverReady === true", timeout=20000)
        page.evaluate("() => ws.send(JSON.stringify({type:'kill'}))")
        page.wait_for_timeout(3000)

        assert len(dialogs) == 1
        assert "Going back" in dialogs[0]

        page.wait_for_timeout(3000)
        assert len(dialogs) == 1  # no silent retry


class TestNormalMode:
    def test_is_unchanged(self, page, stub_server):
        page.goto(f"{stub_server}/?server={SERVER}")
        page.wait_for_selector("#uploadContainer", timeout=15000)
        page.wait_for_timeout(1500)

        assert not page.evaluate("document.body.classList.contains('obs-mode')")
        assert visible(page, "#titleContainer")
        assert visible(page, "#uploadContainer")
        assert visible(page, ".social-links")
        assert not visible(page, "#statsOverlay")
        assert not page.evaluate("document.body.classList.contains('no-mirror')")
        assert page.evaluate("!document.getElementById('uploadButton').disabled")
        assert page.evaluate(
            "!document.querySelector('.preset-image').classList.contains('disabled')"
        )
