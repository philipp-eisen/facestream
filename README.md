# FaceStream

This is a web app that let's you swap your face in real-time without having a powerful
machine. You can try it out live [here](https://facestream.phileisen.com).

See the gif below for a demo:

[![Demo](assets/demo.gif)](https://facestream.phileisen.com)

## Motivation

I discovered the [Deep Live Cam](https://github.com/hacksider/Deep-Live-Cam) project and tried to run it on my M1 MacBook, but only got 0.5 FPS. So I wanted to explore how fast and at what latency you could get this to run on a remote server via WebRTC.

## Deploying from GitHub

If you would rather not install anything locally, the workflows in
[.github/workflows](.github/workflows) run the tests and deploy to Modal for
you. Pushing to `main` deploys; opening a pull request just runs the checks.

### Accounts you need

| Account | What for | Cost |
| --- | --- | --- |
| [GitHub](https://github.com) | Runs the workflows. | Free for public repos. |
| [Modal](https://modal.com) | Runs the GPU backend. | Pay per second of GPU time. |
| [Cloudflare](https://dash.cloudflare.com) | *Optional.* TURN relay, needed for cellular networks. Also a custom domain if you want one. | Free tier covers the key; relayed traffic is billed per GB. |

Nothing else. No Hugging Face account — the model weights are public and are
fetched on first boot into a Modal volume, so only the first container start
pays for the download.

### Repository secrets

**Settings → Secrets and variables → Actions → Secrets.** These are the only
values that need to be secret:

| Secret | Required | Where it comes from |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | yes | [modal.com/settings/tokens](https://modal.com/settings/tokens) → New token |
| `MODAL_TOKEN_SECRET` | yes | Same page, shown alongside the ID |
| `TURN_TOKEN_ID` | only with TURN | Cloudflare → Realtime → TURN → your key |
| `TURN_API_TOKEN` | only with TURN | Same place |

The deploy workflow pushes the two TURN values into a Modal secret named
`facestream` on every run, so you never have to create it by hand.

### Repository variables

**Settings → Secrets and variables → Actions → Variables.** All optional — the
defaults in the table are what you get if you set nothing.

| Variable | Default | Set it when |
| --- | --- | --- |
| `FACESTREAM_GPU` | `L40S` | You want different hardware. See [Choosing compute](#choosing-compute). |
| `FACESTREAM_REGION` | Modal picks | You know where your users are, e.g. `us-east-1`. |
| `FACESTREAM_MIN_CONTAINERS` | `0` | You want no cold start. `1` keeps a GPU warm, and bills for it. |
| `FACESTREAM_MAX_CONTAINERS` | unlimited | You want a ceiling on spend. |
| `FACESTREAM_MAX_CONCURRENT_INPUTS` | `2` | You want viewers to share a GPU rather than get one each. |
| `FACESTREAM_TIMEOUT` | `3600` | You want a shorter hard cap per session. |
| `FACESTREAM_TURN` | `0` | Set to `1` to switch on Cloudflare TURN. |
| `MODAL_ENVIRONMENT` | your default | Your Modal workspace has several environments. |
| `FACESTREAM_URL` | unset | You want the deploy to verify `/healthz` afterwards, e.g. `https://facestream.example.com`. |

A minimal production setup is four secrets-and-variables entries:
`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `FACESTREAM_MIN_CONTAINERS=1` and
`FACESTREAM_MAX_CONTAINERS=4`.

### What the workflows do

- **CI** (on every pull request) — ruff, the unit tests, and a Chromium job
  that drives the real page against a stub backend running the actual aiortc
  peer connection. No GPU and no Modal account involved, so it runs on a
  standard free runner.
- **Deploy** (on push to `main`, or manually from the Actions tab) — runs lint
  and unit tests, syncs the TURN secret if TURN is on, deploys to Modal, then
  optionally checks `/healthz`. Deploys are serialised, so two pushes in quick
  succession queue rather than race.

To require approval before anything reaches production, add required reviewers
to the `production` environment under **Settings → Environments**. The deploy
job already targets it.

To deploy without pushing code, use **Actions → Deploy → Run workflow**.

## How to run this yourself

You only need this section if you want to deploy from your own machine rather
than from GitHub.

### Prerequisites

To run this yourself you wil need:

- [uv](https://docs.astral.sh/uv/)
- A [modal](https://modal.com/) account as well configured credentials

Modal is the only account involved — there are no API keys to paste anywhere and
no model to download by hand. The face swap weights are pulled from Hugging Face
on first boot and cached in a Modal volume, so only the first container start
pays for it.

Set the credentials up once with:

```
uv run modal setup
```

Optional extras, depending on what you want:

- A **source photo** of the face you want to wear. Any clear, front-facing
  JPEG/PNG works; six presets are built in if you just want to try it.
- A **webcam**, and Chrome or OBS to view the result.
- A **Cloudflare TURN app** — only if you need this to work over a cellular
  network. See the section further down.

Then you can run the following command to start the server:

```
uv run modal serve facestream.main
```

Or this command to deploy it:

```
uv run modal deploy -m facestream.main
```

## Choosing compute

Every knob below is an environment variable read when you deploy, so you can
change hardware without touching code:

```
FACESTREAM_GPU=L40S uv run modal deploy -m facestream.main
```

### Which GPU

The pipeline runs two small models per frame — SCRFD face detection at 640x640
and inswapper_128 at 128x128. It is bound by kernel launch latency, not by
memory bandwidth, so the fastest card is **not** the biggest card. Past roughly
an L40S you are paying for capacity this workload never touches.

| `FACESTREAM_GPU` | Notes |
| --- | --- |
| `T4` | The old default. Works, and is the cheapest thing that does. |
| `L4` | Solidly faster than T4 for not much more. Good value pick. |
| `A10G` | Similar tier to L4, sometimes better availability. |
| `L40S` | **Default.** Best latency per dollar for this workload. |
| `A100-40GB`, `A100-80GB` | More expensive than L40S with no gain here. |
| `H100`, `H200`, `B200` | Overkill. You are renting HBM you will not use. |

See [Modal's pricing page](https://modal.com/pricing) for current rates.

If the stream still isn't smooth after switching cards, the bottleneck is
probably not the GPU:

- **The network.** WebRTC latency is mostly round trips. Pin the deployment near
  yourself with `FACESTREAM_REGION=us-east-1` (or `eu-west-1`, etc.). This
  usually helps more than the GPU does.
- **Your upstream bandwidth.** 720p30 needs roughly 2.5 Mbit/s upstream. Lower
  it with `?bitrate=1500` on the page URL, or drop to `?res=480`.
- **Cold starts.** The first connection waits ~20s for a container. Set
  `FACESTREAM_MIN_CONTAINERS=1` to keep one warm — that bills GPU time while
  idle, so it is off by default.

### All settings

| Variable | Default | What it does |
| --- | --- | --- |
| `FACESTREAM_GPU` | `L40S` | GPU type, any string Modal accepts. |
| `FACESTREAM_CPU` | `8` | Cores per container. WebRTC decode/encode is CPU work, so don't go too low. |
| `FACESTREAM_TIMEOUT` | `3600` | Hard cap on a single session, in seconds. |
| `FACESTREAM_WS_IDLE_TIMEOUT` | `120` | Hang up on a peer that has stopped talking. |
| `FACESTREAM_SCALEDOWN_WINDOW` | `300` | How long an idle container is kept. Longer means OBS scene switches reconnect instantly. |
| `FACESTREAM_MIN_CONTAINERS` | `0` | Containers kept warm. `1` removes cold starts, and bills for it. |
| `FACESTREAM_MAX_CONTAINERS` | unset | Ceiling on concurrent containers, i.e. on spend. |
| `FACESTREAM_MAX_CONCURRENT_INPUTS` | `2` | Requests per container. Must stay above 1 — a live websocket holds one slot for the whole session. At 2 each stream effectively gets its own GPU; raise it to share one. |
| `FACESTREAM_REGION` | unset | Modal region, e.g. `us-east-1`. |
| `FACESTREAM_TURN` | `0` | Relay through Cloudflare TURN. Needs the `facestream` Modal secret — see below. |
| `FACESTREAM_DET_SIZE` | `640` | Detector input size. `320` is much cheaper but misses small faces. |
| `FACESTREAM_TRACK_DET_SIZE` | `320` | Detector input size while tracking a face already found. |
| `FACESTREAM_FACE_TRACKING` | `1` | Search a crop around the last known face instead of the whole frame. |
| `FACESTREAM_FAST_PASTE_BACK` | `1` | Composite the face inside its bounding box rather than over the whole frame. |
| `FACESTREAM_STATS_INTERVAL` | `150` | Log a timing summary every N frames. `0` disables. |

The session timeout deserves a note. The upstream default was 120 seconds, which
protects the public demo's GPU budget but cuts a stream off after two minutes.
It is an hour here, and the client heartbeats every 15s so a browser that dies
without closing its socket gets dropped after `FACESTREAM_WS_IDLE_TIMEOUT`
rather than holding a GPU open. Set `FACESTREAM_TIMEOUT` back down if you would
rather have the hard cap.

`GET /healthz` on the deployment reports the configuration a container is
actually running with.

## Use it as a camera in OBS

Add `?obs=1` to the URL for a chromeless, full-bleed version of the page: no
title, no upload panel, no preview thumbnail, nothing but the swapped video. It
picks its face from the URL, so nothing needs clicking, and it reconnects on its
own instead of putting up a dialog no one is there to answer.

```
https://<your-app>.modal.run/?obs=1&face=rock
```

### Option A: Browser Source

The cleanest route — OBS composites the video directly, no second window.

1. **Launch OBS with camera access enabled.** Its embedded browser blocks
   webcams unless you ask for them, and it cannot show a permission prompt:

   - Windows — edit your shortcut's Target:
     `"C:\Program Files\obs-studio\bin\64bit\obs64.exe" --enable-media-stream --use-fake-ui-for-media-stream`
   - macOS: `/Applications/OBS.app/Contents/MacOS/OBS --enable-media-stream --use-fake-ui-for-media-stream`
   - Linux: `obs --enable-media-stream --use-fake-ui-for-media-stream`

   Without `--enable-media-stream` the source stays black and the page reports
   that the MediaDevices API is missing.

2. **Add a Browser source** with your `?obs=1&face=...` URL, Width `1280`,
   Height `720` (match `?res=`, so the page renders 1:1).

3. **Tick "Shutdown source when not visible"** and **"Refresh browser when scene
   becomes active"**. The stream — and the GPU bill — then only runs while the
   scene is live.

4. If OBS grabs the wrong webcam, name the one you want with
   `&camera=logitech` (any part of the device label, case insensitive).

### Option B: Window Capture

No launch flags needed, at the cost of a browser window on screen and a little
more CPU.

1. Open the `?obs=1&face=...` URL in Chrome, ideally with `--kiosk`.
2. Add a **Window Capture** source in OBS and pick that window.

### Sending it to Zoom, Meet or Discord

Once the source looks right in OBS, hit **Start Virtual Camera**. FaceStream
then shows up as a webcam in any other app.

### URL options

| Option | Default | What it does |
| --- | --- | --- |
| `obs=1` | off | Chromeless output, auto-reconnect, no dialogs. |
| `face=<key\|url>` | none | Start immediately with this face. Keys: `rock`, `michelle`, `adam`, `selena`, `lebron`, `lucy`. A URL has to allow cross-origin fetches. |
| `res=` | `720` | Camera resolution: `480`, `540`, `720` or `1080`. |
| `fps=` | `30` | Camera frame rate. |
| `bitrate=` | `3000` | Max video bitrate in kbit/s. Lower it on a thin uplink. |
| `mirror=0` | mirrored | Stop flipping the image, so text reads correctly on stream. |
| `fit=contain` | `cover` | Letterbox instead of filling the canvas. |
| `camera=<text>` | none | Pick a webcam by label substring or deviceId. |
| `stats=1` | off | Overlay live resolution, fps, round trip time and packet loss. |
| `degradation=` | `maintain-resolution` in OBS mode | How the encoder gives up under pressure: `maintain-resolution`, `maintain-framerate` or `balanced`. |
| `retries=` | `20` | Reconnect attempts before giving up. |
| `server=<host>` | this host | Point the page at a different deployment. |

Start with `&stats=1` while you tune. If it reports a resolution below what you
asked for, you are bandwidth limited, not GPU limited — lower `bitrate` or
`res`.

## Cloudflare

### TURN, for cellular and locked-down networks

WebRTC needs a relay when neither end can be reached directly, which on most
cellular networks and plenty of corporate wifi is the normal case. Without one
the connection simply never establishes.

1. Create a TURN key in
   [Cloudflare Realtime](https://developers.cloudflare.com/realtime/turn/). You
   get a Token ID and an API token.

2. Put them in a Modal secret named `facestream`:

   ```
   uv run modal secret create facestream \
       TURN_TOKEN_ID=your-turn-token-id \
       TURN_API_TOKEN=your-turn-api-token
   ```

3. Deploy with TURN switched on:

   ```
   FACESTREAM_TURN=1 uv run modal deploy -m facestream.main
   ```

Leave `FACESTREAM_TURN` unset and the app runs on STUN alone — naming a Modal
secret that doesn't exist fails the deploy, so this stays opt-in rather than
something you edit in the source.

Cloudflare only issues short-lived TURN credentials, so the backend mints a
fresh set per session from your key. If Cloudflare is unreachable or the token
is rejected, the session falls back to STUN and logs it rather than failing.
Relayed traffic is billed per GB, so it is worth knowing that TURN is only used
when a direct path can't be found.

### Putting Cloudflare in front

The page talks to whatever origin served it, so a custom domain works with no
configuration: point `facestream.yourdomain.com` at the Modal deployment
([Modal custom domains](https://modal.com/docs/guide/custom-domains)), proxy it
through Cloudflare if you like, and the websocket follows.

Two things worth knowing:

- **Leave WebSocket support on** in Cloudflare (it is on by default). The whole
  signalling path is one websocket.
- Cloudflare closes idle websockets. The client heartbeats every 15s, so this
  doesn't bite, but don't lengthen that interval past a minute.

To serve the page from Cloudflare Pages while Modal keeps the GPU backend, host
`web/index.html` on Pages and point it at the backend explicitly:

```
https://your-pages-site.com/?obs=1&face=rock&server=your-app.modal.run
```

Nothing else needs configuring — the websocket handshake isn't subject to CORS.

## Running in production

- **Turn off cold starts.** `FACESTREAM_MIN_CONTAINERS=1` keeps a GPU warm so
  the first connection doesn't sit through ~20s of container boot. This is the
  single biggest difference between a demo and something that feels ready when
  you hit it.
- **Cap the spend.** `FACESTREAM_MAX_CONTAINERS` bounds how many GPUs can run
  at once. Without it, a link that gets shared around scales as far as your
  budget does.
- **Decide how much GPU each viewer gets.** The default of
  `FACESTREAM_MAX_CONCURRENT_INPUTS=2` gives each stream a container to itself.
  Raise it to share a GPU between viewers and spend less, at the cost of frame
  rate when more than one is connected.
- **Set `FACESTREAM_REGION`** near your users. On a good GPU the network is the
  larger share of end-to-end latency.
- **Enable TURN** if anyone will connect over mobile data. See above.
- **Watch it.** `GET /healthz` reports the live configuration. Container logs
  carry a frame timing line every `FACESTREAM_STATS_INTERVAL` frames
  (`swap stats: 29.4 fps over 150 frames, 21.3 ms/frame, 2 repeated, 0 dropped`)
  — that is the number to watch when deciding whether a bigger GPU would help.
- **Check the licensing.** The inswapper model this depends on is licensed for
  **non-commercial use only** (see Credits). That applies to your deployment as
  much as to the original project, so it rules out most commercial production
  use regardless of how the infrastructure is set up.
- **Get consent for the faces you use.** Swapping a real person's likeness onto
  a live stream has legal exposure in many jurisdictions, and several platforms
  have their own rules about synthetic likenesses.

A reasonable production deploy ends up looking like:

```
FACESTREAM_GPU=L40S \
FACESTREAM_REGION=us-east-1 \
FACESTREAM_MIN_CONTAINERS=1 \
FACESTREAM_MAX_CONTAINERS=4 \
FACESTREAM_TURN=1 \
uv run modal deploy -m facestream.main
```

## Tests

```
uv run pytest -m "not browser"   # fast, no browser needed
uv run playwright install chromium
uv run pytest                    # everything, ~2 minutes
```

None of it needs a GPU or a Modal account. The browser tests drive the real
page against [tests/stub_server.py](tests/stub_server.py), which speaks the same
websocket protocol and runs the real aiortc peer connection and the real
`ProcessFrameTrack`, with a cheap CPU tint standing in for the GPU swap. If you
already have a Chromium build, point `FACESTREAM_CHROMIUM` at it instead of
letting Playwright download one.

## Credits

- This project was inspired by and uses the model of [Deep Live Cam](https://github.com/hacksider/Deep-Live-Cam). Go check out their project. They have some extra features that I didn't implement here. Note that the model used in that project (and therefore also this one) is only for non-commercial use.

- [Modal](https://modal.com) made this easy to build and deploy. They are generously providing free credits to host the live demo.
