# FaceStream

This is a web app that let's you swap your face in real-time without having a powerful
machine. You can try it out live [here](https://facestream.phileisen.com).

See the gif below for a demo:

[![Demo](assets/demo.gif)](https://facestream.phileisen.com)

## Motivation

I discovered the [Deep Live Cam](https://github.com/hacksider/Deep-Live-Cam) project and tried to run it on my M1 MacBook, but only got 0.5 FPS. So I wanted to explore how fast and at what latency you could get this to run on a remote server via WebRTC.

## How to run this yourself

### Prerequisites

To run this yourself you wil need:

- [uv](https://docs.astral.sh/uv/)
- A [modal](https://modal.com/) account as well configured credentials

Then you can run the following command to start the server:

```
uv run modal serve facestream.main
```

Or this command to deploy it:

```
uv run modal deploy facestream.main
```

### Optional: TURN Server for Cellular Networks

On most cellular networks you need a TURN server for WebRTC to work. You can create a TURN app on [Cloudflare](https://developers.cloudflare.com/calls/turn/). To use those with this app you need to:

1. Create a secret called `facestream` in modal with the following values:

```
TURN_TOKEN_ID=your-turn-token-id
TURN_API_TOKEN=your-turn-api-token
```

2.  Comment out the following line in [src/facestream/main.py](src/facestream/main.py):

```
...
secrets=[
    modal.Secret.from_name(
        "facestream",
        required_keys=[SECRET_KEY_TURN_TOKEN_ID, SECRET_KEY_TURN_API_TOKEN],
    )
]
...
```

## Credits

- This project was inspired by and uses the model of [Deep Live Cam](https://github.com/hacksider/Deep-Live-Cam). Go check out their project. They have some extra features that I didn't implement here. Note that the model used in that project (and therefore also this one) is only for non-commercial use.

- [Modal](https://modal.com) made this easy to build and deploy. They are generously providing free credits to host the live demo.
