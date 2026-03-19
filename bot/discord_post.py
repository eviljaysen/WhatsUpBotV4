"""bot/discord_post.py — Discord webhook posting (v4.0).

Posts scan reports directly to a Discord channel via webhook.
Uses only stdlib (urllib.request) — no extra dependencies.

Configure: set "discord_webhook_url" in config.json.
If the URL is empty, all post functions return False silently.
"""

import json
import urllib.request
import urllib.error
import os

from bot.config import DISCORD_WEBHOOK_URL, get_logger

_log = get_logger("discord_post")


def post_report(report_text: str, image_path: str = None,
                webhook_url: str = None) -> bool:
    """Post a scan report to Discord.

    Args:
        report_text: the full report string (Discord markdown)
        image_path:  optional path to an image file to attach
        webhook_url: override config URL (useful for testing)

    Returns:
        True on success (HTTP 2xx), False on failure or if URL not configured.
    """
    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        return False

    if image_path and os.path.isfile(image_path):
        return _post_multipart(url, report_text, image_path)

    # Discord content limit is 2000 chars — chunk at line boundaries if needed
    chunks = _chunk_text(report_text, 2000)
    return all(_post_json(url, {"content": chunk}) for chunk in chunks)


def post_alert(message: str, webhook_url: str = None) -> bool:
    """Post a short alert message to Discord.

    Args:
        message: plain text or Discord markdown alert string
        webhook_url: override config URL

    Returns:
        True on success, False on failure or unconfigured.
    """
    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        return False

    return _post_json(url, {"content": message})


def _chunk_text(text: str, limit: int) -> list:
    """Split text into chunks ≤ limit chars, breaking at line boundaries."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], []
    length = 0
    for line in text.splitlines(keepends=True):
        if length + len(line) > limit and current:
            chunks.append("".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _post_multipart(url: str, text: str, image_path: str) -> bool:
    """POST text + image file as multipart/form-data to a Discord webhook."""
    import uuid
    boundary = uuid.uuid4().hex
    payload_json = json.dumps({"content": text}).encode("utf-8")
    filename = os.path.basename(image_path)

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
    except OSError as e:
        _log.error("Cannot read image %s: %s", image_path, e)
        return False

    def _part(name, data, content_type, fname=None):
        cd = f'Content-Disposition: form-data; name="{name}"'
        if fname:
            cd += f'; filename="{fname}"'
        header = f"--{boundary}\r\n{cd}\r\nContent-Type: {content_type}\r\n\r\n"
        return header.encode() + data + b"\r\n"

    body = (
        _part("payload_json", payload_json, "application/json")
        + _part("file", image_data, "image/png", filename)
        + f"--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        url, data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": _UA,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        _log.error("HTTP %s: %s", e.code, e.reason)
        return False
    except Exception as e:
        _log.error("Multipart post failed: %s", e, exc_info=True)
        return False


_UA = "WhatsUpBot/5.0"


def _post_json(url: str, payload: dict) -> bool:
    """POST a JSON payload to url. Returns True on HTTP 2xx."""
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": _UA,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        _log.error("HTTP %s: %s", e.code, e.reason)
        return False
    except Exception as e:
        _log.error("Post failed: %s", e, exc_info=True)
        return False
