import asyncio
import logging

import aiohttp
from aiohttp import web
from yarl import URL

from .endpoint import SERVED_PREFIX, not_serving

CHUNK = 1 << 20
UP_HEADERS = ("Content-Type", "Content-Length", "X-Enc-Auth")
DOWN_HEADERS = ("Content-Type", "Content-Length", "X-Encode-Ms", "X-Serialize-Ms")
READ_TIMEOUT_S = 600
OFF = ("this ComfyUI does not serve text encoders: turn on Settings > Remote Text Encoder > Serve text encoders to "
       "other PCs there (or start it with REMOTE_TE_SERVE=1)")
LOCAL_ONLY = ("keep", "restart", "stop")
REFUSED = ("/{name} changes this PC's resources and is not open to other PCs. Change the kept models in this PC's "
           "own ComfyUI (Remote Encoder Status with server_url local); start or stop the encoder server in its "
           "Settings > Remote Text Encoder.")


def local_only(path):
    name = path.strip("/").lower()
    return name if name in LOCAL_ONLY else None


def _error(status, message):
    return web.json_response({"error": message}, status=status)


def _down(child):
    last = "\n".join(list(child.tail)[-8:])
    if child.state == "failed":
        return _error(503, f"the encoder server on this ComfyUI stopped and was not restarted. Its last output:\n"
                           f"{last}")
    return _error(503, f"the encoder server on this ComfyUI is starting or restarting; try again in a moment. "
                       f"Its last output:\n{last}")


def make_handler(get_child):
    session = None

    async def forward(request):
        nonlocal session
        tail = URL(f"http://127.0.0.1/{request.match_info['tail']}")
        name = local_only(tail.path)
        if name:
            return _error(403, REFUSED.format(name=name))
        child = get_child()
        if child is None:
            return _error(503, not_serving(OFF))
        if child.state == "failed" or child.port is None:
            return _down(child)
        if session is None or session.closed:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=5, sock_read=READ_TIMEOUT_S),
                auto_decompress=False)
        url = tail.with_port(child.port)
        headers = {h: request.headers[h] for h in UP_HEADERS if h in request.headers}
        data = request.content if request.body_exists else None
        try:
            async with session.request(request.method, url, params=request.query, headers=headers, data=data,
                                       allow_redirects=False) as up:
                reply = web.StreamResponse(status=up.status)
                for h in DOWN_HEADERS:
                    if h in up.headers:
                        reply.headers[h] = up.headers[h]
                await reply.prepare(request)
                try:
                    async for chunk in up.content.iter_chunked(CHUNK):
                        await reply.write(chunk)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logging.warning(f"[remote_te server] reply from the encoder server cut off: {e}")
                    if request.transport is not None:
                        request.transport.close()
                    return reply
                await reply.write_eof()
                return reply
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
            return _down(child)

    async def close():
        if session is not None and not session.closed:
            await session.close()

    forward.close = close
    return forward


def register(prompt_server, get_child):
    handler = make_handler(get_child)
    prompt_server.routes.route("*", SERVED_PREFIX + "/{tail:.*}")(handler)

    async def close_session(app):
        await handler.close()
    prompt_server.app.on_cleanup.append(close_session)
