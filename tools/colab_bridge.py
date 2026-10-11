"""Minimal local bridge to a Colab notebook tab (same wire protocol as googlecolab/colab-mcp), driven from the shell.

The Colab frontend connects to a local websocket server (origin colab.research.google.com, subprotocol "mcp",
?access_token=<token>) and then acts as an MCP *server* over that socket; this process is the MCP *client*.
A tiny HTTP control server on localhost lets shell commands list / call the notebook tools (tools/colab_call.py).

usage:  python tools/colab_bridge.py [--ws-port 8781] [--ctl-port 8782] [--no-open]
State (token, ports, Colab URL) is written to work/colab_bridge.json; log to work/colab_bridge.log.
"""
import argparse, asyncio, json, logging, os, secrets, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

p = argparse.ArgumentParser()
p.add_argument("--ws-port", type=int, default=8781)
p.add_argument("--ctl-port", type=int, default=8782)
p.add_argument("--no-open", action="store_true")
p.add_argument("--reuse", action="store_true", help="keep the WebSocket token of the state file (tab may reattach without reload)")
p.add_argument("--notebook", default="https://colab.research.google.com/notebooks/empty.ipynb",
               help="notebook URL to attach (a second runtime needs a different notebook)")
p.add_argument("--state", default="colab_bridge", help="state/log file stem in work/ (a second bridge: colab_bridge2)")
args = p.parse_args()
WORK = Path(r"D:\Projects\pleasure\work")
logging.basicConfig(filename=WORK / f"{args.state}.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bridge")
TOKEN = secrets.token_urlsafe(16)
if args.reuse and (WORK / f"{args.state}.json").exists():
    _old = json.loads((WORK / f"{args.state}.json").read_text())
    _tok = _old.get("token") or _old.get("url", "").split("mcpProxyToken=")[-1].split("&")[0]
    if _old.get("ws_port") == args.ws_port and _tok:
        TOKEN = _tok
CTL_KEY = secrets.token_urlsafe(24)       # required as X-Bridge-Key on every control request (control port is otherwise open)
URL = f"{args.notebook}#mcpProxyToken={TOKEN}&mcpProxyPort={args.ws_port}"
ORIGINS = ["https://colab.research.google.com", "https://colab.google.com"]


class Session:
    def __init__(self):
        self.ws = None
        self.next_id = 1
        self.pending = {}
        self.tools = []
        self.ready = asyncio.Event()
        self.server_info = None

    async def request(self, method, params=None, timeout=600):
        if self.ws is None:
            raise RuntimeError("no Colab tab connected")
        rid = self.next_id; self.next_id += 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}))
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self.pending.pop(rid, None)

    async def notify(self, method, params=None):
        await self.ws.send(json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}}))

    async def reader(self, ws):
        async for raw in ws:
            msg = json.loads(raw)
            if "id" in msg and ("result" in msg or "error" in msg):
                fut = self.pending.get(msg["id"])
                if fut and not fut.done():
                    fut.set_result(msg)
            elif "method" in msg and "id" in msg:            # request from the frontend
                if msg["method"] == "ping":
                    await ws.send(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {}}))
                else:
                    log.info("frontend request %s", msg["method"])
                    await ws.send(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                                              "error": {"code": -32601, "message": "method not found"}}))
            else:
                log.info("frontend notification %s", str(msg)[:300])
                if msg.get("method") == "notifications/tools/list_changed":
                    asyncio.create_task(self.refresh_tools())

    async def refresh_tools(self):
        r = await self.request("tools/list")
        self.tools = r.get("result", {}).get("tools", [])
        log.info("tools: %s", [t["name"] for t in self.tools])


S = Session()


def check_token(conn, request):
    if f"access_token={TOKEN}" in request.path:
        return None
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer ") and auth.split(None, 1)[1] == TOKEN:
        return None
    return Response(403, "Bad authorization token", Headers([]))


async def handler(ws):
    if S.ws is not None:
        await ws.close(code=1013, reason="Server is busy")
        return
    S.ws = ws
    log.info("Colab tab connected from %s", ws.remote_address)
    rd = asyncio.create_task(S.reader(ws))
    try:
        r = await S.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "pleasure-colab-bridge", "version": "0.1"}}, timeout=60)
        S.server_info = r.get("result")
        log.info("initialize -> %s", json.dumps(r)[:500])
        await S.notify("notifications/initialized")
        await S.refresh_tools()
        S.ready.set()
        await rd
    except Exception as e:  # noqa
        log.exception("session error: %s", e)
    finally:
        rd.cancel()
        S.ws = None; S.ready.clear()
        for f in S.pending.values():
            if not f.done():
                f.set_exception(RuntimeError("Colab tab disconnected"))
        log.info("Colab tab disconnected")


LOOP = None


class Ctl(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _authorized(self, post=False):
        if self.headers.get("X-Bridge-Key") != CTL_KEY:
            self._send(403, {"error": "missing or bad X-Bridge-Key"}); return False
        if post and not (self.headers.get("Content-Type") or "").startswith("application/json"):
            self._send(415, {"error": "Content-Type must be application/json"}); return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        if self.path == "/status":
            self._send(200, {"connected": S.ws is not None, "ready": S.ready.is_set(), "server": S.server_info,
                             "n_tools": len(S.tools)})
        elif self.path == "/tools":
            self._send(200, S.tools)
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self):
        if not self._authorized(post=True):
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            if self.path == "/call":
                fut = asyncio.run_coroutine_threadsafe(
                    S.request("tools/call", {"name": body["name"], "arguments": body.get("arguments", {})},
                              timeout=body.get("timeout", 600)), LOOP)
                self._send(200, fut.result(body.get("timeout", 600) + 5))
            elif self.path == "/rpc":
                fut = asyncio.run_coroutine_threadsafe(S.request(body["method"], body.get("params")), LOOP)
                self._send(200, fut.result(605))
            else:
                self._send(404, {"error": "unknown path"})
        except Exception as e:  # noqa
            self._send(500, {"error": repr(e)})


async def main():
    global LOOP
    LOOP = asyncio.get_running_loop()
    async with websockets.serve(handler, "localhost", args.ws_port, subprotocols=["mcp"], origins=ORIGINS,
                                process_request=check_token, max_size=64 * 2 ** 20):
        srv = ThreadingHTTPServer(("127.0.0.1", args.ctl_port), Ctl)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        (WORK / f"{args.state}.json").write_text(json.dumps({"ws_port": args.ws_port, "ctl_port": args.ctl_port, "url": URL,
                                                            "token": TOKEN, "ctl_key": CTL_KEY, "pid": os.getpid()}))
        log.info("listening ws://localhost:%d (control http://127.0.0.1:%d)", args.ws_port, args.ctl_port)
        print("bridge ready; control port", args.ctl_port, flush=True)
        if not args.no_open:
            webbrowser.open_new(URL)
        await asyncio.Future()


asyncio.run(main())
