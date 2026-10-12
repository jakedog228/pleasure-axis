"""Live steering demo: a local web app that runs Qwen2.5-3B-Instruct and steers it in real time with the study's vectors.

    python demo/live/server.py                 # GPU if available, opens http://127.0.0.1:8765
    python demo/live/server.py --device cpu    # start on the CPU instead (also switchable in the page)

Options: --model <results folder name> (any model with results/<name>/analysis/vectors.npz; default
Qwen_2.5_3B_instruct), --port, --cpu-threads (default 8, so the rest of the machine stays responsive), --no-browser.
One page, one generation at a time; the server binds to localhost only.
"""
import argparse, asyncio, json, sys, threading, webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import MAX_LEN, ROOT, Engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen_2.5_3B_instruct")
ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--cpu-threads", type=int, default=8)
ap.add_argument("--no-browser", action="store_true")
args = ap.parse_args()

@asynccontextmanager
async def lifespan(app):
    S["loop"] = asyncio.get_running_loop()
    threading.Thread(target=load_engine, args=(args.device,), daemon=True).start()
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    yield


app = FastAPI(lifespan=lifespan)
S = {"engine": None, "loading": "starting", "error": None, "busy": False, "clients": set(), "loop": None}


def research_meta(eng):
    """Per-model numbers from the study's results for the page: where each variety sits on the sign and significance
    axes, and how well the varieties decode."""
    A = ROOT / "results" / eng.name / "analysis"
    out = {}
    p = A / "varieties_proj_plL.csv"
    if p.exists():
        vp = pd.read_csv(p)
        out["variety_map"] = [{"key": r.variety.replace("V_", ""), "sign": round(float(r.sign_axis), 3),
                               "shared": round(float(r.shared_axis), 3), "pleasure": round(float(r.pleasure_axis), 3)}
                              for r in vp.itertuples()]
    p = A / "varieties_cls.json"
    if p.exists():
        c = json.load(open(p))
        out["variety_decoding"] = {"accuracy": c["accuracy_logreg"], "chance": c["chance"]}
    s = json.load(open(A / "summary.json"))
    out["auc"] = {"pleasure": s["pleasure_kfold_auc_S2_at_best_final"], "pain": s["pain_kfold_auc_S2_at_best_final"]}
    return out


def load_engine(device):
    S["loading"] = f"loading {args.model} on {device or 'the default device'}"
    S["error"] = None
    broadcast_status()
    try:
        if S["engine"] is None:
            S["engine"] = Engine(args.model, device=device, cpu_threads=args.cpu_threads, log=lambda m: print(m, flush=True))
        else:
            S["engine"].load(device)
        S["loading"] = None
    except Exception as ex:                       # surface it in the page (e.g. not enough memory)
        S["loading"] = None
        err = f"{type(ex).__name__}: {ex}"
        print("load failed:", err, flush=True)
        if S["engine"] is not None and S["engine"].model is not None:
            broadcast({"type": "error", "error": f"Could not switch: {ex}"})      # still running on the old device
        else:
            S["error"] = err
    broadcast_status()
    if S["engine"] is not None and S["engine"].model is not None:
        broadcast({"type": "meta", **S["engine"].meta(), **research_meta(S["engine"])})


def status_msg():
    e = S["engine"]
    st = {"type": "status", "loading": S["loading"], "error": S["error"], "busy": S["busy"]}
    if e is not None and e.model is not None and not S["loading"]:
        st.update(e.status())
    return st


def broadcast(msg):
    loop = S["loop"]
    if loop is None:
        return
    data = json.dumps(msg, ensure_ascii=False)
    for ws in list(S["clients"]):
        asyncio.run_coroutine_threadsafe(_send(ws, data), loop)


async def _send(ws, data):
    try:
        await ws.send_text(data)
    except Exception:
        S["clients"].discard(ws)


def broadcast_status():
    broadcast(status_msg())


def build_ids(eng, req):
    """Token ids for a request. Chat: oldest turns are dropped until prompt + reply fit the cache."""
    if req.get("mode") == "raw":
        return eng.raw_ids(req["text"]), 0
    msgs = [m for m in req["messages"] if m.get("content")]
    system = req.get("system")
    budget = MAX_LEN - int(req.get("max_new", 160)) - 4
    dropped = 0
    while True:
        full = ([{"role": "system", "content": system}] if system else []) + msgs
        ids = eng.chat_ids(full)
        if ids.shape[1] <= budget or len(msgs) <= 1:
            return ids, dropped
        msgs = msgs[2:] if len(msgs) > 2 else msgs[1:]
        dropped += 1


def run_generation(req):
    eng = S["engine"]
    try:
        ids, dropped = build_ids(eng, req)
        broadcast({"type": "start", "id": req.get("id"), "prompt_tokens": int(ids.shape[1]), "dropped_turns": dropped})
        res = eng.generate(ids, lambda m: broadcast({**m, "id": req.get("id")}), max_new=int(req.get("max_new", 160)),
                           temperature=float(req.get("temperature", 0)), seed=req.get("seed"))
        broadcast({"type": "done", "id": req.get("id"), **res})
    except Exception as ex:
        broadcast({"type": "error", "id": req.get("id"), "error": f"{type(ex).__name__}: {ex}"})
    finally:
        S["busy"] = False
        broadcast_status()


@app.get("/")
async def index():
    return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    S["loop"] = asyncio.get_running_loop()
    S["clients"].add(ws)
    await ws.send_text(json.dumps(status_msg()))
    e = S["engine"]
    if e is not None and e.model is not None and not S["loading"]:
        await ws.send_text(json.dumps({"type": "meta", **e.meta(), **research_meta(e)}, ensure_ascii=False))
    try:
        while True:
            req = json.loads(await ws.receive_text())
            t = req.get("type")
            e = S["engine"]
            ready = e is not None and e.model is not None and not S["loading"]
            if t == "steer" and e is not None:
                sv = e.set_steer(req.get("mix", {}), req.get("layer", e.L_chat), req.get("enabled", True))
                hn = e.resid_norms.get(e.layer) if e.resid_norms else None
                await ws.send_text(json.dumps({"type": "steer_ack", "sv": sv, "mix": e.mix, "layer": e.layer,
                                               "enabled": e.enabled, "norm": e.vec_norm(),
                                               "ratio_est": (e.vec_norm() / hn) if hn else None}))
            elif t == "generate":
                if not ready or S["busy"]:
                    await ws.send_text(json.dumps({"type": "error", "id": req.get("id"),
                                                   "error": "busy" if S["busy"] else "model not loaded"}))
                    continue
                S["busy"] = True
                threading.Thread(target=run_generation, args=(req,), daemon=True).start()
            elif t == "stop" and e is not None:
                e.stop_flag.set()
            elif t == "device":
                if S["busy"] or S["loading"]:
                    if e is not None:
                        e.stop_flag.set()
                    await ws.send_text(json.dumps({"type": "error", "error": "wait for the current reply to finish"}))
                    continue
                threading.Thread(target=load_engine, args=(req["device"],), daemon=True).start()
            elif t == "status":
                await ws.send_text(json.dumps(status_msg()))
    except WebSocketDisconnect:
        pass
    finally:
        S["clients"].discard(ws)


if __name__ == "__main__":
    print(f"Live steering demo on http://127.0.0.1:{args.port}  (Ctrl+C to quit)", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
