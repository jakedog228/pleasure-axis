"""Shell client for tools/colab_bridge.py (pleasure project; adapted from D:/Projects/WEAR/tools/colab_call.py).

python tools/colab_call.py status
python tools/colab_call.py tools                         # names + input schemas
python tools/colab_call.py call <tool> '<json args>' [--timeout S]
python tools/colab_call.py call <tool> @args.json        # arguments from a file
python tools/colab_call.py run "<python>" [--timeout S] [--max N] [--keep]
python tools/colab_call.py runfile <file.py> [--timeout S] [--max N] [--keep]
python tools/colab_call.py push <local_file> <remote_path>      # chunked base64+gzip, md5-verified
python tools/colab_call.py pull <remote_path> <local_file>      # chunked base64+gzip, md5-verified

run/runfile append a cell at the end, execute it, print its flattened output, then delete the cell
(unless --keep), so the scratch notebook does not grow without bound.
"""
import base64, gzip, hashlib, io, json, re, sys, urllib.error, urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
CFG = json.loads(Path(r"D:\Projects\pleasure\work", os.environ.get("BRIDGE", "colab_bridge") + ".json").read_text())
BASE = f"http://127.0.0.1:{CFG['ctl_port']}"
HDR = {"X-Bridge-Key": CFG.get("ctl_key", "")}
CHUNK = 4 * 2 ** 20          # raw bytes per push/pull cell


def _open(req, timeout):
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:      # bridge-side failure: print its message (tab disconnected, timeout, ...)
        sys.exit(f"bridge error HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:2000]}")
    except urllib.error.URLError as e:
        sys.exit(f"bridge not reachable ({e.reason}): start tools/colab_bridge.py (see COLAB.md)")


def get(path):
    return _open(urllib.request.Request(BASE + path, headers=HDR), 30)


def post(path, obj, timeout):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json", **HDR})
    return _open(req, timeout + 15)


def tool(name, arguments, timeout=60):
    return post("/call", {"name": name, "arguments": arguments, "timeout": timeout}, timeout)


def _text(r):
    return "".join(c.get("text", "") for c in r.get("result", {}).get("content", []))


def flatten(res):
    """Notebook outputs -> plain text (streams, results, errors with traceback)."""
    out = []
    for c in (res.get("content") or []) if isinstance(res, dict) else []:
        t = c["text"] if c.get("type") == "text" else json.dumps(c)
        try:
            o = json.loads(t)
            parts = []
            for x in o.get("outputs", []):
                ot = x.get("output_type")
                if ot == "stream":
                    parts.append("".join(x["text"]) if isinstance(x.get("text"), list) else str(x.get("text", "")))
                elif ot in ("execute_result", "display_data"):
                    d = x.get("data", {})
                    v = d.get("text/plain", d)
                    parts.append(("".join(v) if isinstance(v, list) else str(v)) + "\n")
                elif ot == "error":
                    tb = "\n".join(x.get("traceback", []))
                    parts.append(f"ERROR {x.get('ename')}: {x.get('evalue')}\n{tb}\n")
                else:
                    parts.append(str(x.get("text", x.get("data", x.get("evalue", "")))))
            t = "".join(parts)
        except Exception:
            pass
        out.append(t)
    return re.sub(r"\x1b\[[0-9;]*m", "", "".join(out))


def run_cell(code, timeout=600, keep=False):
    # insert at the top and use the id add_code_cell returns: no index lookup, so concurrent callers cannot
    # pick up each other's cells
    cid = json.loads(_text(tool("add_code_cell", {"cellIndex": 0, "language": "python", "code": code})))["newCellId"]
    r = post("/call", {"name": "run_code_cell", "arguments": {"cellId": cid}, "timeout": timeout}, timeout)
    res = r.get("result", r)
    text = flatten(res) if isinstance(res, dict) and res.get("content") else json.dumps(r, indent=1)[:20000]
    if not keep:
        try:
            tool("delete_cell", {"cellId": cid})
        except SystemExit:
            pass
    return cid, text


def arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


cmd = sys.argv[1]
if cmd == "status":
    print(json.dumps(get("/status"), indent=1))
elif cmd == "tools":
    for t in get("/tools"):
        print(f"## {t['name']}\n{t.get('description', '')[:400]}\nschema: {json.dumps(t.get('inputSchema', {}))[:800]}\n")
elif cmd == "call":
    name, a = sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "{}"
    timeout = arg("--timeout", 600)
    arguments = json.loads(Path(a[1:]).read_text(encoding="utf-8") if a.startswith("@") else a)
    r = post("/call", {"name": name, "arguments": arguments, "timeout": timeout}, timeout)
    res = r.get("result", r)
    for c in (res.get("content") or []) if isinstance(res, dict) else []:
        print(c["text"] if c.get("type") == "text" else json.dumps(c)[:2000])
    if not (isinstance(res, dict) and res.get("content")):
        print(json.dumps(r, indent=1)[:20000])
elif cmd in ("run", "runfile"):
    code = Path(sys.argv[2]).read_text(encoding="utf-8") if cmd == "runfile" else sys.argv[2]
    cid, t = run_cell(code, arg("--timeout", 600), keep="--keep" in sys.argv)
    LIM = arg("--max", 4000)
    print(f"[cell {cid}]")
    cut = "\n... [%d chars cut] ...\n" % (len(t) - LIM)
    print(t if len(t) <= LIM else t[:LIM // 4] + cut + t[-3 * LIM // 4:])
elif cmd == "push":
    src, dst = Path(sys.argv[2]), sys.argv[3]
    data = src.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)] or [b""]
    for i, ch in enumerate(chunks):
        b64 = base64.b64encode(gzip.compress(ch)).decode()
        mode = "wb" if i == 0 else "ab"
        code = (f"import base64,gzip,os\nos.makedirs(os.path.dirname({dst!r}) or '.',exist_ok=True)\n"
                f"open({dst!r},{mode!r}).write(gzip.decompress(base64.b64decode('{b64}')))\nprint('ok')")
        _, t = run_cell(code, 300)
        if "ok" not in t:
            sys.exit(f"push chunk {i} failed: {t[:2000]}")
    _, t = run_cell(f"import hashlib;print(hashlib.md5(open({dst!r},'rb').read()).hexdigest())", 120)
    print(f"pushed {src} -> {dst} ({len(data)} bytes, {len(chunks)} chunk(s)); md5 local {md5} remote {t.strip()}")
    if t.strip() != md5:
        sys.exit("MD5 MISMATCH")
elif cmd == "pull":
    src, dst = sys.argv[2], Path(sys.argv[3])
    _, t = run_cell(f"import os,hashlib;b=open({src!r},'rb').read();print(len(b),hashlib.md5(b).hexdigest())", 120)
    try:
        size, md5 = t.split()[:2]; size = int(size)
    except Exception:
        sys.exit(f"pull: cannot stat {src}: {t[:2000]}")
    buf = bytearray()
    for off in range(0, max(size, 1), CHUNK):
        code = (f"import base64,gzip\nf=open({src!r},'rb');f.seek({off});"
                f"print('B64:'+base64.b64encode(gzip.compress(f.read({CHUNK}))).decode()+':END')")
        _, t = run_cell(code, 300)
        if "B64:" not in t or ":END" not in t:
            sys.exit(f"pull chunk at {off} failed: {t[:2000]}")
        buf += gzip.decompress(base64.b64decode(t.split("B64:", 1)[1].split(":END", 1)[0]))
        if size == 0:
            break
    got = hashlib.md5(bytes(buf)).hexdigest()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(bytes(buf))
    print(f"pulled {src} -> {dst} ({len(buf)} bytes); md5 remote {md5} local {got}")
    if got != md5:
        sys.exit("MD5 MISMATCH")
else:
    sys.exit(__doc__)
