"""
=============================================================================
app.py — Flask Backend Server for Vocal Remover Web UI
=============================================================================
"""

import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Import both pipeline runners
from main import run_pipeline, run_ai_pipeline

app = Flask(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB

JOBS      = {}
JOBS_LOCK = threading.Lock()


class SimpleNamespace:
    """Lightweight struct to pass args to pipeline functions."""
    def __init__(self, **kw): self.__dict__.update(kw)


def async_pipeline_worker(job_id, input_path, output_dir, opts):
    """Background thread: runs DSP or AI pipeline, updates JOBS dict."""
    mode = opts.get("mode", "dsp")
    try:
        args = SimpleNamespace(
            input      = input_path,
            output_dir = output_dir,
            demo       = opts.get("demo", False),
            plot       = True,
            # DSP params
            decimation = int(opts.get("decimation", 2)),
            n_fft      = int(opts.get("n_fft", 2048)),
            hop_length = int(opts.get("hop_length", 512)),
            nlms_order = int(opts.get("nlms_order", 256)),
            nlms_mu    = float(opts.get("nlms_mu", 0.3)),
            stft_blend = float(opts.get("stft_blend", 0.6)),
            nlms_blend = float(opts.get("nlms_blend", 0.4)),
            # AI params
            ai_model   = str(opts.get("ai_model", "htdemucs")),
            ai_shifts  = int(opts.get("ai_shifts", 1)),
            ai_segment = float(opts.get("ai_segment", 5.0)),
        )

        def progress_cb(stage, pct):
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["stage"]   = stage
                    JOBS[job_id]["percent"] = pct

        results   = run_ai_pipeline(args) if mode == "ai" else run_pipeline(args, progress_cb=progress_cb)
        plot_file = "ai_pipeline_results.png" if mode == "ai" else "pipeline_results.png"
        cb        = uuid.uuid4().hex[:6]

        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update({
                    "status":  "completed",
                    "percent": 100,
                    "stage":   "Process Complete",
                    "metrics": results.get("metrics", {}),
                    "mode":    mode,
                    "results": {
                        "vocals":  f"/outputs/{job_id}/vocals.wav?v={cb}",
                        "karaoke": f"/outputs/{job_id}/karaoke.wav?v={cb}",
                        "plot":    f"/outputs/{job_id}/{plot_file}?v={cb}",
                        "drums":   f"/outputs/{job_id}/drums.wav?v={cb}" if mode == "ai" else None,
                        "bass":    f"/outputs/{job_id}/bass.wav?v={cb}"  if mode == "ai" else None,
                        "other":   f"/outputs/{job_id}/other.wav?v={cb}" if mode == "ai" else None,
                    }
                })
    except Exception as e:
        import traceback; traceback.print_exc()
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update({"status": "failed", "stage": "Error occurred", "error": str(e)})


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files and "demo" not in request.form:
        return jsonify({"error": "No file uploaded"}), 400

    job_id         = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_FOLDER, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    opts = {
        "mode":       request.form.get("mode", "dsp"),
        "decimation": request.form.get("decimation", 2),
        "nlms_mu":    request.form.get("nlms_mu", 0.3),
        "stft_blend": request.form.get("stft_blend", 0.6),
        "nlms_blend": request.form.get("nlms_blend", 0.4),
        "ai_model":   request.form.get("ai_model", "htdemucs"),
        "ai_shifts":  request.form.get("ai_shifts", 1),
    }

    if "demo" in request.form and request.form["demo"] == "true":
        opts["demo"] = True
        input_path   = ""
        init_stage   = "Generating synthetic test track..."
        filename     = "demo_synthetic_mix.wav"
    else:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        filename   = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_{filename}")
        file.save(input_path)
        init_stage = "Initializing separation..."

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "stage": init_stage,
                        "percent": 0, "filename": filename,
                        "mode": opts["mode"], "error": None}

    threading.Thread(target=async_pipeline_worker,
                     args=(job_id, input_path, job_output_dir, opts),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def get_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job)


@app.route("/outputs/<job_id>/<filename>")
def serve_output_file(job_id, filename):
    job_dir = os.path.join(OUTPUT_FOLDER, secure_filename(job_id))
    return send_from_directory(job_dir, secure_filename(filename))


if __name__ == "__main__":
    print("Starting Vocal Remover Server at http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
