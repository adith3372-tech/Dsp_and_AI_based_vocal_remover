/* =============================================================================
   app.js — Core Client Logic for Vocal Remover Web UI
   Supports: DSP mode (classical pipeline) + AI mode (Demucs 4-stem)
   ============================================================================= */

// ── Global mode state ──────────────────────────────────────────────────────
let currentMode = "dsp";

function setMode(mode) {
    currentMode = mode;
    document.getElementById("modeInput").value = mode;

    const btnDSP = document.getElementById("btnModeDSP");
    const btnAI  = document.getElementById("btnModeAI");
    const panelDSP = document.getElementById("panelDSP");
    const panelAI  = document.getElementById("panelAI");

    if (mode === "ai") {
        btnDSP.classList.remove("active");
        btnAI.classList.add("active", "active-ai");
        panelDSP.classList.remove("panel-visible");
        panelDSP.classList.add("panel-hidden");
        panelAI.classList.remove("panel-hidden");
        panelAI.classList.add("panel-visible");
    } else {
        btnAI.classList.remove("active", "active-ai");
        btnDSP.classList.add("active");
        panelAI.classList.remove("panel-visible");
        panelAI.classList.add("panel-hidden");
        panelDSP.classList.remove("panel-hidden");
        panelDSP.classList.add("panel-visible");
    }
}

document.addEventListener("DOMContentLoaded", () => {

    // ── Sliders Value Sync ────────────────────────────────────────────────
    [
        { id: "nlms_mu",    valId: "muVal"    },
        { id: "stft_blend", valId: "stftVal"  },
        { id: "nlms_blend", valId: "nlmsVal"  },
        { id: "ai_shifts",  valId: "shiftsVal" },
    ].forEach(s => {
        const el = document.getElementById(s.id);
        const vl = document.getElementById(s.valId);
        if (el && vl) el.addEventListener("input", e => { vl.textContent = e.target.value; });
    });

    // ── Dropzone & File Input ─────────────────────────────────────────────
    const dropzone  = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");

    if (dropzone && fileInput) {
        dropzone.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) updateDropzoneText(fileInput.files[0].name);
        });
        ["dragenter", "dragover"].forEach(ev =>
            dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add("dragover"); }, false));
        ["dragleave", "drop"].forEach(ev =>
            dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove("dragover"); }, false));
        dropzone.addEventListener("drop", e => {
            const files = e.dataTransfer.files;
            if (files.length > 0) { fileInput.files = files; updateDropzoneText(files[0].name); }
        });
    }

    function updateDropzoneText(name) {
        const tc = dropzone.querySelector(".dropzone-text");
        if (tc) tc.innerHTML = `
            <div class="upload-icon" style="animation:none;transform:scale(1.1)">✓</div>
            <p class="main-msg" style="color:var(--green-accent)">File Selected</p>
            <p class="sub-msg" style="color:var(--text-primary)">${name}</p>`;
    }

    // ── UI element refs ───────────────────────────────────────────────────
    const form              = document.getElementById("uploadForm");
    const btnSubmit         = document.getElementById("btnSubmit");
    const btnDemo           = document.getElementById("btnDemo");
    const statusPlaceholder = document.getElementById("statusPlaceholder");
    const progressPanel     = document.getElementById("progressPanel");
    const resultsPanel      = document.getElementById("resultsPanel");
    const currentStage      = document.getElementById("currentStage");
    const percentVal        = document.getElementById("percentVal");
    const progressBarFill   = document.getElementById("progressBarFill");
    const processConsole    = document.getElementById("processConsole");
    const sdrVocals         = document.getElementById("sdrVocals");
    const sdrKaraoke        = document.getElementById("sdrKaraoke");
    const audioVocals       = document.getElementById("audioVocals");
    const audioKaraoke      = document.getElementById("audioKaraoke");
    const audioDrums        = document.getElementById("audioDrums");
    const audioBass         = document.getElementById("audioBass");
    const audioOther        = document.getElementById("audioOther");
    const resultsPlot       = document.getElementById("resultsPlot");
    const aiStemsBlock      = document.getElementById("aiStemsBlock");
    const resultModeTag     = document.getElementById("resultModeTag");

    let consoleHistory = [];

    function addConsoleLine(text, type = "") {
        const p = document.createElement("p");
        p.className = `console-line ${type}`;
        p.textContent = `> ${text}`;
        processConsole.appendChild(p);
        processConsole.scrollTop = processConsole.scrollHeight;
    }

    function resetUIState() {
        statusPlaceholder.classList.add("hidden");
        progressPanel.classList.remove("hidden");
        resultsPanel.classList.add("hidden");
        processConsole.innerHTML = "";
        consoleHistory = [];
        progressBarFill.style.width = "0%";
        percentVal.textContent = "0%";
        currentStage.textContent = "Initializing...";
    }

    // ── Form submit (file upload) ─────────────────────────────────────────
    if (form) {
        form.addEventListener("submit", e => {
            e.preventDefault();
            if (!fileInput.files || fileInput.files.length === 0) {
                alert("Please select or drop an audio file first.");
                return;
            }
            const fd = new FormData(form);
            fd.set("mode", currentMode);          // ensure mode is sent
            startJob(fd);
        });
    }

    // ── Demo button ───────────────────────────────────────────────────────
    if (btnDemo) {
        btnDemo.addEventListener("click", () => {
            const fd = new FormData();
            fd.append("demo",       "true");
            fd.append("mode",       currentMode);
            fd.append("decimation", document.getElementById("decimation")?.value || 2);
            fd.append("nlms_mu",    document.getElementById("nlms_mu")?.value    || 0.3);
            fd.append("stft_blend", document.getElementById("stft_blend")?.value || 0.6);
            fd.append("nlms_blend", document.getElementById("nlms_blend")?.value || 0.4);
            fd.append("ai_model",   document.getElementById("ai_model")?.value   || "htdemucs");
            fd.append("ai_shifts",  document.getElementById("ai_shifts")?.value  || 1);
            startJob(fd);
        });
    }

    // ── Start job ─────────────────────────────────────────────────────────
    function startJob(formData) {
        resetUIState();
        const modeLabel = currentMode === "ai" ? "🤖 AI (Demucs)" : "⚙️ Classical DSP";
        addConsoleLine(`Starting ${modeLabel} pipeline...`);
        addConsoleLine("Uploading audio file to backend server...");

        btnSubmit.disabled = true;
        btnDemo.disabled   = true;

        fetch("/upload", { method: "POST", body: formData })
            .then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.error || "Upload failed"); }))
            .then(data => {
                addConsoleLine(`Upload success. Job ID: ${data.job_id}`, "success");
                addConsoleLine("Pipeline worker started. Polling for updates...");
                pollJobStatus(data.job_id);
            })
            .catch(err => {
                addConsoleLine(`Error: ${err.message}`, "error");
                btnSubmit.disabled = false;
                btnDemo.disabled   = false;
            });
    }

    // ── Poll job status ───────────────────────────────────────────────────
    function pollJobStatus(jobId) {
        const interval = setInterval(() => {
            fetch(`/status/${jobId}`)
                .then(r => r.json())
                .then(job => {
                    currentStage.textContent      = job.stage;
                    percentVal.textContent        = `${job.percent}%`;
                    progressBarFill.style.width   = `${job.percent}%`;

                    const msg = `[${job.percent}%] ${job.stage}`;
                    if (!consoleHistory.includes(msg)) {
                        consoleHistory.push(msg);
                        addConsoleLine(msg);
                    }

                    if (job.status === "completed") {
                        clearInterval(interval);
                        const modeLabel = (job.mode === "ai") ? "AI (Demucs)" : "Classical DSP";
                        addConsoleLine(`${modeLabel} pipeline finished successfully! ✓`, "success");
                        displayResults(job);
                    } else if (job.status === "failed") {
                        clearInterval(interval);
                        addConsoleLine(`Process Failed: ${job.error}`, "error");
                        btnSubmit.disabled = false;
                        btnDemo.disabled   = false;
                    }
                })
                .catch(err => {
                    clearInterval(interval);
                    addConsoleLine(`Polling error: ${err.message}`, "error");
                    btnSubmit.disabled = false;
                    btnDemo.disabled   = false;
                });
        }, 1500);
    }

    // ── Display results ───────────────────────────────────────────────────
    function displayResults(job) {
        progressPanel.classList.add("hidden");
        resultsPanel.classList.remove("hidden");
        btnSubmit.disabled = false;
        btnDemo.disabled   = false;

        // Mode tag
        const isAI = job.mode === "ai";
        resultModeTag.textContent  = isAI ? "🤖 AI MODE — Demucs htdemucs" : "⚙️ DSP MODE — Classical Pipeline";
        resultModeTag.className    = `result-mode-tag ${isAI ? "tag-ai" : "tag-dsp"}`;

        // Metrics
        const vSDR = job.metrics?.blind_sdr_vocals;
        const kSDR = job.metrics?.blind_sdr_karaoke;
        sdrVocals.textContent  = vSDR !== undefined ? `${vSDR.toFixed(2)} dB` : "N/A";
        sdrKaraoke.textContent = kSDR !== undefined ? `${kSDR.toFixed(2)} dB` : "N/A";
        sdrVocals.style.color  = (vSDR >= 8.0) ? "var(--green-accent)" : "var(--orange-accent)";
        sdrKaraoke.style.color = (kSDR >= 8.0) ? "var(--green-accent)" : "var(--orange-accent)";

        // Main players
        audioVocals.src  = job.results.vocals;
        audioKaraoke.src = job.results.karaoke;
        audioVocals.load();
        audioKaraoke.load();

        // AI stem players
        if (isAI && job.results.drums) {
            aiStemsBlock.classList.remove("hidden");
            audioDrums.src = job.results.drums;
            audioBass.src  = job.results.bass;
            audioOther.src = job.results.other;
            audioDrums.load();
            audioBass.load();
            audioOther.load();
        } else {
            aiStemsBlock.classList.add("hidden");
        }

        // Plot
        resultsPlot.src = job.results.plot;
    }
});
