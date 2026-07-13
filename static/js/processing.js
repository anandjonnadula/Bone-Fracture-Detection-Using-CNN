/* Live progress stepper: polls /api/jobs/<id> every second and walks the
   pipeline steps. Gives up after ~3 minutes with a retry prompt. */

(function () {
    "use strict";

    const root = document.getElementById("processing");
    if (!root) return;

    const apiUrl = root.dataset.api;
    const uploadUrl = root.dataset.uploadUrl;

    // status -> index of the step it lights up (steps listed in the template)
    const STEP_OF = {
        queued: 0,
        preprocessing: 0,
        ood_check: 1,
        stage1: 2,
        stage2: 3,
        localizing: 3,
        explaining: 4,
        reporting: 5,
        done: 6,
    };

    const steps = Array.from(root.querySelectorAll(".stepper li"));
    const messageEl = document.getElementById("processingMessage");
    const spinnerTpl = '<div class="spinner"></div>';

    function render(activeIdx, failedIdx) {
        steps.forEach((li, i) => {
            li.classList.remove("active", "done", "failed");
            const dot = li.querySelector(".step-dot");
            if (failedIdx !== null && i === failedIdx) {
                li.classList.add("failed");
                dot.textContent = "!";
            } else if (i < activeIdx) {
                li.classList.add("done");
                dot.textContent = "✓";
            } else if (i === activeIdx && failedIdx === null) {
                li.classList.add("active");
                dot.innerHTML = spinnerTpl;
            } else {
                dot.textContent = i + 1;
            }
        });
    }

    function showMessage(html, kind) {
        messageEl.innerHTML = html;
        messageEl.className = "alert " + (kind === "error" ? "alert-danger" : "alert-info");
        messageEl.classList.remove("hidden");
    }

    const started = Date.now();
    const TIMEOUT_MS = 3 * 60 * 1000;
    let lastIdx = 0;

    async function poll() {
        if (Date.now() - started > TIMEOUT_MS) {
            render(lastIdx, lastIdx);
            showMessage(
                'This is taking longer than expected. ' +
                '<a href="' + uploadUrl + '">Try uploading again</a> or refresh this page.',
                "error");
            return;
        }
        try {
            const resp = await fetch(apiUrl, { headers: { "Accept": "application/json" } });
            if (resp.status === 404) throw new Error("Job not found");
            const job = await resp.json();

            if (job.status === "done") {
                render(steps.length, null);
                window.location.href = job.redirect;
                return;
            }
            if (job.status === "rejected") {
                render(1, 1);
                showMessage(
                    (job.message || "The image was rejected.") +
                    ' <a href="' + uploadUrl + '">Upload a different image</a>.',
                    "error");
                return;
            }
            if (job.status === "failed") {
                render(lastIdx, lastIdx);
                showMessage(
                    "Something went wrong while analyzing this scan. " +
                    '<a href="' + uploadUrl + '">Please try again</a>.' +
                    (job.error ? "<br><small>" + job.error + "</small>" : ""),
                    "error");
                return;
            }

            lastIdx = STEP_OF[job.status] ?? lastIdx;
            render(lastIdx, null);
            setTimeout(poll, 1000);
        } catch (err) {
            // transient network error — keep trying within the timeout
            setTimeout(poll, 2000);
        }
    }

    render(0, null);
    poll();
})();
