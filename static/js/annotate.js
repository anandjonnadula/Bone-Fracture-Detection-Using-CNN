/* Doctor annotation editor: draws vector shapes (arrow, ellipse, freehand,
   label) in IMAGE space on a canvas layered inside the shared X-ray viewer,
   so annotations survive zoom/pan. Saves normalized-coordinate JSON to
   POST /api/scans/<id>/annotations with the CSRF header; the server render
   (Pillow) is the canonical one that lands in the PDF.

   Markup contract:
   <div class="annot-toolbar" data-annotator-for="{viewer block id}"
        data-save-url="/api/scans/123/annotations"> …buttons… </div> */

(function () {
    "use strict";

    const COLORS = ["#ff5252", "#ffd740", "#40c4ff"];

    function initAnnotator(toolbar) {
        const block = document.getElementById(toolbar.dataset.annotatorFor);
        if (!block) return;
        const api = block._viewer || window.initXrayViewer(block);
        if (!api) return;

        const canvas = api.addCanvasLayer("edit");
        const status = toolbar.querySelector(".annot-status");

        const state = {
            tool: "select",
            color: COLORS[0],
            width: 3,
            shapes: [],       // committed, normalized 0-1
            drawing: null,    // in-progress shape (image-space px)
            dirty: false,
        };

        function imgSize() {
            return { w: api.base.naturalWidth || 1, h: api.base.naturalHeight || 1 };
        }

        function syncCanvas() {
            const { w, h } = imgSize();
            if (canvas.width !== w || canvas.height !== h) {
                canvas.width = w;
                canvas.height = h;
            }
        }

        function redraw() {
            syncCanvas();
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const { w, h } = imgSize();
            state.shapes.forEach((s) => window.XrayShapes.drawShape(ctx, s, w, h));
            if (state.drawing) window.XrayShapes.drawShape(ctx, state.drawing, w, h);
        }
        if (api.base.complete && api.base.naturalWidth) syncCanvas();
        else api.base.addEventListener("load", () => { syncCanvas(); redraw(); });

        function setStatus(text) {
            if (status) status.textContent = text;
        }

        function markDirty() {
            state.dirty = true;
            setStatus(state.shapes.length + " shape(s) — unsaved");
        }

        // ---------- toolbar ----------
        function selectTool(tool) {
            state.tool = tool;
            api.setPanDisabled(tool !== "select");
            toolbar.querySelectorAll("[data-tool]").forEach((b) =>
                b.classList.toggle("active", b.dataset.tool === tool));
        }

        toolbar.querySelectorAll("[data-tool]").forEach((b) =>
            b.addEventListener("click", () => selectTool(b.dataset.tool)));

        toolbar.querySelectorAll(".annot-color").forEach((b, i) => {
            b.style.background = COLORS[i % COLORS.length];
            b.dataset.color = COLORS[i % COLORS.length];
            if (i === 0) b.classList.add("active");
            b.addEventListener("click", () => {
                state.color = b.dataset.color;
                toolbar.querySelectorAll(".annot-color").forEach((x) =>
                    x.classList.toggle("active", x === b));
            });
        });

        toolbar.querySelector("[data-annot-undo]")?.addEventListener("click", () => {
            state.shapes.pop();
            markDirty();
            redraw();
        });
        toolbar.querySelector("[data-annot-clear]")?.addEventListener("click", () => {
            if (state.shapes.length && !confirm("Remove all unsaved shapes?")) return;
            state.shapes = [];
            markDirty();
            redraw();
        });

        toolbar.querySelector("[data-annot-save]")?.addEventListener("click", async () => {
            if (!state.shapes.length) { setStatus("Nothing to save."); return; }
            setStatus("Saving…");
            try {
                const { w, h } = imgSize();
                const resp = await fetch(toolbar.dataset.saveUrl, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-CSRFToken": window.csrfToken(),
                    },
                    body: JSON.stringify({ image_w: w, image_h: h, shapes: state.shapes }),
                });
                if (!resp.ok) throw new Error("HTTP " + resp.status);
                state.dirty = false;
                setStatus("Saved ✓ — annotations will appear in the regenerated PDF after you record the review.");
            } catch (err) {
                setStatus("Save failed — " + err.message);
            }
        });

        // ---------- drawing ----------
        function norm(pt) {
            const { w, h } = imgSize();
            return { x: pt.x / w, y: pt.y / h };
        }

        api.viewer.addEventListener("pointerdown", (e) => {
            if (state.tool === "select") return;
            e.stopPropagation();
            api.viewer.setPointerCapture(e.pointerId);
            const p = api.clientToImage(e.clientX, e.clientY);
            const n = norm(p);

            if (state.tool === "label") {
                const text = prompt("Label text:");
                if (text && text.trim()) {
                    state.shapes.push({ type: "label", x: n.x, y: n.y,
                        text: text.trim().slice(0, 120), color: state.color });
                    markDirty();
                    redraw();
                }
                return;
            }

            if (state.tool === "arrow") {
                state.drawing = { type: "arrow", x1: n.x, y1: n.y, x2: n.x, y2: n.y,
                    color: state.color, width: state.width };
            } else if (state.tool === "ellipse") {
                state.drawing = { type: "ellipse", cx: n.x, cy: n.y, rx: 0, ry: 0,
                    color: state.color, width: state.width, _ox: n.x, _oy: n.y };
            } else if (state.tool === "path") {
                state.drawing = { type: "path", points: [[n.x, n.y]],
                    color: state.color, width: Math.max(2, state.width - 1) };
            }
        }, true);

        api.viewer.addEventListener("pointermove", (e) => {
            if (!state.drawing) return;
            e.stopPropagation();
            const n = norm(api.clientToImage(e.clientX, e.clientY));
            const d = state.drawing;
            if (d.type === "arrow") {
                d.x2 = n.x; d.y2 = n.y;
            } else if (d.type === "ellipse") {
                d.cx = (d._ox + n.x) / 2;
                d.cy = (d._oy + n.y) / 2;
                d.rx = Math.abs(n.x - d._ox) / 2;
                d.ry = Math.abs(n.y - d._oy) / 2;
            } else if (d.type === "path") {
                const last = d.points[d.points.length - 1];
                if (Math.hypot(n.x - last[0], n.y - last[1]) > 0.003) d.points.push([n.x, n.y]);
            }
            redraw();
        }, true);

        function finish(e) {
            if (!state.drawing) return;
            e.stopPropagation();
            const d = state.drawing;
            state.drawing = null;
            // discard degenerate shapes (accidental clicks)
            const tooSmall =
                (d.type === "arrow" && Math.hypot(d.x2 - d.x1, d.y2 - d.y1) < 0.01) ||
                (d.type === "ellipse" && (d.rx < 0.005 || d.ry < 0.005)) ||
                (d.type === "path" && d.points.length < 3);
            if (!tooSmall) {
                delete d._ox; delete d._oy;
                state.shapes.push(d);
                markDirty();
            }
            redraw();
        }
        api.viewer.addEventListener("pointerup", finish, true);
        api.viewer.addEventListener("pointercancel", finish, true);

        window.addEventListener("beforeunload", (e) => {
            if (state.dirty && state.shapes.length) {
                e.preventDefault();
                e.returnValue = "";
            }
        });

        selectTool("select");
        setStatus("Pick a tool to mark the radiograph.");
    }

    document.addEventListener("DOMContentLoaded", () => {
        document.querySelectorAll("[data-annotator-for]").forEach(initAnnotator);
    });
})();
