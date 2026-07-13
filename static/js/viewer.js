/* Interactive X-ray viewer: stacked layers (original / Grad-CAM RGBA /
   detector boxes / doctor annotations) with opacity slider, wheel + pinch
   zoom, drag pan, double-click reset, and an "M" shortcut toggling the map.

   Markup contract (see result.html / doctor_dashboard.html):

   <div class="viewer-block" data-viewer
        data-cam="{url|empty}" data-detections="{json|empty}"
        data-annotations="{json|empty}">
     <div class="xray-viewer">
       <div class="layers"><img class="base" src="..."></div>
     </div>
     <div class="viewer-controls">…slider/checkbox controls…</div>
   </div>

   Layers draw in IMAGE space (canvas pixel size = natural image size) and
   inherit the shared pan/zoom transform, so zooming never blurs vectors'
   positions or requires redraws. Shape-drawing helpers are shared with
   annotate.js via window.XrayShapes. */

(function () {
    "use strict";

    // ---------- shared shape drawing (image-space pixels) ----------
    function drawShape(ctx, s, w, h) {
        const scale = Math.max(w, h) / 1000;
        ctx.strokeStyle = s.color || "#ff5252";
        ctx.fillStyle = s.color || "#ff5252";
        ctx.lineWidth = Math.max(1, (s.width || 3) * scale);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        if (s.type === "ellipse") {
            ctx.beginPath();
            ctx.ellipse(s.cx * w, s.cy * h, Math.abs(s.rx * w), Math.abs(s.ry * h), 0, 0, Math.PI * 2);
            ctx.stroke();
        } else if (s.type === "arrow") {
            const x1 = s.x1 * w, y1 = s.y1 * h, x2 = s.x2 * w, y2 = s.y2 * h;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
            const angle = Math.atan2(y2 - y1, x2 - x1);
            const head = Math.max(10, ctx.lineWidth * 4);
            [Math.PI / 7, -Math.PI / 7].forEach((off) => {
                ctx.beginPath();
                ctx.moveTo(x2, y2);
                ctx.lineTo(x2 - head * Math.cos(angle + off), y2 - head * Math.sin(angle + off));
                ctx.stroke();
            });
        } else if (s.type === "path") {
            if (!s.points || s.points.length < 2) return;
            ctx.beginPath();
            ctx.moveTo(s.points[0][0] * w, s.points[0][1] * h);
            for (let i = 1; i < s.points.length; i++) {
                ctx.lineTo(s.points[i][0] * w, s.points[i][1] * h);
            }
            ctx.stroke();
        } else if (s.type === "label") {
            const size = Math.max(14, Math.round(16 * scale));
            ctx.font = "600 " + size + "px Inter, sans-serif";
            const text = String(s.text || "").slice(0, 120);
            const x = s.x * w, y = s.y * h;
            const m = ctx.measureText(text);
            ctx.save();
            ctx.fillStyle = "rgba(0,0,0,0.75)";
            ctx.fillRect(x - 4, y - size, m.width + 8, size + 8);
            ctx.restore();
            ctx.fillText(text, x, y);
        }
    }

    function drawDocument(ctx, doc, w, h) {
        (doc.shapes || []).forEach((s) => drawShape(ctx, s, w, h));
    }

    window.XrayShapes = { drawShape, drawDocument };

    // ---------- viewer ----------
    function initViewer(block) {
        const viewer = block.querySelector(".xray-viewer");
        const layers = block.querySelector(".layers");
        const base = block.querySelector("img.base");
        if (!viewer || !layers || !base) return null;

        const state = {
            scale: 1, tx: 0, ty: 0,
            lastOpacity: 0.7,
            pointers: new Map(),
            pinchDist: 0,
            panDisabled: false,
        };

        function applyTransform() {
            layers.style.transform =
                "translate(" + state.tx + "px," + state.ty + "px) scale(" + state.scale + ")";
        }

        function reset() {
            state.scale = 1; state.tx = 0; state.ty = 0;
            applyTransform();
        }

        // --- layers ---
        function addCanvasLayer(cls) {
            const c = document.createElement("canvas");
            c.className = "layer " + cls;
            layers.appendChild(c);
            return c;
        }

        const camUrl = block.dataset.cam;
        let camImg = null;
        if (camUrl) {
            camImg = document.createElement("img");
            camImg.className = "layer cam";
            camImg.alt = "";
            camImg.src = camUrl;
            camImg.style.opacity = state.lastOpacity;
            layers.appendChild(camImg);
        }

        let detections = [];
        let annotations = [];
        try { detections = JSON.parse(block.dataset.detections || "[]") || []; } catch (e) { /* ignore */ }
        try { annotations = JSON.parse(block.dataset.annotations || "[]") || []; } catch (e) { /* ignore */ }

        const boxCanvas = detections.length ? addCanvasLayer("boxes") : null;
        const noteCanvas = annotations.length ? addCanvasLayer("notes") : null;

        function renderStatic() {
            const w = base.naturalWidth, h = base.naturalHeight;
            if (!w || !h) return;
            if (boxCanvas) {
                boxCanvas.width = w; boxCanvas.height = h;
                const ctx = boxCanvas.getContext("2d");
                const lw = Math.max(2, Math.round(3 * Math.max(w, h) / 1000));
                ctx.lineWidth = lw;
                ctx.strokeStyle = "#22d3ee";
                ctx.font = "700 " + Math.max(14, Math.round(16 * Math.max(w, h) / 1000)) + "px Inter, sans-serif";
                detections.forEach((d) => {
                    ctx.strokeRect(d.x * w, d.y * h, d.w * w, d.h * h);
                    const label = "fracture " + Math.round((d.conf || 0) * 100) + "%";
                    const tw = ctx.measureText(label).width;
                    ctx.fillStyle = "#0e7490";
                    ctx.fillRect(d.x * w, Math.max(0, d.y * h - 22), tw + 10, 20);
                    ctx.fillStyle = "#fff";
                    ctx.fillText(label, d.x * w + 5, Math.max(14, d.y * h - 7));
                });
            }
            if (noteCanvas) {
                noteCanvas.width = w; noteCanvas.height = h;
                const ctx = noteCanvas.getContext("2d");
                annotations.forEach((doc) => drawDocument(ctx, doc, w, h));
            }
        }
        if (base.complete && base.naturalWidth) renderStatic();
        else base.addEventListener("load", renderStatic);

        // --- controls ---
        const slider = block.querySelector("[data-cam-opacity]");
        const valueOut = block.querySelector("[data-cam-value]");
        function setOpacity(v, remember) {
            if (!camImg) return;
            camImg.style.opacity = v;
            if (remember && v > 0) state.lastOpacity = v;
            if (slider) {
                slider.value = Math.round(v * 100);
                slider.setAttribute("aria-valuetext", "Heatmap " + Math.round(v * 100) + "% visible");
            }
            if (valueOut) valueOut.textContent = Math.round(v * 100) + "%";
        }
        if (slider && camImg) {
            slider.addEventListener("input", () => setOpacity(slider.value / 100, true));
            setOpacity(state.lastOpacity, true);
        }
        function toggleCam() {
            if (!camImg) return;
            const current = parseFloat(camImg.style.opacity || "0");
            setOpacity(current > 0 ? 0 : state.lastOpacity, false);
        }
        block.querySelectorAll("[data-layer-toggle]").forEach((cb) => {
            cb.addEventListener("change", () => {
                const which = cb.dataset.layerToggle;
                const el = which === "cam" ? camImg
                    : which === "boxes" ? boxCanvas
                    : which === "notes" ? noteCanvas : null;
                if (which === "cam") { setOpacity(cb.checked ? state.lastOpacity : 0, false); return; }
                if (el) el.style.display = cb.checked ? "" : "none";
            });
        });
        const resetBtn = block.querySelector(".viewer-reset");
        if (resetBtn) resetBtn.addEventListener("click", reset);

        // Keyboard: "M" toggles the map when the pointer is over the viewer.
        let hovered = false;
        viewer.addEventListener("mouseenter", () => { hovered = true; });
        viewer.addEventListener("mouseleave", () => { hovered = false; });
        document.addEventListener("keydown", (e) => {
            if (hovered && (e.key === "m" || e.key === "M") &&
                !/input|textarea/i.test(document.activeElement.tagName)) {
                toggleCam();
            }
        });

        // --- zoom / pan (mouse + touch via pointer events) ---
        viewer.addEventListener("wheel", (e) => {
            e.preventDefault();
            const rect = viewer.getBoundingClientRect();
            const mx = e.clientX - rect.left, my = e.clientY - rect.top;
            const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            const next = Math.min(12, Math.max(1, state.scale * factor));
            const applied = next / state.scale;
            // zoom toward the cursor
            state.tx = mx - (mx - state.tx) * applied;
            state.ty = my - (my - state.ty) * applied;
            state.scale = next;
            if (state.scale === 1) { state.tx = 0; state.ty = 0; }
            applyTransform();
        }, { passive: false });

        viewer.addEventListener("dblclick", reset);

        viewer.addEventListener("pointerdown", (e) => {
            if (state.panDisabled) return;
            viewer.setPointerCapture(e.pointerId);
            state.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
            if (state.pointers.size === 2) {
                const pts = [...state.pointers.values()];
                state.pinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
            }
        });
        viewer.addEventListener("pointermove", (e) => {
            if (!state.pointers.has(e.pointerId)) return;
            const prev = state.pointers.get(e.pointerId);
            state.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

            if (state.pointers.size === 1) {
                state.tx += e.clientX - prev.x;
                state.ty += e.clientY - prev.y;
                applyTransform();
            } else if (state.pointers.size === 2) {
                const pts = [...state.pointers.values()];
                const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
                if (state.pinchDist > 0) {
                    const rect = viewer.getBoundingClientRect();
                    const cx = (pts[0].x + pts[1].x) / 2 - rect.left;
                    const cy = (pts[0].y + pts[1].y) / 2 - rect.top;
                    const next = Math.min(12, Math.max(1, state.scale * (dist / state.pinchDist)));
                    const applied = next / state.scale;
                    state.tx = cx - (cx - state.tx) * applied;
                    state.ty = cy - (cy - state.ty) * applied;
                    state.scale = next;
                    applyTransform();
                }
                state.pinchDist = dist;
            }
        });
        function endPointer(e) {
            state.pointers.delete(e.pointerId);
            if (state.pointers.size < 2) state.pinchDist = 0;
        }
        viewer.addEventListener("pointerup", endPointer);
        viewer.addEventListener("pointercancel", endPointer);

        // API used by annotate.js
        const api = {
            block, viewer, layers, base,
            addCanvasLayer,
            setPanDisabled(v) { state.panDisabled = v; viewer.classList.toggle("annotating", v); },
            clientToImage(clientX, clientY) {
                // invert the pan/zoom transform, then map display px -> natural px
                const rect = viewer.getBoundingClientRect();
                const dx = (clientX - rect.left - state.tx) / state.scale;
                const dy = (clientY - rect.top - state.ty) / state.scale;
                return {
                    x: dx * (base.naturalWidth / base.clientWidth),
                    y: dy * (base.naturalHeight / base.clientHeight),
                };
            },
        };
        block._viewer = api;
        return api;
    }

    document.addEventListener("DOMContentLoaded", () => {
        document.querySelectorAll("[data-viewer]").forEach(initViewer);
    });

    window.initXrayViewer = initViewer;
})();
