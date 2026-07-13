/* Shared UI behaviour: theme persistence + dark reading mode, responsive nav,
   upload zone (JPG/PNG/DICOM), CSRF helper for fetch POSTs.
   No inline handlers — the CSP forbids them; everything wires up here. */

// ---------- CSRF ----------
function csrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.content : "";
}
window.csrfToken = csrfToken;

// ---------- Theme (global choice ▸ per-scan-page choice ▸ page default) ----------
function isScanPage() {
    return !!document.body.dataset.themeDefault;
}

function resolveDark() {
    const pageDefault = document.body.dataset.themeDefault || "";
    const choice = isScanPage()
        ? (localStorage.getItem("theme-scan") || localStorage.getItem("theme"))
        : localStorage.getItem("theme");
    if (choice) return choice === "dark";
    return pageDefault === "dark";
}

function applyTheme() {
    const dark = resolveDark();
    document.body.classList.toggle("dark-mode", dark);
    // Reading palette: scan pages in the dark get the deep-dark canvas.
    document.body.classList.toggle("reading-mode", dark && isScanPage());
    document.documentElement.classList.remove("dark-mode-boot");
    if (typeof window.updateChartTheme === "function") window.updateChartTheme();
    const lights = document.querySelector(".lights-toggle");
    if (lights) lights.textContent = dark ? "☀ Lights" : "☾ Reading mode";
}

function toggleDarkMode() {
    const dark = !document.body.classList.contains("dark-mode");
    localStorage.setItem("theme", dark ? "dark" : "light");
    if (isScanPage()) localStorage.setItem("theme-scan", dark ? "dark" : "light");
    applyTheme();
}

function toggleLights() {
    // Scan-page-only toggle: persists separately from the global choice.
    const dark = !document.body.classList.contains("dark-mode");
    localStorage.setItem("theme-scan", dark ? "dark" : "light");
    applyTheme();
}

// ---------- Responsive nav ----------
function toggleNav() {
    const links = document.getElementById("navLinks");
    if (links) links.classList.toggle("open");
}

// ---------- Upload zone (index page) ----------
const IMAGE_MAX_BYTES = 15 * 1024 * 1024;
const DICOM_MAX_BYTES = 50 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];

function isDicomName(name) {
    return /\.dcm$/i.test(name || "");
}

function showUploadError(msg) {
    const el = document.getElementById("uploadError");
    if (el) {
        el.textContent = msg;
        el.classList.remove("hidden");
    } else {
        alert(msg);
    }
}

function clearUploadError() {
    const el = document.getElementById("uploadError");
    if (el) el.classList.add("hidden");
}

function acceptFile(file) {
    clearUploadError();
    if (!file) return;

    const dicom = isDicomName(file.name);
    if (!dicom && !ALLOWED_TYPES.includes(file.type)) {
        showUploadError("Unsupported file type — please choose a JPG/PNG X-ray image or a DICOM (.dcm) file.");
        return;
    }
    const cap = dicom ? DICOM_MAX_BYTES : IMAGE_MAX_BYTES;
    if (file.size > cap) {
        showUploadError("File is larger than " + (cap / 1024 / 1024) + " MB. Please choose a smaller file.");
        return;
    }

    const img = document.getElementById("previewImage");
    if (img) {
        if (dicom) {
            // Browsers can't render DICOM; show a neutral placeholder tile.
            img.src = "data:image/svg+xml;utf8," + encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" width="140" height="140">' +
                '<rect width="140" height="140" fill="#0b0e11"/>' +
                '<text x="70" y="66" fill="#7c9fd6" font-family="sans-serif" font-size="20" font-weight="700" text-anchor="middle">DICOM</text>' +
                '<text x="70" y="88" fill="#64748b" font-family="sans-serif" font-size="11" text-anchor="middle">converted on upload</text></svg>');
        } else {
            const reader = new FileReader();
            reader.onload = (e) => { img.src = e.target.result; };
            reader.readAsDataURL(file);
        }
    }

    const nameEl = document.getElementById("fileName");
    if (nameEl) nameEl.textContent = file.name + " (" + (file.size / 1024 / 1024).toFixed(2) + " MB)";
    document.getElementById("uploadZone")?.classList.add("hidden");
    document.getElementById("previewPanel")?.classList.remove("hidden");
}

function removeFile() {
    const input = document.getElementById("fileInput");
    if (input) input.value = "";
    clearUploadError();
    document.getElementById("previewPanel")?.classList.add("hidden");
    document.getElementById("uploadZone")?.classList.remove("hidden");
}

function submitForm() {
    const input = document.getElementById("fileInput");
    if (!input || !input.files.length) return false;
    document.getElementById("btnSubmit")?.classList.add("hidden");
    document.getElementById("btnLoading")?.classList.remove("hidden");
    return true;
}

// ---------- Wire up ----------
document.addEventListener("DOMContentLoaded", () => {
    applyTheme();

    // Delegated actions (no inline handlers under CSP).
    document.addEventListener("click", (e) => {
        const target = e.target.closest("[data-action]");
        if (!target) return;
        switch (target.dataset.action) {
            case "toggle-theme": toggleDarkMode(); break;
            case "toggle-nav": toggleNav(); break;
            case "toggle-lights": toggleLights(); break;
            case "dismiss-alert": target.parentElement.remove(); break;
            case "remove-file": removeFile(); break;
        }
    });

    const input = document.getElementById("fileInput");
    if (input) input.addEventListener("change", (e) => acceptFile(e.target.files[0]));

    const uploadForm = document.getElementById("uploadForm");
    if (uploadForm) {
        uploadForm.addEventListener("submit", (e) => {
            if (!submitForm()) e.preventDefault();
        });
    }

    const zone = document.getElementById("uploadZone");
    if (zone && input) {
        ["dragenter", "dragover"].forEach((evt) =>
            zone.addEventListener(evt, (e) => {
                e.preventDefault();
                zone.classList.add("dragover");
            })
        );
        ["dragleave", "drop"].forEach((evt) =>
            zone.addEventListener(evt, (e) => {
                e.preventDefault();
                zone.classList.remove("dragover");
            })
        );
        zone.addEventListener("drop", (e) => {
            const file = e.dataTransfer.files[0];
            if (file) {
                // Reflect the dropped file into the real input so the form submits it.
                const dt = new DataTransfer();
                dt.items.add(file);
                input.files = dt.files;
                acceptFile(file);
            }
        });
    }
});
