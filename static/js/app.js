/* Shared UI behaviour: theme persistence, responsive nav, upload zone. */

// ---------- Theme ----------
function applyTheme() {
    const dark = localStorage.getItem("theme") === "dark";
    document.body.classList.toggle("dark-mode", dark);
    document.documentElement.classList.remove("dark-mode-boot");
}

function toggleDarkMode() {
    const dark = !document.body.classList.contains("dark-mode");
    document.body.classList.toggle("dark-mode", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
    if (typeof window.updateChartTheme === "function") window.updateChartTheme();
}

// ---------- Responsive nav ----------
function toggleNav() {
    const links = document.getElementById("navLinks");
    if (links) links.classList.toggle("open");
}

// ---------- Upload zone (index page) ----------
const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];

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

    if (!ALLOWED_TYPES.includes(file.type)) {
        showUploadError("Unsupported file type — please choose a JPG or PNG X-ray image.");
        return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
        showUploadError("File is larger than 15 MB. Please choose a smaller image.");
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById("previewImage");
        if (img) img.src = e.target.result;
    };
    reader.readAsDataURL(file);

    const nameEl = document.getElementById("fileName");
    if (nameEl) nameEl.textContent = file.name + " (" + (file.size / 1024 / 1024).toFixed(2) + " MB)";
    document.getElementById("uploadZone")?.classList.add("hidden");
    document.getElementById("previewPanel")?.classList.remove("hidden");
}

function handleFileSelect(event) {
    acceptFile(event.target.files[0]);
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

    const zone = document.getElementById("uploadZone");
    const input = document.getElementById("fileInput");
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
