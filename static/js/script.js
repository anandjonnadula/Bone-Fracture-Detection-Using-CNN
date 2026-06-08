document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");
    const loader = document.getElementById("loader");
    const fileInput = document.getElementById("fileInput");
    const uploadLabel = document.querySelector('.upload-label span');

    const previewBox = document.getElementById("previewBox");
    const previewImage = document.getElementById("previewImage");
    const fileName = document.getElementById("fileName");
    const successText = document.getElementById("successText");

    // 🔄 SHOW LOADER ON SUBMIT
    if (form) {
        form.addEventListener("submit", function () {
            if (loader) {
                loader.classList.remove("hidden");
            }

            const btn = document.getElementById("submitBtn");
            if (btn) {
                btn.disabled = true;
                btn.innerText = "Processing...";
            }
        });
    }

    // 🖼️ IMAGE PREVIEW
    if (fileInput) {
        fileInput.addEventListener("change", function (e) {

            const file = e.target.files[0];

            if (file) {
                // ✅ Show preview box
                previewBox.classList.remove("hidden");

                // ✅ Show image
                previewImage.src = URL.createObjectURL(file);

                // ✅ Show file name
                fileName.innerText = "📄 " + file.name;

                // ✅ Update upload label
                uploadLabel.innerText = "✅ " + file.name;

                // ✅ Show success text
                successText.classList.remove("hidden");
            }
        });
    }

    // ❌ REMOVE IMAGE FUNCTION (GLOBAL)
    window.removeImage = function () {
        fileInput.value = "";
        previewImage.src = "";
        fileName.innerText = "";
        previewBox.classList.add("hidden");
        successText.classList.add("hidden");
        uploadLabel.innerText = "📤 Click to upload X-ray image";
    };

});