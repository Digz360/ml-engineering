async function sendMessage() {
    const input = document.getElementById("prompt");
    const chat = document.getElementById("chat");
    const rag = document.getElementById("rag").checked;

    const prompt = input.value.trim();

    if (!prompt) return;

    chat.innerHTML += `
        <div class="message user">
            <strong>Tu:</strong> ${prompt}
        </div>
    `;

    input.value = "";

    const response = await fetch("/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            prompt: prompt,
            use_rag: rag,
            max_new_tokens: 100
        })
    });

    const data = await response.json();

    chat.innerHTML += `
        <div class="message ai">
            <strong>MiniGPT:</strong> ${data.answer}
        </div>
    `;

    chat.scrollTop = chat.scrollHeight;
}


// ELEMENTOS
const promptInput = document.getElementById("prompt");
const ragToggle = document.getElementById("rag");
const ragPanel = document.getElementById("rag-panel");
const ragFile = document.getElementById("rag-file");


// MOSTRAR / ESCONDER RAG
ragToggle.addEventListener("change", () => {
    ragPanel.classList.toggle("hidden", !ragToggle.checked);
});


// ENTER ENVIA
promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});


// UPLOAD RAG
ragFile.addEventListener("change", async () => {
    const file = ragFile.files[0];

    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/rag/upload", {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        alert("Erro ao carregar documento.");
        return;
    }

    const data = await response.json();

    alert(`RAG carregado: ${data.filename}`);
});