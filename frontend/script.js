
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const chatMessages = document.getElementById("chat-messages");

const API_URL = "http://127.0.0.1:8000/chat";

function addMessage(text, className) {
    const message = document.createElement("div");
    message.className = `message ${className}`;
    message.textContent = text;
    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return message;
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    addMessage(question, "user-message");
    questionInput.value = "";

    const sendButton = chatForm.querySelector("button");
    sendButton.disabled = true;

    const loadingMessage = addMessage("Thinking...", "assistant-message");

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }

        const data = await response.json();

        loadingMessage.innerHTML = DOMPurify.sanitize(marked.parse(data.answer));
        if (data.sources && data.sources.length > 0) {
            const sourcesHeading = document.createElement("p");
            sourcesHeading.textContent = "Sources:";
            sourcesHeading.style.fontWeight = "bold";
            loadingMessage.appendChild(sourcesHeading);

        for (const source of data.sources) {
            const sourceElement = document.createElement("div");

            if (source.url && source.source === "confluence") {
                const link = document.createElement("a");
                link.href = source.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.textContent = source.title;
                sourceElement.appendChild(link);
            } else {
                sourceElement.textContent = source.document || source.title;
                sourceElement.classList.add("local-source");
            }

            loadingMessage.appendChild(sourceElement);
        }
        }
    } catch (error) {
        loadingMessage.textContent =
            "Sorry, I couldn't reach the assistant. Please check that the API server is running and try again.";

        console.error("Chat request failed:", error);
    } finally {
        sendButton.disabled = false;
        questionInput.focus();
    }
});
