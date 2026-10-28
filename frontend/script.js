const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const chatMessages = document.getElementById("chat-messages");
const welcomeSection = document.getElementById("welcome-section");
const newChatButton = document.getElementById("new-chat-button");

const API_URL = "/chat";

let conversationId = null;

const initialGreeting = `
    <div class="message assistant-message">
        Hello! How can I help you today?
    </div>
`;

function startChat() {
    chatMessages.classList.add("has-messages");

    if (welcomeSection) {
        welcomeSection.style.display = "none";
    }
}

function addMessage(text, className) {
    const message = document.createElement("div");
    message.className = `message ${className}`;
    message.textContent = text;

    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return message;
}

// Send a question to the API
chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    startChat();

    addMessage(question, "user-message");
    questionInput.value = "";

    const sendButton = chatForm.querySelector("button");
    sendButton.disabled = true;

    const loadingMessage = addMessage(
        "Thinking...",
        "assistant-message"
    );

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question,
                conversation_id: conversationId
            })
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }

        const data = await response.json();
        conversationId = data.conversation_id;

        // Render the answer safely as Markdown
        loadingMessage.innerHTML = DOMPurify.sanitize(
            marked.parse(data.answer)
        );

        // Display source documents
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
                    link.textContent = source.title || "View source";

                    sourceElement.appendChild(link);
                } else {
                    sourceElement.textContent =
                        source.document || source.title || "Local document";

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
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});

// Make suggested questions clickable
document.querySelectorAll("[data-question]").forEach((suggestion) => {
    suggestion.addEventListener("click", () => {
        questionInput.value = suggestion.dataset.question;
        chatForm.requestSubmit();
    });
});

// Start a new conversation
if (newChatButton) {
    newChatButton.addEventListener("click", () => {
        chatMessages.innerHTML = initialGreeting;
        conversationId = null;
        chatMessages.classList.remove("has-messages");

        if (welcomeSection) {
            welcomeSection.style.display = "";
        }

        questionInput.value = "";
        questionInput.focus();
    });
}