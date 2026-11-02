
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const chatMessages = document.getElementById("chat-messages");
const welcomeSection = document.getElementById("welcome-section");
const newChatButton = document.getElementById("new-chat-button");

const authScreen = document.getElementById("auth-screen");
const appLayout = document.getElementById("app-layout");
const signInContainer = document.getElementById("clerk-sign-in");
const authError = document.getElementById("auth-error");

const API_URL = "/chat";

let conversationId = null;
let clerk = null;
let signInMounted = false;
let organizationSwitcherMounted = false;
let userButtonMounted = false;

const initialGreeting = `
    <div class="message assistant-message">
        Hello! How can I help you today?
    </div>
`;

function showSignedOut() {
    appLayout.hidden = true;
    authScreen.hidden = false;

    if (clerk && !signInMounted) {
        clerk.mountSignIn(signInContainer);
        signInMounted = true;
    }
}

function setupClerkControls() {
    let controls = document.getElementById("clerk-controls");

    if (controls) {
        return controls;
    }

    const topbar = document.querySelector(".topbar");

    controls = document.createElement("div");
    controls.id = "clerk-controls";
    controls.className = "clerk-controls";

    controls.innerHTML = `
        <div id="clerk-org-switcher"></div>
        <div id="clerk-user-button"></div>
    `;

    topbar.appendChild(controls);

    return controls;
}

function showSignedIn() {
    authScreen.hidden = true;
    appLayout.hidden = false;

    if (signInMounted) {
        clerk.unmountSignIn(signInContainer);
        signInMounted = false;
    }

    setupClerkControls();

    if (!organizationSwitcherMounted) {
        clerk.mountOrganizationSwitcher(
            document.getElementById("clerk-org-switcher"),
            { hidePersonal: true }
        );

        organizationSwitcherMounted = true;
    }

    if (!userButtonMounted) {
        clerk.mountUserButton(
            document.getElementById("clerk-user-button")
        );

        userButtonMounted = true;
    }
}

async function initializeClerk() {
    try {
        if (!window.Clerk) {
            throw new Error(
                "Clerk did not load. Check the Clerk script tags in index.html."
            );
        }

        clerk = window.Clerk;

        await clerk.load({
            ui: { ClerkUI: window.__internal_ClerkUICtor },
        });

        clerk.addListener(({ user, session }) => {
            if (user && session) {
                showSignedIn();
            } else if (user === null && session === null) {
                showSignedOut();
            }
        });

        if (clerk.isSignedIn) {
            showSignedIn();
        } else {
            showSignedOut();
        }
    } catch (error) {
        console.error("Clerk initialization failed:", error);

        appLayout.hidden = true;
        authScreen.hidden = false;
        authError.hidden = false;
        authError.textContent =
            "Authentication could not be loaded. Check the Clerk configuration and reload the page.";
    }
}

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

async function getAuthToken() {
    console.log("getAuthToken function called");
    if (!clerk || !clerk.session) {
        throw new Error("Your session is not active. Please sign in again.");
    }

    const organizationId = clerk.organization?.id;

    if (!organizationId) {
        throw new Error(
            "Select your company using the organization selector in the top bar."
        );
    }

    const token = await clerk.session.getToken({ organizationId });
    console.log("Token received:", Boolean(token));
    console.log("Organization selected:", Boolean(organizationId));
    if (!token) {
        throw new Error("Could not obtain a session token. Please sign in again.");
    }

    return token;
}

// Send an authenticated question to the API.
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
        const token = await getAuthToken();
        console.log("Token available before fetch:", Boolean(token));
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`,
            },
            body: JSON.stringify({
                question,
                conversation_id: conversationId,
            }),
        });

        if (response.status === 401) {
            throw new Error(
                "Your session was not accepted. Please sign in again."
            );
        }

        if (response.status === 403) {
            throw new Error(
                "You don't have permission to access this company."
            );
        }

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }

        const data = await response.json();
        conversationId = data.conversation_id;

        // Render the answer safely as Markdown.
        loadingMessage.innerHTML = DOMPurify.sanitize(
            marked.parse(data.answer)
        );

        // Display source documents.
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
        console.error("Chat request failed:", error);

        loadingMessage.textContent =
            error instanceof Error &&
            (
                error.message.startsWith("Select your company") ||
                error.message.startsWith("Your session") ||
                error.message.startsWith("Could not obtain") ||
                error.message.startsWith("You don't have permission")
            )
                ? error.message
                : "Sorry, I couldn't reach the assistant. Please check the server and try again.";
    } finally {
        sendButton.disabled = false;
        questionInput.focus();
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});

// Make suggested questions clickable.
document.querySelectorAll("[data-question]").forEach((suggestion) => {
    suggestion.addEventListener("click", () => {
        questionInput.value = suggestion.dataset.question;
        chatForm.requestSubmit();
    });
});

// Start a new conversation.
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

// Clerk's two CDN scripts should appear before this script in index.html.
window.addEventListener("load", initializeClerk, { once: true });
