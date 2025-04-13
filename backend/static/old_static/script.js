// const chatbox = document.getElementById('chatbox');
// const userInput = document.getElementById('userInput');
// const sendButton = document.getElementById('sendButton');
// const loadingIndicator = document.getElementById('loading');
// const errorDisplay = document.getElementById('error');

// // Use a simple session ID stored in the browser's session storage
// let sessionId = sessionStorage.getItem('ragSessionId');
// if (!sessionId) {
//     sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
//     sessionStorage.setItem('ragSessionId', sessionId);
// }

// console.log("Using Session ID:", sessionId);

// function addMessage(message, type) {
//     const messageDiv = document.createElement('div');
//     messageDiv.classList.add('message', type === 'user' ? 'user-message' : 'bot-message');

//     // Sanitize message content before adding to DOM (basic example)
//     const textNode = document.createTextNode(message);
//     messageDiv.appendChild(textNode);

//     chatbox.appendChild(messageDiv);
//     chatbox.scrollTop = chatbox.scrollHeight; // Scroll to bottom
// }

// function addBotMessageWithSources(answer, sources) {
//     const messageDiv = document.createElement('div');
//     messageDiv.classList.add('message', 'bot-message');

//     // Sanitize and add answer
//     const answerNode = document.createElement('p');
//     answerNode.appendChild(document.createTextNode(answer));
//     messageDiv.appendChild(answerNode);

//     if (sources && sources.length > 0) {
//         const sourcesDiv = document.createElement('div');
//         sourcesDiv.classList.add('sources');
//         sourcesDiv.innerHTML = '<strong>Sources:</strong>';

//         sources.forEach((source, index) => {
//             const sourceItemDiv = document.createElement('div');
//             sourceItemDiv.classList.add('source-item');
//             // Sanitize source info
//             const sourceName = source.source ? document.createTextNode(`[${index+1}] ${source.source}: `) : document.createTextNode(`[${index+1}] Unknown Source: `);
//             const sourceContent = document.createTextNode(source.content.substring(0, 150) + '...'); // Show snippet
//             sourceItemDiv.appendChild(sourceName);
//             sourceItemDiv.appendChild(sourceContent);
//             sourcesDiv.appendChild(sourceItemDiv);
//         });
//         messageDiv.appendChild(sourcesDiv);
//     }

//     chatbox.appendChild(messageDiv);
//     chatbox.scrollTop = chatbox.scrollHeight;
// }

// async function sendMessage() {
//     const query = userInput.value.trim();
//     if (!query) return;

//     addMessage(query, 'user');
//     userInput.value = '';
//     loadingIndicator.style.display = 'block';
//     errorDisplay.style.display = 'none'; // Hide previous errors
//     sendButton.disabled = true;

//     try {
//         const response = await fetch('/chat', { // Assumes API is served on the same origin
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({ query: query, session_id: sessionId }),
//         });

//         loadingIndicator.style.display = 'none';
//         sendButton.disabled = false;

//         if (!response.ok) {
//             const errorData = await response.json();
//             throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
//         }

//         const data = await response.json();
//         addBotMessageWithSources(data.answer, data.sources);

//     } catch (error) {
//         console.error('Error:', error);
//         loadingIndicator.style.display = 'none';
//         errorDisplay.textContent = `Error: ${error.message}`;
//         errorDisplay.style.display = 'block';
//         sendButton.disabled = false;
//     }
// }

// sendButton.addEventListener('click', sendMessage);
// userInput.addEventListener('keypress', function (e) {
//     if (e.key === 'Enter') {
//         sendMessage();
//     }
// });

// // Initial welcome message (optional)
// // addMessage("Hello! Ask me anything about our documents.", 'bot');

// --- Existing Elements ---
const chatbox = document.getElementById('chatbox');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const loadingIndicator = document.getElementById('loading');
const errorDisplay = document.getElementById('error');

// --- New Elements (Add these to index.html) ---
const loginForm = document.getElementById('loginForm'); // A form with email/password inputs
const emailInput = document.getElementById('emailInput');
const passwordInput = document.getElementById('passwordInput');
const loginButton = document.getElementById('loginButton');
const logoutButton = document.getElementById('logoutButton'); // A button to log out
const userStatus = document.getElementById('userStatus'); // An area to show login status/email
const chatArea = document.getElementById('chatArea'); // Container for chatbox/input (to hide/show)
const loginArea = document.getElementById('loginArea'); // Container for login form (to hide/show)
const authErrorDisplay = document.getElementById('authError'); // Separate error display for login

// --- Authentication State ---
const API_BASE_URL = "/api/v1"; // Use the API prefix from settings
let authToken = sessionStorage.getItem('authToken'); // Use sessionStorage (cleared on browser close) or localStorage (persists)
let userEmail = sessionStorage.getItem('userEmail');

// --- Utility Functions ---

function setAuthState(token, email) {
    if (token) {
        authToken = token;
        userEmail = email;
        sessionStorage.setItem('authToken', token);
        sessionStorage.setItem('userEmail', email);
        // UI Updates for logged-in state
        if(loginArea) loginArea.style.display = 'none';
        if(chatArea) chatArea.style.display = 'block'; // Show chat
        if(userStatus) userStatus.textContent = `Logged in as: ${email}`;
        if(logoutButton) logoutButton.style.display = 'inline-block';
        if(authErrorDisplay) authErrorDisplay.style.display = 'none';
        // Potentially clear chatbox on login/logout?
        // chatbox.innerHTML = '';
    } else {
        authToken = null;
        userEmail = null;
        sessionStorage.removeItem('authToken');
        sessionStorage.removeItem('userEmail');
        // UI Updates for logged-out state
        if(loginArea) loginArea.style.display = 'block'; // Show login
        if(chatArea) chatArea.style.display = 'none'; // Hide chat
        if(userStatus) userStatus.textContent = 'Not logged in.';
        if(logoutButton) logoutButton.style.display = 'none';
        chatbox.innerHTML = ''; // Clear chat on logout
    }
}

// --- API Call Functions ---

async function loginUser(email, password) {
    authErrorDisplay.style.display = 'none'; // Hide previous auth errors
    loadingIndicator.style.display = 'block'; // Show general loading maybe?
    loginButton.disabled = true;

    try {
        // FastAPI login expects form data, not JSON
        const formData = new FormData();
        formData.append('username', email); // OAuth2PasswordRequestForm uses 'username' field
        formData.append('password', password);

        const response = await fetch(`${API_BASE_URL}/auth/login`, {
            method: 'POST',
            body: formData, // Send as form data
        });

        loginButton.disabled = false;
        loadingIndicator.style.display = 'none';

        if (!response.ok) {
            let errorMsg = `Login failed! Status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.detail || errorMsg;
            } catch (e) { /* Ignore if response is not JSON */ }
            throw new Error(errorMsg);
        }

        const data = await response.json(); // Should contain access_token
        setAuthState(data.access_token, email); // Store token and update UI

    } catch (error) {
        console.error('Login Error:', error);
        setAuthState(null, null); // Ensure logged out state on error
        authErrorDisplay.textContent = `Login Error: ${error.message}`;
        authErrorDisplay.style.display = 'block';
        loginButton.disabled = false;
        loadingIndicator.style.display = 'none';
    }
}

function logoutUser() {
    setAuthState(null, null); // Clear token and update UI
    // Optional: Send request to backend /auth/logout endpoint if implemented
    console.log("User logged out.");
}

// --- Chat Functions (Modified) ---

function addMessage(message, type) {
    // ... (keep existing addMessage function) ...
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', type === 'user' ? 'user-message' : 'bot-message');
    const textNode = document.createTextNode(message);
    messageDiv.appendChild(textNode);
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function addBotMessageWithSources(answer, sources) {
    // ... (keep existing addBotMessageWithSources function) ...
     const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot-message');
    const answerNode = document.createElement('p');
    answerNode.appendChild(document.createTextNode(answer));
    messageDiv.appendChild(answerNode);
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.classList.add('sources');
        sourcesDiv.innerHTML = '<strong>Sources:</strong>';
        sources.forEach((source, index) => {
            const sourceItemDiv = document.createElement('div');
            sourceItemDiv.classList.add('source-item');
            const sourceName = source.document_id ? document.createTextNode(`[${index+1}] ${source.document_id} (Chunk ${source.chunk_number}): `) : document.createTextNode(`[${index+1}] Unknown Source: `);
            const sourceContent = document.createTextNode(source.content.substring(0, 150) + '...');
            sourceItemDiv.appendChild(sourceName);
            sourceItemDiv.appendChild(sourceContent);
            sourcesDiv.appendChild(sourceItemDiv);
        });
        messageDiv.appendChild(sourcesDiv);
    }
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
}

async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;

    // --- Authentication Check ---
    if (!authToken) {
        errorDisplay.textContent = "Error: You must be logged in to chat.";
        errorDisplay.style.display = 'block';
        return;
    }

    addMessage(query, 'user');
    userInput.value = '';
    loadingIndicator.style.display = 'block';
    errorDisplay.style.display = 'none';
    sendButton.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/chat/chat`, { // Correct endpoint path
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // --- Add Authorization Header ---
                'Authorization': `Bearer ${authToken}`
            },
            // Body remains the same (session_id is now optional/handled by backend)
            body: JSON.stringify({ query: query }),
        });

        loadingIndicator.style.display = 'none';
        sendButton.disabled = false;

        if (response.status === 401) { // Handle unauthorized specifically
             setAuthState(null, null); // Force logout on client
             errorDisplay.textContent = "Authentication error or session expired. Please log in again.";
             errorDisplay.style.display = 'block';
             return; // Stop processing
        }

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        addBotMessageWithSources(data.answer, data.sources);
        // Session ID is now managed mostly server-side, less need to track here unless needed for history endpoint

    } catch (error) {
        console.error('Chat Error:', error);
        loadingIndicator.style.display = 'none';
        errorDisplay.textContent = `Chat Error: ${error.message}`;
        errorDisplay.style.display = 'block';
        sendButton.disabled = false;
    }
}

// --- Event Listeners ---

// Login Form Submission
if (loginForm) {
    loginForm.addEventListener('submit', (e) => {
        e.preventDefault(); // Prevent default form submission
        const email = emailInput.value;
        const password = passwordInput.value;
        if (email && password) {
            loginUser(email, password);
        } else {
             if(authErrorDisplay) {
                authErrorDisplay.textContent = "Please enter both email and password.";
                authErrorDisplay.style.display = 'block';
             }
        }
    });
}

// Logout Button Click
if (logoutButton) {
    logoutButton.addEventListener('click', logoutUser);
}

// Chat Send Button Click
if (sendButton) {
    sendButton.addEventListener('click', sendMessage);
}

// Chat Input Enter Keypress
if (userInput) {
    userInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
}

// --- Initial UI State ---
document.addEventListener('DOMContentLoaded', () => {
    setAuthState(authToken, userEmail); // Set initial UI based on stored token
});