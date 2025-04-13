// --- DOM Elements ---
const chatbox = document.getElementById('chatbox');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const loadingIndicator = document.getElementById('loading');
const errorDisplay = document.getElementById('error');
const loginSection = document.getElementById('login-section');
const chatSection = document.getElementById('chat-section');
const loginForm = document.getElementById('loginForm');
const usernameInput = document.getElementById('username');
const passwordInput = document.getElementById('password');
const loginError = document.getElementById('login-error');
const logoutButton = document.getElementById('logoutButton');
const welcomeMessage = document.getElementById('welcome-message');

// --- State ---
let jwtToken = sessionStorage.getItem('authToken'); // Use sessionStorage for token

// --- API Functions ---
async function loginUser(username, password) {
    try {
        const response = await fetch('/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded', // FastAPI expects form data for OAuth2
            },
            body: new URLSearchParams({ // Encode form data
                'username': username,
                'password': password
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Login failed: ${response.status}`);
        }

        const data = await response.json();
        jwtToken = data.access_token;
        sessionStorage.setItem('authToken', jwtToken); // Store token
        return true; // Login successful

    } catch (error) {
        console.error('Login Error:', error);
        loginError.textContent = error.message;
        loginError.style.display = 'block';
        return false; // Login failed
    }
}

async function fetchUserDetails() {
    if (!jwtToken) return null;
    try {
        const response = await fetch('/users/me', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${jwtToken}`
            }
        });
        if (!response.ok) {
            // Handle expired token or other errors
            if (response.status === 401) logoutUser();
            throw new Error(`Failed to fetch user details: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error("Error fetching user details:", error);
        return null;
    }
}

async function sendMessageToAgent(query) {
    if (!jwtToken) {
        showError("Authentication token missing. Please log in again.");
        logoutUser();
        return;
    }

    loadingIndicator.style.display = 'block';
    errorDisplay.style.display = 'none';
    sendButton.disabled = true;

    try {
        const response = await fetch('/agent/chat', { // Use agent endpoint
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${jwtToken}` // Send JWT token
            },
            body: JSON.stringify({ query: query }), // No session_id needed in body
        });

        loadingIndicator.style.display = 'none';
        sendButton.disabled = false;

        if (!response.ok) {
             if (response.status === 401) { // Unauthorized
                 logoutUser(); // Force logout if token is invalid/expired
                 showError("Session expired. Please log in again.");
                 return; // Stop processing
             }
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        addBotMessageWithSources(data.answer, null); // Sources implicit in agent steps

        // Optional: Log intermediate steps
         if (data.intermediate_steps && data.intermediate_steps.length > 0) {
            console.log("Agent Steps:", data.intermediate_steps);
         }

    } catch (error) {
        console.error('Error:', error);
        showError(`Error: ${error.message}`);
        loadingIndicator.style.display = 'none';
        sendButton.disabled = false;
    }
}


// --- UI Functions ---
function addMessage(message, type) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', type === 'user' ? 'user-message' : 'bot-message');
    const textNode = document.createTextNode(message); // Basic sanitization
    messageDiv.appendChild(textNode);
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function addBotMessageWithSources(answer, sources) { // Keep signature, sources are null now
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot-message');
    const answerNode = document.createElement('p');
    answerNode.appendChild(document.createTextNode(answer)); // Sanitize
    messageDiv.appendChild(answerNode);
    // Optional: Add logic here to display intermediate steps if needed
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function showError(message) {
    errorDisplay.textContent = message;
    errorDisplay.style.display = 'block';
}

function clearChat() {
    chatbox.innerHTML = '';
    errorDisplay.style.display = 'none';
}

function showLogin() {
    loginSection.style.display = 'block';
    chatSection.style.display = 'none';
    jwtToken = null;
    sessionStorage.removeItem('authToken'); // Clear token
}

async function showChat() {
    const user = await fetchUserDetails();
    if (user) {
        welcomeMessage.textContent = `Welcome, ${user.full_name || user.username}!`;
        loginSection.style.display = 'none';
        chatSection.style.display = 'block';
        clearChat(); // Clear previous chat on login
    } else {
        showLogin(); // If token invalid or fetching fails, force login
    }
}

function logoutUser() {
    showLogin();
}

// --- Event Listeners ---
loginForm.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent default form submission
    loginError.style.display = 'none'; // Hide previous errors
    const username = usernameInput.value;
    const password = passwordInput.value;
    const success = await loginUser(username, password);
    if (success) {
        showChat();
    }
});

sendButton.addEventListener('click', () => {
     const query = userInput.value.trim();
        if (!query) return;
        addMessage(query, 'user');
        userInput.value = '';
        sendMessageToAgent(query);
});

userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
         const query = userInput.value.trim();
        if (!query) return;
        addMessage(query, 'user');
        userInput.value = '';
        sendMessageToAgent(query);
    }
});

logoutButton.addEventListener('click', logoutUser);


// --- Initial Check ---
if (jwtToken) {
    showChat(); // If token exists, try to show chat (will verify token)
} else {
    showLogin();
}