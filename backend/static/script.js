const chatbox = document.getElementById('chatbox');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const loadingIndicator = document.getElementById('loading');
const errorDisplay = document.getElementById('error');

// Use a simple session ID stored in the browser's session storage
let sessionId = sessionStorage.getItem('ragSessionId');
if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
    sessionStorage.setItem('ragSessionId', sessionId);
}

console.log("Using Session ID:", sessionId);

function addMessage(message, type) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', type === 'user' ? 'user-message' : 'bot-message');

    // Sanitize message content before adding to DOM (basic example)
    const textNode = document.createTextNode(message);
    messageDiv.appendChild(textNode);

    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight; // Scroll to bottom
}

function addBotMessageWithSources(answer, sources) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot-message');

    // Sanitize and add answer
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
            // Sanitize source info
            const sourceName = source.source ? document.createTextNode(`[${index+1}] ${source.source}: `) : document.createTextNode(`[${index+1}] Unknown Source: `);
            const sourceContent = document.createTextNode(source.content.substring(0, 150) + '...'); // Show snippet
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

    addMessage(query, 'user');
    userInput.value = '';
    loadingIndicator.style.display = 'block';
    errorDisplay.style.display = 'none'; // Hide previous errors
    sendButton.disabled = true;

    try {
        const response = await fetch('/chat', { // Assumes API is served on the same origin
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query, session_id: sessionId }),
        });

        loadingIndicator.style.display = 'none';
        sendButton.disabled = false;

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        addBotMessageWithSources(data.answer, data.sources);

    } catch (error) {
        console.error('Error:', error);
        loadingIndicator.style.display = 'none';
        errorDisplay.textContent = `Error: ${error.message}`;
        errorDisplay.style.display = 'block';
        sendButton.disabled = false;
    }
}

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Initial welcome message (optional)
// addMessage("Hello! Ask me anything about our documents.", 'bot');