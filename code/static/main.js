document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const promptView = document.getElementById('prompt-view');
    const chatView = document.getElementById('chat-view');
    const chatMessages = document.querySelector('.chat-messages');
    const promptForm = document.querySelector('.chat-form');
    const promptQuestionInput = promptForm.querySelector('textarea[name="question"]');
    const categoryButtonsContainer = document.querySelector('.category-buttons');
    const suggestedQuestionsContainer = document.querySelector('.suggested-questions');
    const chatFormBottom = document.querySelector('.chat-form-bottom');
    const clearButton = document.querySelector('.clear-button');

    // --- Data from HTML ---
    const configElement = document.getElementById('js-config');
    const hasHistory = JSON.parse(configElement.dataset.hasHistory);
    const chatHistory = JSON.parse(configElement.dataset.chatHistory);
    const questionsByCat = JSON.parse(configElement.dataset.questionsByCategory);
    const chatUrl = configElement.dataset.chatUrl;
    const clearUrl = configElement.dataset.clearUrl;

    // --- Functions ---

    function showPromptView() {
        promptView.style.display = 'flex';
        chatView.style.display = 'none';
        displayCategoriesAndQuestions();
    }

    function showChatView(isFirstMessage = true, question = '') {
        promptView.style.display = 'none';
        chatView.style.display = 'flex';
        if (isFirstMessage && question) {
            chatMessages.innerHTML = '';
            appendMessage(question, 'user-message');
        }
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function displayCategoriesAndQuestions() {
        categoryButtonsContainer.innerHTML = '';
        suggestedQuestionsContainer.innerHTML = '';

        if (Object.keys(questionsByCat).length > 0) {
            Object.keys(questionsByCat).forEach(category => {
                const button = document.createElement('button');
                button.textContent = category;
                button.className = 'category-button';
                button.onclick = () => displayQuestionsForCategory(category);
                categoryButtonsContainer.appendChild(button);
            });

            // Initially, no category is selected, and no questions are shown.
            suggestedQuestionsContainer.innerHTML = '';
        }
    }

    function displayQuestionsForCategory(category) {
        const alreadyActive = document.querySelector(`.category-button.active`);
        const clickedButton = Array.from(document.querySelectorAll('.category-button')).find(btn => btn.textContent === category);

        // If the clicked button is already active, deactivate it and clear questions.
        if (alreadyActive && alreadyActive === clickedButton) {
            alreadyActive.classList.remove('active');
            suggestedQuestionsContainer.innerHTML = '';
            return;
        }

        // Deactivate the currently active button.
        if (alreadyActive) {
            alreadyActive.classList.remove('active');
        }

        // Activate the new button and show its questions.
        if (clickedButton) {
            clickedButton.classList.add('active');
        }

        suggestedQuestionsContainer.innerHTML = '';
        const questions = questionsByCat[category];
        if (questions) {
            questions.forEach(q => {
                const button = document.createElement('button');
                button.textContent = q;
                button.className = 'suggested-question';
                button.onclick = () => {
                    promptQuestionInput.value = q;
                    promptForm.dispatchEvent(new Event('submit', { cancelable: true }));
                };
                suggestedQuestionsContainer.appendChild(button);
            });
        }
    }

    async function handleFormSubmit(event) {
        event.preventDefault();
        const form = event.target;
        const input = form.querySelector('textarea');
        const question = input.value.trim();
        if (!question) return;

        const isFirstSubmit = !chatView.style.display || chatView.style.display === 'none';
        if (isFirstSubmit) {
            showChatView(true, question);
        } else {
            appendMessage(question, 'user-message');
        }
        
        input.value = '';
        autoResize(input);

        const aiMessageContainer = appendMessage('', 'ai-message', true);
        const answerTextElement = aiMessageContainer.querySelector('.answer-text');

        try {
            const response = await fetch(chatUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.json();
            if (data.answer && data.answer.trim() !== '') {
                await typeWriter(answerTextElement, data.answer);
            } else {
                // If the answer is empty, add a class to hide the bubble via CSS
                aiMessageContainer.closest('.message').classList.add('empty');
            }

        } catch (error) {
            console.error('Error:', error);
            answerTextElement.innerHTML = "<p>申し訳ありません、エラーが発生しました。</p>";
        } finally {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    function appendMessage(content, type, isTyping = false) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `message ${type}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const p = document.createElement('p');
        p.className = 'answer-text';
        if (isTyping) {
            p.innerHTML = '<span class="cursor"></span>';
        } else {
            p.innerHTML = content;
        }
        
        contentDiv.appendChild(p);
        messageWrapper.appendChild(contentDiv);
        chatMessages.appendChild(messageWrapper);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return contentDiv;
    }

    async function typeWriter(element, text, speed = 50) {
        element.innerHTML = '';
        const cursor = document.createElement('span');
        cursor.className = 'cursor';
        element.appendChild(cursor);

        for (let i = 0; i < text.length; i++) {
            await new Promise(resolve => setTimeout(resolve, speed));
            cursor.insertAdjacentText('beforebegin', text.charAt(i));
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        cursor.style.display = 'none';
    }

    // --- Event Listeners ---
    promptForm.addEventListener('submit', handleFormSubmit);
    chatFormBottom.addEventListener('submit', handleFormSubmit);

    clearButton.addEventListener('click', async () => {
        try {
            const response = await fetch(clearUrl, { method: 'POST' });
            if (response.ok) {
                chatMessages.innerHTML = '';
                showPromptView();
            } else {
                throw new Error('Failed to clear chat history.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('チャット履歴のクリアに失敗しました。');
        }
    });

    // --- Initial State ---
    if (hasHistory) {
        chatHistory.forEach(chat => {
            appendMessage(chat.question, 'user-message');
            const aiMessageContainer = appendMessage('', 'ai-message', false);
            const answerTextElement = aiMessageContainer.querySelector('.answer-text');
            answerTextElement.innerHTML = chat.answer;
        });
        showChatView(false);
    } else {
        showPromptView();
    }
});

// --- Global Helper Functions (can be called from HTML) ---
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        const form = event.target.closest('form');
        form.dispatchEvent(new Event('submit', { cancelable: true }));
    }
}