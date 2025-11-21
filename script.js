const inputField = document.getElementById("input");
const sendButton = document.getElementById("button");
const chatBox = document.getElementById("chat");

function addMessage(text, sender) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", sender);

  const textDiv = document.createElement("div");
  textDiv.classList.add("text");
  textDiv.innerText = text;

  messageDiv.appendChild(textDiv);
  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

sendButton.addEventListener("click", async () => {
  const userText = inputField.value.trim();
  if (!userText) return;

  addMessage(userText, "user");
  inputField.value = "";

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: userText }),
    });

    const data = await response.json();

    if (data.sentiment) {
      addMessage("Sentiment: " + data.sentiment, "bot");
    } else {
      addMessage("Error: " + (data.error || "Something went wrong"), "bot");
    }
  } catch (err) {
    addMessage("⚠️ Server not responding", "bot");
  }
});

inputField.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendButton.click();
});
