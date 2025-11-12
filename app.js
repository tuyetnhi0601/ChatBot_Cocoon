const API = "http://127.0.0.1:8000/chat"; // backend FastAPI

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");

function addMsg(text, who = "bot") {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.innerHTML = text.replace(/\*\*(.*?)\*\*/g, "<b>$1</b>");
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMsg() {
  const text = inputEl.value.trim();
  if (!text) return;
  addMsg(text, "user");
  inputEl.value = "";

  try {
    const res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }) // ✅ phải là 'message'
    });
    const data = await res.json();
    addMsg(data.reply || "Mình đang xử lý nè...");
  } catch (e) {
    addMsg("Lỗi kết nối API, kiểm tra server FastAPI đang chạy chưa nhé 💻", "bot");
  }
}

sendBtn.addEventListener("click", sendMsg);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMsg();
});

addMsg("Chào bạn 🌿 Mình là CocoBot. Hỏi mình về loại da hoặc sản phẩm Cocoon nha!");
