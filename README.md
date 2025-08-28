# 🍴 Restaurant Name & Menu Generator

A fun **Streamlit + LangChain + Ollama** project that generates fancy restaurant names and creative menu items for any cuisine.

---

## 🚀 Features
- Generates unique restaurant names
- Creates 8–10 creative menu items
- Powered by **LangChain** and **Ollama (Mistral model)**
- Interactive UI built with **Streamlit**

---

## 🖥️ How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone https://github.com/RajShah03/restaurant-name-generator.git
   cd restaurant-name-generator
Create virtual environment & install dependencies:


python -m venv .venv
.venv\Scripts\activate   # (Windows)
pip install -r requirements.txt
Start Streamlit app:


streamlit run main.py
📌 Notes
You need Ollama installed locally.

The default model is mistral, but you can change it in langchain_helper.py.

💡 Example Output

Restaurant Name: Rasoi Royale
Menu Items:
- Coconut Shrimp Kebabs
- Spiced Falafel Platter
- Royal Biryani
- Gulab Jamun