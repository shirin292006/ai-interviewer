# AI Interviewer

## 1️⃣ Prerequisites

- Install Python (3.9 or above)
- Install Node.js (v18+ recommended)
- Install npm
- Ensure you have a valid GROQ API key

---

## 2️⃣ Project Structure

The project contains two main folders:

- backend → FastAPI application
- Interview-Coach → Frontend (React + Express)

---

## 3️⃣ Backend Setup

- Open a terminal
- Navigate to the backend folder
- Install dependencies from the requirements file
- Set your GROQ_API_KEY as an environment variable (or create a .env file inside backend with the key)
- Start the backend using:

*unicorn mainver2:app –reload*

Once running:

- Backend will be available at http://localhost:8000
- Swagger documentation will be available at http://localhost:8000/docs

Keep this terminal running.

---

## 4️⃣ Frontend Setup

- Open a new terminal window
- Navigate to the Interview-Coach folder
- Install frontend dependencies using npm
- Start the frontend using:

*npm run dev*

Once running:

- Frontend will be available at http://localhost:5000

Keep this terminal running.

---

## 5️⃣ Important Configuration

- Ensure the frontend is configured to call the backend at:
  http://localhost:8000
- Both backend and frontend must be running at the same time.

---

## 6️⃣ Running the Application

- Start the backend first
- Start the frontend second
- Open http://localhost:5000 in your browser
- Begin the interview flow through the UI

---

## 7️⃣ Troubleshooting

- If the frontend cannot connect → verify backend is running
- If CORS errors appear → ensure CORS middleware is enabled in the backend
- If evaluation fails → confirm GROQ_API_KEY is correctly set