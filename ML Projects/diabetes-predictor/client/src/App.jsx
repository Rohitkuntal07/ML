import React, { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    pregnancies: "",
    glucose: "",
    bloodPressure: "",
    skinThickness: "",
    insulin: "",
    bmi: "",
    pedigree: "",
    age: "",
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const payload = {
      Pregnancies: Number(formData.pregnancies),
      Glucose: Number(formData.glucose),
      BloodPressure: Number(formData.bloodPressure),
      SkinThickness: Number(formData.skinThickness),
      Insulin: Number(formData.insulin),
      BMI: Number(formData.bmi),
      DiabetesPedigreeFunction: Number(formData.pedigree),
      Age: Number(formData.age),
    };

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      setResult({
        status: data.label,
        probability: (data.probability * 100).toFixed(2) + "%",
        model: data.model,
      });
    } catch (error) {
      console.error("Prediction Error:", error);
      alert("Server Error — Make sure API is running!");
    }
  };

  return (
    <div className="app">
      <div className="center-block">

        <header className="header">
          <h1>🩺 Diabetes Risk Predictor</h1>
          <p className="subtitle">
            Enter patient details to get a quick ML-based risk estimate
          </p>
        </header>

        <main className="card">
          <form onSubmit={handleSubmit} className="form">
            {Object.keys(formData).map((field) => (
              <div className="form-group" key={field}>
                <label>
                  {field.charAt(0).toUpperCase() + field.slice(1)}
                </label>
                <input
                  type="number"
                  name={field}
                  value={formData[field]}
                  onChange={handleChange}
                  required
                />
              </div>
            ))}
            <button type="submit" className="btn">Predict</button>
          </form>

          {result && (
            <div className="result">
              <h2>Prediction Result</h2>
              <p><strong>Status:</strong> {result.status}</p>
              <p><strong>Probability:</strong> {result.probability}</p>
              <p><strong>Model:</strong> {result.model}</p>
            </div>
          )}
        </main>

        <footer className="footer">
          <p>Made with ❤️ by Rohit | Powered by ML</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
