// api/server.js
import express from "express";
import cors from "cors";
import { spawn } from "child_process";

const app = express();
app.use(cors());
app.use(express.json());

app.get("/", (_req, res) => res.json({ ok: true, service: "diabetes-api" }));

app.post("/predict", (req, res) => {
  // Basic input guard; Python will do the real work
  const payload = req.body || {};

  const py = spawn(process.platform.startsWith("win") ? "python" : "python3", ["ml/predict.py"], {
    cwd: process.cwd(), // run from repo root
  });

  let out = "", err = "";
  py.stdout.on("data", (d) => (out += d.toString()));
  py.stderr.on("data", (d) => (err += d.toString()));

  py.on("close", (code) => {
    if (code !== 0 || err) {
      return res.status(500).json({ error: "Prediction failed", detail: err, code });
    }
    try {
      return res.json(JSON.parse(out));
    } catch (e) {
      return res.status(500).json({ error: "Bad JSON from predictor", detail: out });
    }
  });

  py.stdin.write(JSON.stringify(payload));
  py.stdin.end();
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`API on http://localhost:${PORT}`));
