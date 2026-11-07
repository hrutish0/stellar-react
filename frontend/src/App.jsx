import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const FIELDS = [
  { key: "alpha", label: "Right Ascension (alpha)", step: "any" },
  { key: "delta", label: "Declination (delta)", step: "any" },
  { key: "i", label: "Near-infrared filter (i)", step: "any" },
  { key: "z", label: "Infrared filter (z)", step: "any" },
  { key: "cam_col", label: "Camera column (1-6)", step: 1, min: 1, max: 6 },
  { key: "field_ID", label: "Field ID", step: 1 },
  { key: "plate", label: "Plate ID", step: 1 },
  { key: "MJD", label: "Modified Julian Date", step: 1 },
  { key: "spec_obj_ID", label: "Spectroscopic object ID", step: 1 },
];

const EMPTY = Object.fromEntries(FIELDS.map((f) => [f.key, ""]));

const LABELS = { GALAXY: "Galaxy", QSO: "Quasar", STAR: "Star" };

export default function App() {
  const [form, setForm] = useState(EMPTY);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onChange = (key) => (e) => setForm({ ...form, [key]: e.target.value });

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const body = Object.fromEntries(
        Object.entries(form).map(([k, v]) => [k, Number(v)])
      );
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error("Request failed");
      setResult(await res.json());
    } catch {
      setError("Could not reach the API. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const complete = Object.values(form).every((v) => v !== "");

  return (
    <main className="app">
      <h1>Stellar Object Classifier</h1>
      <p className="sub">
        Predict whether an SDSS observation is a star, galaxy or quasar.
      </p>

      <form onSubmit={onSubmit} className="card">
        <div className="grid">
          {FIELDS.map((f) => (
            <label key={f.key}>
              <span>{f.label}</span>
              <input
                type="number"
                step={f.step}
                min={f.min}
                max={f.max}
                value={form[f.key]}
                onChange={onChange(f.key)}
                required
              />
            </label>
          ))}
        </div>
        <div className="actions">
          <button type="submit" disabled={!complete || loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
          <button type="button" className="ghost" onClick={() => { setForm(EMPTY); setResult(null); setError(""); }}>
            Clear
          </button>
        </div>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="card result">
          <h2>
            {LABELS[result.prediction]}{" "}
            <small>{(result.confidence * 100).toFixed(1)}% confidence</small>
          </h2>
          {Object.entries(result.probabilities).map(([cls, p]) => (
            <div className="bar-row" key={cls}>
              <span>{LABELS[cls]}</span>
              <div className="bar"><div style={{ width: `${p * 100}%` }} /></div>
              <span>{(p * 100).toFixed(1)}%</span>
            </div>
          ))}
          {result.warning && <p className="error">{result.warning}</p>}
        </section>
      )}
    </main>
  );
}
