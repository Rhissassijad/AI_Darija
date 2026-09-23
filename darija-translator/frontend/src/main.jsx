import React from "react";
import { createRoot } from "react-dom/client";
import { Check, Languages, LoaderCircle, Save, Sparkles } from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function sourceLabel(source) {
  if (source === "dataset_exact") {
    return "Source: dataset exact match";
  }

  if (source === "none") {
    return "Source: waiting for input";
  }

  return "Source: AI guess (rules/model)";
}

function App() {
  const [englishText, setEnglishText] = React.useState("");
  const [translation, setTranslation] = React.useState("");
  const [source, setSource] = React.useState("none");
  const [correction, setCorrection] = React.useState("");
  const [isTranslating, setIsTranslating] = React.useState(false);
  const [isSaving, setIsSaving] = React.useState(false);
  const [message, setMessage] = React.useState("");
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    const text = englishText.trim();
    setMessage("");
    setError("");

    if (!text) {
      setTranslation("");
      setCorrection("");
      setSource("none");
      setIsTranslating(false);
      return;
    }

    const controller = new AbortController();
    setIsTranslating(true);

    const timeout = window.setTimeout(async () => {
      try {
        const response = await fetch(`${API_URL}/translate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error("Translation request failed.");
        }

        const data = await response.json();
        setTranslation(data.translation);
        setCorrection(data.translation);
        setSource(data.source);
      } catch (requestError) {
        if (requestError.name !== "AbortError") {
          setError("Could not reach the translation API. Make sure the backend is running.");
          setTranslation("");
          setCorrection("");
          setSource("none");
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsTranslating(false);
        }
      }
    }, 250);

    return () => {
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [englishText]);

  async function handleSaveCorrection(event) {
    event.preventDefault();
    setMessage("");
    setError("");
    setIsSaving(true);

    try {
      const response = await fetch(`${API_URL}/corrections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          english: englishText,
          darija: correction,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not save the correction.");
      }

      setMessage(data.message);
      setSource("dataset_exact");
      setTranslation(correction.trim());
    } catch (saveError) {
      setError(saveError.message);
    } finally {
      setIsSaving(false);
    }
  }

  const hasInput = englishText.trim().length > 0;
  const canSave = hasInput && correction.trim().length > 0 && !isSaving;

  return (
    <main className="app-shell">
      <section className="translator-panel" aria-labelledby="app-title">
        <div className="title-row">
          <div className="brand-mark" aria-hidden="true">
            <Languages size={28} />
          </div>
          <div>
            <p className="eyebrow">Live translation mode</p>
            <h1 id="app-title">English to Moroccan Darija Translator</h1>
          </div>
        </div>

        <label className="field-label" htmlFor="english-input">
          Enter a sentence in English
        </label>
        <textarea
          id="english-input"
          className="text-input"
          value={englishText}
          onChange={(event) => setEnglishText(event.target.value)}
          placeholder="Example: How are you?"
          rows={4}
        />

        <div className="result-header">
          <div className="result-title">
            <Sparkles size={18} aria-hidden="true" />
            <span>Darija translation</span>
          </div>
          {isTranslating && (
            <span className="status-inline">
              <LoaderCircle size={16} className="spin" aria-hidden="true" />
              Translating
            </span>
          )}
        </div>

        <div className="result-box" aria-live="polite">
          {hasInput ? translation || "..." : "Start typing to see a live translation."}
        </div>
        <p className="source-label">{sourceLabel(source)}</p>

        <form className="correction-form" onSubmit={handleSaveCorrection}>
          <label className="field-label" htmlFor="correction-input">
            Improve this translation
          </label>
          <textarea
            id="correction-input"
            className="text-input correction-input"
            value={correction}
            onChange={(event) => setCorrection(event.target.value)}
            disabled={!hasInput}
            placeholder="Write the correct Darija translation here."
            rows={3}
          />
          <button className="save-button" type="submit" disabled={!canSave}>
            {isSaving ? (
              <LoaderCircle size={18} className="spin" aria-hidden="true" />
            ) : (
              <Save size={18} aria-hidden="true" />
            )}
            Save to dataset
          </button>
        </form>

        {message && (
          <p className="feedback success">
            <Check size={18} aria-hidden="true" />
            {message}
          </p>
        )}
        {error && <p className="feedback error">{error}</p>}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
