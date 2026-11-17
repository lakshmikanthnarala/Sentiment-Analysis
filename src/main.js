import "./style.css";

const form = document.querySelector("#sentiment-form");
const textInput = document.querySelector("#text-input");
const characterCount = document.querySelector("#character-count");
const analyzeButton = document.querySelector("#analyze-button");
const status = document.querySelector("#status");
const result = document.querySelector("#result");
const prediction = document.querySelector("#prediction");
const confidence = document.querySelector("#confidence");

let classifierPromise;

function getClassifier() {
  if (!classifierPromise) {
    classifierPromise = import("@huggingface/transformers")
      .then(({ pipeline }) =>
        pipeline(
          "sentiment-analysis",
          "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
        ),
      )
      .catch((error) => {
        classifierPromise = undefined;
        throw error;
      });
  }
  return classifierPromise;
}

textInput.addEventListener("input", () => {
  characterCount.textContent = `${textInput.value.length} / 5000`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = textInput.value.trim();
  if (!text) {
    textInput.focus();
    return;
  }

  analyzeButton.disabled = true;
  result.hidden = true;
  status.textContent = "Loading the model and analyzing your text…";

  try {
    const classifier = await getClassifier();
    const [output] = await classifier(text);
    if (!output || typeof output.label !== "string" || typeof output.score !== "number") {
      throw new Error("The model returned an unexpected result.");
    }

    prediction.textContent = output.label === "POSITIVE" ? "Positive" : "Negative";
    confidence.textContent = `${Math.round(output.score * 100)}% confidence`;
    result.hidden = false;
    status.textContent = "Analysis complete.";
  } catch (error) {
    console.error("Sentiment analysis failed:", error);
    status.textContent =
      "Could not load the model or analyze this text. Check your internet connection and try again.";
  } finally {
    analyzeButton.disabled = false;
  }
});
