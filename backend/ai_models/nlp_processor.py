# --- nlp_processor.py ---
import spacy
import os # To handle potential model loading errors gracefully

# --- Load the spaCy model ---
# It's good practice to load the model once when the module is imported
# rather than inside the function, for efficiency.
NLP_MODEL_NAME = "en_core_web_sm"
nlp = None
try:
    nlp = spacy.load(NLP_MODEL_NAME)
    print(f"Successfully loaded spaCy model '{NLP_MODEL_NAME}'")
except OSError:
    print(f"Could not find spaCy model '{NLP_MODEL_NAME}'.")
    print(f"Please run: python -m spacy download {NLP_MODEL_NAME}")
    # You might want to raise an exception or handle this more robustly
    # depending on whether the NLP model is critical for startup.
    # For now, we'll let it proceed, but functions using 'nlp' will fail.


def process_text(text: str) -> dict:
    """
    Processes the input text using spaCy for basic NLP tasks.

    Args:
        text: The input string from the user.

    Returns:
        A dictionary containing processed information (e.g., tokens).
        Returns an error message if the model wasn't loaded.
    """
    if not nlp:
        return {"error": f"spaCy model '{NLP_MODEL_NAME}' not loaded."}

    doc = nlp(text)

    # --- Extract basic information ---
    tokens = [token.text for token in doc]
    num_tokens = len(tokens)

    # You can extract much more here later:
    # entities = [(ent.text, ent.label_) for ent in doc.ents]
    # pos_tags = [(token.text, token.pos_) for token in doc]
    # lemmas = [token.lemma_ for token in doc]

    # --- Prepare results ---
    results = {
        "original_text": text,
        "num_tokens": num_tokens,
        "tokens": tokens,
        # "entities": entities, # Uncomment when needed
        # "pos_tags": pos_tags, # Uncomment when needed
    }

    return results