import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


def handle_missing_text(text: Any) -> str:
    """Return an empty string for None; convert other values to strings."""
    if text is None:
        return ""
    return str(text)


def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalization."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Replace repeated whitespace with one space and trim the result."""
    return re.sub(r"\s+", " ", text).strip()


def replace_urls(text: str) -> str:
    """Replace HTTP/HTTPS URLs with <URL>, preserving terminal punctuation."""

    def replace_match(match: re.Match) -> str:
        url = match.group(0)
        trailing_punctuation = ""

        while url and url[-1] in ".!?":
            trailing_punctuation = url[-1] + trailing_punctuation
            url = url[:-1]

        if trailing_punctuation:
            return f"<URL> {trailing_punctuation}"
        return "<URL>"

    return re.sub(r"https?://\S+", replace_match, text, flags=re.IGNORECASE)


def normalize_hashtags(text: str) -> str:
    """Convert #word to <HASHTAG> word while preserving the hashtag word."""
    return re.sub(r"#([^\s#]+)", r"<HASHTAG> \1", text)


def limit_repeated_punctuation(text: str, maximum: int = 3) -> str:
    """Limit consecutive runs of !, ?, and . to the requested maximum."""
    if maximum < 1:
        raise ValueError("maximum must be at least 1")

    pattern = rf"([.!?])\1{{{maximum},}}"
    return re.sub(pattern, lambda match: match.group(1) * maximum, text)


def clean_text(text: Any) -> str:
    """Apply the complete text-cleaning pipeline in a consistent order."""
    text = handle_missing_text(text)
    if not text:
        return ""

    text = normalize_unicode(text)
    text = replace_urls(text)
    text = normalize_hashtags(text)
    text = limit_repeated_punctuation(text)
    text = normalize_whitespace(text)
    return text


def clean_text_collection(texts: Iterable[Any]) -> list[str]:
    """Clean every item in a collection."""
    return [clean_text(text) for text in texts]


def remove_exact_duplicates(texts: Iterable[str]) -> list[str]:
    """Remove exact duplicates while preserving original order."""
    seen = set()
    unique_texts = []

    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    return unique_texts


def read_text_file(file_path: str | Path) -> list[str]:
    """Read a UTF-8 text file and return its lines without line endings."""
    file_path = Path(file_path)

    try:
        with file_path.open("r", encoding="utf-8-sig") as file:
            return [line.rstrip("\r\n") for line in file]
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except UnicodeDecodeError:
        print(f"Error: Unable to decode file - {file_path}")
    except OSError as error:
        print(f"Error: Unable to read file - {file_path}. Details: {error}")

    return []


def write_text_file(texts: Iterable[str], file_path: str | Path) -> bool:
    """Write texts to a UTF-8 file, one text per line."""
    file_path = Path(file_path)

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        text_list = list(texts)

        with file_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write("\n".join(text_list))
            if text_list:
                file.write("\n")
        return True
    except OSError as error:
        print(f"Error: Unable to write file - {file_path}. Details: {error}")
        return False


def main() -> None:
    """Run the sample file-cleaning pipeline."""
    project_directory = Path(__file__).resolve().parent.parent
    input_path = project_directory / "data" / "sample_raw_texts.txt"
    output_path = project_directory / "data" / "sample_cleaned_texts.txt"

    raw_texts = read_text_file(input_path)
    if not raw_texts:
        print("No texts were loaded. The program will stop.")
        return

    cleaned_texts = clean_text_collection(raw_texts)
    unique_texts = remove_exact_duplicates(cleaned_texts)
    writing_succeeded = write_text_file(unique_texts, output_path)

    for raw, cleaned in zip(raw_texts, cleaned_texts):
        print("RAW:    ", raw)
        print("CLEANED:", cleaned)
        print("-" * 50)

    if writing_succeeded:
        print(f"Cleaned texts saved to: {output_path}")


if __name__ == "__main__":
    main()