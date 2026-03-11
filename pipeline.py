from pathlib import Path

from pdf2image import convert_from_path
import ollama


class OCRPipeline:
    """
    OCR pipeline for PDF-to-image conversion and image text extraction.
    """

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        image_output_dir: str = "images",
        text_output_dir: str = "extracted_texts",
    ):
        self.model = model
        self.image_output_dir = Path(image_output_dir)
        self.text_output_dir = Path(text_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        self.text_output_dir.mkdir(parents=True, exist_ok=True)

    def convert_pdf_to_images(self, pdf_path: str, dpi: int = 300) -> list[str]:
        """
        Convert a PDF file to PNG images and return saved file paths.
        """
        pdf_file = Path(pdf_path).expanduser()
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_file}")

        images = convert_from_path(str(pdf_file), dpi=dpi)
        if not images:
            raise ValueError("No images were extracted from the PDF file.")

        file_paths: list[str] = []
        for i, image in enumerate(images, start=1):
            output_path = self.image_output_dir / f"{pdf_file.stem}_page_{i}.png"
            image.save(output_path, "PNG")
            file_paths.append(str(output_path))

        return file_paths

    def process_image(
        self,
        image_path: str,
        prompt: str,
        max_output_tokens: int | None = 300,
        think: bool = False,
    ) -> dict[str, object]:
        """
        Process an image and return final model output plus runtime metrics.
        """
        image_file = Path(image_path.strip()).expanduser()
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_file}")

        chat_options = {}
        # Comment out this block if you want the model to generate without a token cap.
        if max_output_tokens is not None:
            chat_options["num_predict"] = max_output_tokens

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OCR assistant. Extract text from the image and "
                        "provide it in a clear format."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_file)],
                },
            ],
            think=think,
            options=chat_options or None,
        )

        return {
            "final_output": response["message"]["content"],
            "done_reason": response.get("done_reason"),
            "total_duration": response.get("total_duration"),
            "load_duration": response.get("load_duration"),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "eval_count": response.get("eval_count"),
        }

    def save_output_text(
        self,
        image_path: str,
        result: dict[str, object] | str,
        output_dir: str | None = None,
    ) -> str:
        """
        Save OCR output to a .txt file named after the source image.
        """
        image_file = Path(image_path.strip()).expanduser()
        target_dir = Path(output_dir).expanduser() if output_dir else self.text_output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(result, dict):
            text = str(result.get("final_output", ""))
        else:
            text = str(result)

        output_path = target_dir / f"{image_file.stem}.txt"
        output_path.write_text(text, encoding="utf-8")
        return str(output_path)


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    """
    Backward-compatible helper for notebook-style module calls.
    """
    return OCRPipeline().convert_pdf_to_images(pdf_path, dpi=dpi)
