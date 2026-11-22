"""CLI entry point to regenerate the root README.md project gallery."""

from __future__ import annotations

from preview_pipeline import generate_root_readme


def main() -> None:
    readme_path = generate_root_readme()
    print(f"Saved: {readme_path}")


if __name__ == "__main__":
    main()
