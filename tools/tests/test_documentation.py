"""Check the maintained documentation index and local Markdown links."""
import re
import unittest

from echoguard.paths import REPO_ROOT


class DocumentationTests(unittest.TestCase):
    def test_local_documentation_links_resolve(self):
        documents = [REPO_ROOT / "README.md",
                     *(REPO_ROOT / "docs").glob("*.md")]
        for document in documents:
            text = document.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                relative = target.split("#", 1)[0]
                destination = document.parent / relative
                self.assertTrue(destination.is_file(), f"{document}: {target}")
                if "#" in target:
                    anchor = target.split("#", 1)[1]
                    headings = re.findall(r"^#{1,6}\s+(.+)$", destination.read_text(encoding="utf-8"), re.MULTILINE)
                    anchors = {re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-") for heading in headings}
                    self.assertIn(anchor, anchors, f"{document}: {target}")

    def test_index_covers_current_guides(self):
        index = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual({guide.name for guide in (REPO_ROOT / "docs").glob("*.md")},
                         {"guide.md", "evaluation.md"})
        for guide in (REPO_ROOT / "docs").glob("*.md"):
            self.assertIn(f"docs/{guide.name}", index)

    def test_readme_metrics_are_scoped_and_docs_have_no_run_dates(self):
        index = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for value in ("92.3%", "85.7%", "0.092", "527 ms", "358 MiB", "~75%"):
            self.assertIn(value, index)
        self.assertIn("supplied-text", index)
        self.assertIn("acoustic inference", index)
        self.assertIn("[![Live Demo](https://img.shields.io/", index)
        self.assertIn("Microphone audio", index)
        guide = (REPO_ROOT / "docs/guide.md").read_text(encoding="utf-8")
        evaluation = (REPO_ROOT / "docs/evaluation.md").read_text(encoding="utf-8")
        self.assertIn("docs/guide.md#web-demo", index)
        self.assertIn("docs/evaluation.md#run-evaluations", index)
        self.assertNotIn("docker build -f", index)
        self.assertIn("docker build -f webdemo/Dockerfile", guide)
        self.assertIn("## Project structure", guide)
        self.assertNotIn("| `app/` |", index)
        self.assertIn("python tools/evaluation/run_eval.py --repo", evaluation)
        for document in [REPO_ROOT / "README.md", *(REPO_ROOT / "docs").glob("*.md")]:
            text = document.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"\b\d{4}-?\d{2}-?\d{2}\b")
            self.assertNotRegex(text, r"(?i)\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b")


if __name__ == "__main__":
    unittest.main()
