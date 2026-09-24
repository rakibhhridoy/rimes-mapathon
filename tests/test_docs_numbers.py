"""
Tests for docs/build_numbers.py.

The document must take every figure from pipeline output. A missing figure has
to be visible in the PDF as a TODO, never silently absent and never carried
over from an older draft.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))

from build_numbers import _macro, _pct, build, region_numbers  # noqa: E402


class TestMacroFormatting:
    def test_missing_value_becomes_a_visible_todo(self):
        assert r"\todo{missing}" in _macro("SomeFigure", None)

    def test_integers_get_thousands_separators(self):
        assert _macro("Assets", 22111).endswith("{22,111}")

    def test_floats_are_trimmed_not_padded(self):
        assert _macro("Auc", 0.8539).endswith("{0.854}")
        assert _macro("Round", 2.0).endswith("{2}")

    def test_percentages_convert_from_fractions(self):
        assert _pct(0.00750) == 0.8
        assert _pct(None) is None


class TestRegionNumbers:
    def test_processed_region_reports_its_validation_auc(self):
        lines = "\n".join(region_numbers("rangpur_rajshahi"))
        assert r"\newcommand{\RRValAUC}" in lines
        assert r"\RRValAUC}{\todo" not in lines, "AUC should come from metadata"

    def test_unprocessed_region_reports_todos_rather_than_zeros(self, tmp_path, monkeypatch):
        # Point a region at an empty directory so it has no outputs at all.
        import build_numbers

        empty = {kind: tmp_path / kind for kind in ("raw", "processed", "output", "cache")}
        monkeypatch.setattr(build_numbers, "region_paths", lambda region: empty)
        lines = "\n".join(build_numbers.region_numbers("sw_coastal"))
        assert r"\newcommand{\CoastalAssets}{\todo{missing}}" in lines
        assert r"\newcommand{\CoastalValAUC}{\todo{missing}}" in lines

    def test_admin_counts_distinguish_scored_from_total(self):
        lines = "\n".join(region_numbers("rangpur_rajshahi"))
        assert r"\newcommand{\RRNUnions}" in lines
        assert r"\newcommand{\RRNUnionsScored}" in lines


class TestBuild:
    def test_build_writes_macros_and_defines_todo(self, tmp_path):
        out = build(tmp_path / "numbers.tex")
        text = out.read_text()
        assert r"\providecommand{\todo}" in text
        assert text.count(r"\newcommand") > 50
        # Macro names must be letters only or LaTeX will reject them.
        import re
        for name in re.findall(r"\\newcommand\{\\(\w+)\}", text):
            assert name.isalpha(), f"invalid LaTeX macro name: {name}"


class TestDocumentCitesDefinedMacros:
    """Every macro the document uses must be generated, and every figure it
    includes must exist. A typo in a macro name otherwise renders as nothing
    at all, which is worse than a visible TODO."""

    def _document_macros(self):
        import re

        docs = ROOT / "docs"
        text = (docs / "sgmdi_technical_document.tex").read_text()
        text += (docs / "abstract.tex").read_text()
        body = re.sub(r"%.*", "", text)
        used = set(re.findall(r"\\([A-Z][A-Za-z]*)\{\}", body))
        defined = set(re.findall(r"\\newcommand\{\\(\w+)\}",
                                 (docs / "numbers.tex").read_text()))
        return used, defined

    def test_every_macro_used_is_defined(self):
        used, defined = self._document_macros()
        latex_builtins = {"LaTeX", "TeX"}
        missing = sorted(used - defined - latex_builtins)
        assert not missing, f"undefined macros: {missing}"

    def test_every_included_figure_exists(self):
        import re

        docs = ROOT / "docs"
        text = (docs / "sgmdi_technical_document.tex").read_text()
        for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            assert (docs / path).exists(), f"missing figure: {path}"


class TestPValues:
    def test_small_p_values_read_as_a_bound(self):
        from build_numbers import _p
        assert _p(0.0) == "<0.001"
        assert _p(0.0004) == "<0.001"

    def test_other_p_values_carry_their_equals_sign(self):
        from build_numbers import _p
        assert _p(0.215) == "=0.215"
        assert _p(0.08) == "=0.08"
        assert _p(None) is None
