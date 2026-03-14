"""Tests for the assign_is_tps_label helper in experiment_runner."""

from enzymeexplorer.src.experiments_orchestration.experiment_runner import (
    assign_is_tps_label,
)

FPP = "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
GPP = "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"


class TestAssignIsTpsLabel:
    """Edge cases for assign_is_tps_label."""

    def test_unknown_only(self):
        result = assign_is_tps_label({"Unknown"})
        assert "isTPS" not in result
        assert result == {"Unknown"}

    def test_precursor_substr_only(self):
        result = assign_is_tps_label({"precursor substr"})
        assert "isTPS" not in result
        assert result == {"precursor substr"}

    def test_other_only(self):
        result = assign_is_tps_label({"other"})
        assert "isTPS" not in result
        assert result == {"other"}

    def test_unknown_and_precursor(self):
        result = assign_is_tps_label({"Unknown", "precursor substr"})
        assert "isTPS" not in result

    def test_all_non_tps_sentinels(self):
        result = assign_is_tps_label({"Unknown", "precursor substr", "other"})
        assert "isTPS" not in result

    def test_real_substrate_gets_istps(self):
        result = assign_is_tps_label({GPP})
        assert "isTPS" in result
        assert GPP in result

    def test_real_substrate_with_precursor_gets_istps(self):
        """This is the bug case: precursor substr should NOT block isTPS."""
        result = assign_is_tps_label({FPP, "precursor substr"})
        assert "isTPS" in result
        assert FPP in result
        assert "precursor substr" in result

    def test_real_substrate_with_other_and_precursor(self):
        result = assign_is_tps_label({FPP, "other", "precursor substr"})
        assert "isTPS" in result

    def test_multiple_real_substrates(self):
        result = assign_is_tps_label({FPP, GPP})
        assert "isTPS" in result
        assert FPP in result
        assert GPP in result

    def test_empty_set(self):
        result = assign_is_tps_label(set())
        assert "isTPS" not in result
        assert result == set()

    def test_does_not_mutate_input(self):
        original = {FPP}
        original_copy = original.copy()
        assign_is_tps_label(original)
        assert original == original_copy

    def test_idempotent(self):
        """Calling twice should not add duplicate isTPS or change result."""
        first = assign_is_tps_label({GPP})
        second = assign_is_tps_label(first)
        assert first == second
