"""Tests for negative-sample cleaning filters (from revision_data-preparation)."""

import pandas as pd

from enzymeexplorer.src.data_preparation.constants import (
    PUTATIVE_TPS_IDS,
    TPS_ECS_BASE,
)
from enzymeexplorer.src.data_preparation.negative_filters import (
    filter_by_ec,
    filter_out_putative_tpss,
)


class TestFilterOutPutativeTpss:
    def test_removes_known_ids(self) -> None:
        df = pd.DataFrame({"Entry": ["A0A2B7YDW3", "X1", "X2"]})
        result = filter_out_putative_tpss(df, PUTATIVE_TPS_IDS)
        assert list(result["Entry"]) == ["X1", "X2"]

    def test_keeps_all_when_no_match(self) -> None:
        df = pd.DataFrame({"Entry": ["X1", "X2", "X3"]})
        result = filter_out_putative_tpss(df, PUTATIVE_TPS_IDS)
        assert len(result) == 3

    def test_custom_putative_ids(self) -> None:
        df = pd.DataFrame({"Entry": ["A", "B", "C"]})
        result = filter_out_putative_tpss(df, ["B"])
        assert list(result["Entry"]) == ["A", "C"]

    def test_empty_df(self) -> None:
        df = pd.DataFrame({"Entry": pd.Series([], dtype=str)})
        result = filter_out_putative_tpss(df, PUTATIVE_TPS_IDS)
        assert len(result) == 0


class TestFilterByEc:
    def test_removes_entries_with_tps_ec(self) -> None:
        tps_df = pd.DataFrame({"EC number": ["4.2.3.100"]})
        nontps_df = pd.DataFrame({"EC number": ["4.2.3.100", "1.1.1.1", "4.2.1.123"]})
        result = filter_by_ec(tps_df, nontps_df, TPS_ECS_BASE)
        assert list(result["EC number"]) == ["1.1.1.1"]

    def test_keeps_entries_with_unrelated_ec(self) -> None:
        tps_df = pd.DataFrame({"EC number": ["4.2.3.100"]})
        nontps_df = pd.DataFrame({"EC number": ["1.1.1.1", "2.2.2.2"]})
        result = filter_by_ec(tps_df, nontps_df, TPS_ECS_BASE)
        assert len(result) == 2

    def test_handles_none_ec(self) -> None:
        tps_df = pd.DataFrame({"EC number": ["4.2.3.100"]})
        nontps_df = pd.DataFrame({"EC number": [None, "1.1.1.1"]})
        result = filter_by_ec(tps_df, nontps_df, TPS_ECS_BASE)
        assert len(result) == 2

    def test_semicolon_separated_ecs(self) -> None:
        tps_df = pd.DataFrame({"EC number": ["4.2.3.100; 4.2.3.101"]})
        nontps_df = pd.DataFrame({"EC number": ["4.2.3.101; 1.1.1.1", "2.2.2.2"]})
        result = filter_by_ec(tps_df, nontps_df, TPS_ECS_BASE)
        assert len(result) == 1
        assert result.iloc[0]["EC number"] == "2.2.2.2"

    def test_skips_partial_ecs_with_dash(self) -> None:
        """TPS ECs with '-' (partial) should not be collected from the TPS set."""
        tps_df = pd.DataFrame({"EC number": ["4.2.3.-"]})
        nontps_df = pd.DataFrame({"EC number": ["4.2.3.999"]})
        result = filter_by_ec(tps_df, nontps_df, TPS_ECS_BASE)
        assert len(result) == 1
