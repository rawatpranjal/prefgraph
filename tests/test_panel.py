"""Tests for BehaviorPanel and PanelSummary."""

import numpy as np
import pytest

from pyrevealed import BehaviorLog, BehaviorPanel, PanelSummary


@pytest.fixture
def sample_logs():
    """Create a list of BehaviorLog objects."""
    np.random.seed(42)
    return [
        BehaviorLog(
            np.random.rand(15, 4),
            np.random.rand(15, 4),
            user_id=f"user_{i}",
        )
        for i in range(5)
    ]


@pytest.fixture
def panel(sample_logs):
    return BehaviorPanel.from_logs(sample_logs)


class TestBehaviorPanelConstruction:

    def test_from_logs(self, sample_logs):
        panel = BehaviorPanel.from_logs(sample_logs)
        assert panel.num_users == 5
        assert len(panel) == 5

    def test_from_logs_auto_ids(self):
        """Logs without user_id get auto-assigned IDs."""
        np.random.seed(99)
        logs = [BehaviorLog(np.random.rand(5, 3) + 0.1, np.random.rand(5, 3) + 0.1) for _ in range(3)]
        panel = BehaviorPanel.from_logs(logs)
        assert panel.num_users == 3
        assert "user_0" in panel.user_ids

    def test_from_dict(self, sample_logs):
        d = {f"custom_{i}": log for i, log in enumerate(sample_logs)}
        panel = BehaviorPanel.from_dict(d)
        assert "custom_0" in panel
        assert panel.num_users == 5

    def test_duplicate_user_id_raises(self):
        np.random.seed(99)
        data = np.random.rand(5, 3) + 0.1
        logs = [
            BehaviorLog(data.copy(), data.copy(), user_id="same"),
            BehaviorLog(data.copy(), data.copy(), user_id="same"),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            BehaviorPanel.from_logs(logs)


class TestBehaviorPanelAccess:

    def test_getitem(self, panel):
        log = panel["user_0"]
        assert log.num_observations == 15

    def test_contains(self, panel):
        assert "user_0" in panel
        assert "nonexistent" not in panel

    def test_iter(self, panel):
        items = list(panel)
        assert len(items) == 5
        uid, log = items[0]
        assert isinstance(uid, str)
        assert log.num_observations == 15

    def test_user_ids(self, panel):
        ids = panel.user_ids
        assert len(ids) == 5
        assert "user_0" in ids

    def test_repr(self, panel):
        r = repr(panel)
        assert "users=5" in r
        assert "total_obs=75" in r


class TestBehaviorPanelAnalysis:

    def test_analyze_user(self, panel):
        summary = panel.analyze_user("user_0")
        assert hasattr(summary, "is_consistent")
        assert hasattr(summary, "efficiency_index")

    def test_filter(self, panel):
        filtered = panel.filter(lambda log: log.num_observations >= 15)
        assert filtered.num_users == 5  # all have 15 obs

        filtered = panel.filter(lambda log: log.num_observations > 100)
        assert filtered.num_users == 0


class TestPanelSummary:

    def test_summary_produces_panel_summary(self, panel):
        ps = panel.summary()
        assert isinstance(ps, PanelSummary)
        assert ps.num_users == 5
        assert ps.total_observations == 75

    def test_summary_fields(self, panel):
        ps = panel.summary()
        assert 0 <= ps.garp_pass_rate <= 1
        assert "mean" in ps.aei_distribution
        assert "std" in ps.aei_distribution
        assert "min" in ps.aei_distribution
        assert "max" in ps.aei_distribution

    def test_summary_text_contains_sections(self, panel):
        ps = panel.summary()
        text = ps.summary()
        assert "PANEL SUMMARY" in text
        assert "Consistency Rates:" in text
        assert "Efficiency Distribution:" in text
        assert "Most Inconsistent" in text
        assert "GARP" in text

    def test_summary_repr(self, panel):
        ps = panel.summary()
        r = repr(ps)
        assert "PanelSummary" in r
        assert "users=5" in r

    def test_summary_to_dict(self, panel):
        ps = panel.summary()
        d = ps.to_dict()
        assert "num_users" in d
        assert "garp_pass_rate" in d
        assert "aei_distribution" in d

    def test_summary_score(self, panel):
        ps = panel.summary()
        score = ps.score()
        assert 0 <= score <= 1
