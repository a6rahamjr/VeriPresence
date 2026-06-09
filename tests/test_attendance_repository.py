from datetime import datetime, timezone

from veripresence.storage.repository import AttendanceRepository


def test_cooldown_is_scoped_to_the_camera(tmp_path):
    repository = AttendanceRepository(tmp_path / "attendance.db")
    captured_at = datetime(2026, 6, 9, 8, 30, tzinfo=timezone.utc)

    assert repository.record(
        "alex", True, 0.9, 0.7, "north-door", 300, captured_at
    )
    assert not repository.record(
        "alex", True, 0.9, 0.7, "north-door", 300, captured_at
    )
    assert repository.record(
        "alex", True, 0.9, 0.7, "south-door", 300, captured_at
    )


def test_filters_summary_and_retention(tmp_path):
    repository = AttendanceRepository(tmp_path / "attendance.db")
    morning = datetime(2026, 6, 9, 8, 30, tzinfo=timezone.utc)
    afternoon = datetime(2026, 6, 9, 15, 45, tzinfo=timezone.utc)
    old_event = datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc)

    repository.record("alex", True, 0.92, 0.75, "front-desk", 0, morning)
    repository.record("alex", True, 0.88, 0.69, "front-desk", 0, afternoon)
    repository.record(None, False, 0.43, 0.05, "front-desk", 0, afternoon)
    repository.record("blair", True, 0.91, 0.73, "side-door", 0, old_event)

    events = repository.list_events(
        event_date="2026-06-09",
        event_type="attendance",
        identity="alex",
        source="front-desk",
    )
    summary = repository.daily_summary("2026-06-09", "front-desk")

    assert len(events) == 2
    assert summary["present_count"] == 1
    assert summary["unknown_events"] == 1
    assert summary["identities"][0]["sightings"] == 2
    assert repository.delete_before("2026-06-01") == 1
