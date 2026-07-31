import os
from pathlib import Path

import pytest

from onedrive_destination_recommender.msg_reader import probe_msg_access

pytestmark = pytest.mark.integration


def test_outlook_com_can_access_local_msg_fields() -> None:
    configured_path = os.environ.get("ODR_TEST_MSG_PATH")
    if not configured_path:
        pytest.skip("ODR_TEST_MSG_PATH is not configured")

    result = probe_msg_access(Path(configured_path))

    assert result.subject_accessible
    assert result.body_accessible
    assert isinstance(result.attachment_count, int)
