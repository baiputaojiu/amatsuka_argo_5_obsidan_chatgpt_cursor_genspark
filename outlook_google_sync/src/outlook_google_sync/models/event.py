from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional

from ..constants import TOOL_MARKER


@dataclass
class EventModel:
    sync_key: str
    summary: str
    start: datetime
    end: datetime
    description: str = ""
    location: str = ""
    is_private: bool = False
    is_all_day: bool = False
    source_id: str = ""
    reader_engine: str = ""
    input_method: str = ""
    fingerprint: str = ""
    sync_key_kind: str = "primary"
    color_id: Optional[str] = None
    ol_category_color: Optional[int] = None
    busy_status: int = 2  # 0=Free,1=Tentative,2=Busy,3=OOF
    categories: list[str] = field(default_factory=list)

    def to_google_body(self, detail_level: str = "full", time_zone: str | None = None) -> dict:
        if self.is_all_day:
            start_val = {"date": self.start.strftime("%Y-%m-%d")}
            end_val = {"date": self.end.strftime("%Y-%m-%d")}
        else:
            start_val = {"dateTime": self.start.isoformat()}
            end_val = {"dateTime": self.end.isoformat()}
            # オフセットなし dateTime のとき Google API は timeZone 必須
            if time_zone:
                start_val["timeZone"] = time_zone
                end_val["timeZone"] = time_zone

        now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        body: dict = {
            "summary": self.summary,
            "start": start_val,
            "end": end_val,
            "visibility": "private" if self.is_private else "default",
            "extendedProperties": {
                "private": {
                    "tool_marker": TOOL_MARKER,
                    "sync_key": self.sync_key,
                    "sync_key_kind": self.sync_key_kind,
                    "reader_engine": self.reader_engine,
                    "input_method": self.input_method,
                    "last_tool_write_utc": now_utc,
                }
            },
        }
        if detail_level == "full":
            if self.description:
                body["description"] = self.description
            if self.location:
                body["location"] = self.location
        if self.color_id:
            body["colorId"] = self.color_id
        return body
