from sqlalchemy.orm import Session

from analyst_forecast.infrastructure.db.models import IdSequenceRecord


def next_id(
    session: Session,
    prefix: str,
    *,
    width: int = 6,
    sequence_key: str | None = None,
) -> str:
    key = sequence_key or prefix
    sequence = session.get(IdSequenceRecord, key)
    if sequence is None:
        sequence = IdSequenceRecord(sequence_key=key, current_value=1)
        session.add(sequence)
    else:
        sequence.current_value += 1
    session.flush()
    return f"{prefix}{sequence.current_value:0{width}d}"
