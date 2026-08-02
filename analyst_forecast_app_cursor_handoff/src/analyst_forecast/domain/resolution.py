"""対象解決の状態機械。DB保存値とworkflow判定を単一関数で揃える。"""

from __future__ import annotations

from analyst_forecast.domain.models import (
    ComponentResolutionStage,
    TargetResolutionStatus,
)
from analyst_forecast.infrastructure.db.models import (
    ForecastComponentRecord,
    TargetMappingRecord,
)

# P11提案後（独立P12待ち）
_AWAITING_REVIEW = frozenset(
    {
        TargetResolutionStatus.AWAITING_REVIEW.value,
        TargetResolutionStatus.REVIEW_PENDING.value,
        TargetResolutionStatus.PROPOSED.value,  # 旧保存値の後方互換
    }
)
# P12不一致後（P13待ち）
_AWAITING_ADJUDICATION = frozenset(
    {
        TargetResolutionStatus.AWAITING_ADJUDICATION.value,
        TargetResolutionStatus.NEEDS_ADJUDICATION.value,
    }
)
_LOCKED = frozenset({TargetResolutionStatus.LOCKED.value})
_UNRESOLVABLE = frozenset({TargetResolutionStatus.UNRESOLVABLE.value})
_PENDING_P11 = frozenset({TargetResolutionStatus.PENDING.value})

_EVALUABLE_MAPPING = frozenset({"verified", "corrected"})


def normalize_target_resolution_status(raw: str | None) -> str:
    """旧名・legacy名を現行の正規状態へ変換する。"""
    if raw is None or raw == "":
        return TargetResolutionStatus.PENDING.value
    if raw in _AWAITING_REVIEW:
        return TargetResolutionStatus.AWAITING_REVIEW.value
    if raw in _AWAITING_ADJUDICATION:
        return TargetResolutionStatus.AWAITING_ADJUDICATION.value
    if raw in _LOCKED:
        return TargetResolutionStatus.LOCKED.value
    if raw in _UNRESOLVABLE:
        return TargetResolutionStatus.UNRESOLVABLE.value
    if raw in _PENDING_P11:
        return TargetResolutionStatus.PENDING.value
    return TargetResolutionStatus.PENDING.value


def classify_component_resolution_stage(
    component: ForecastComponentRecord,
    mapping: TargetMappingRecord | None,
) -> ComponentResolutionStage:
    """1件のcomponentについて、次に必要な工程を返す。"""
    status = normalize_target_resolution_status(component.target_resolution_status)
    mapping_status = mapping.mapping_status if mapping is not None else None

    if status == TargetResolutionStatus.UNRESOLVABLE.value or mapping_status == "unresolvable":
        return ComponentResolutionStage.UNRESOLVABLE
    if (
        status == TargetResolutionStatus.LOCKED.value
        and mapping is not None
        and mapping_status in _EVALUABLE_MAPPING
        and mapping.locked_at is not None
    ):
        return ComponentResolutionStage.READY_FOR_EVALUATION
    if status == TargetResolutionStatus.AWAITING_ADJUDICATION.value:
        return ComponentResolutionStage.NEED_P13
    if status == TargetResolutionStatus.AWAITING_REVIEW.value:
        return ComponentResolutionStage.NEED_P12
    if status == TargetResolutionStatus.PENDING.value:
        return ComponentResolutionStage.NEED_P11
    # mapping未固定・未lockは評価へ進ませない
    if mapping is None or mapping_status not in _EVALUABLE_MAPPING or mapping.locked_at is None:
        if status == TargetResolutionStatus.AWAITING_ADJUDICATION.value:
            return ComponentResolutionStage.NEED_P13
        if status in {
            TargetResolutionStatus.AWAITING_REVIEW.value,
            TargetResolutionStatus.PROPOSED.value,
            TargetResolutionStatus.REVIEW_PENDING.value,
        }:
            return ComponentResolutionStage.NEED_P12
        return ComponentResolutionStage.NEED_P11
    return ComponentResolutionStage.READY_FOR_EVALUATION


def is_mapping_locked_for_evaluation(mapping: TargetMappingRecord | None) -> bool:
    if mapping is None:
        return False
    return mapping.mapping_status in _EVALUABLE_MAPPING and mapping.locked_at is not None
