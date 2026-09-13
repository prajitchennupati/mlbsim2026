"""Model registry + evaluation history."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from mlbsim_api.deps import get_session
from mlbsim_api.schemas import EvalOut, ModelOut
from mlbsim_data.models import CalibrationBin, ModelEvalRun, ModelVersion

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelOut])
def list_models(s: Session = Depends(get_session)) -> list[ModelOut]:
    return [
        ModelOut(
            model_id=m.model_id,
            name=m.name,
            kind=m.kind,
            version=m.version,
            trained_at=m.trained_at,
            train_end=m.train_end,
            metrics=m.metrics_json or {},
        )
        for m in s.scalars(select(ModelVersion).order_by(ModelVersion.model_id))
    ]


@router.get("/{model_id}/evaluation", response_model=EvalOut)
def model_evaluation(model_id: str, s: Session = Depends(get_session)) -> EvalOut:
    if s.get(ModelVersion, model_id) is None:
        raise HTTPException(404, f"unknown model {model_id!r}")
    runs = list(
        s.scalars(
            select(ModelEvalRun)
            .where(ModelEvalRun.model_id == model_id)
            .order_by(desc(ModelEvalRun.created_at))
        )
    )
    run_ids = [r.eval_id for r in runs]
    bins = (
        list(
            s.scalars(
                select(CalibrationBin)
                .where(CalibrationBin.eval_id.in_(run_ids))
                .order_by(CalibrationBin.eval_id, CalibrationBin.bin_lower)
            )
        )
        if run_ids
        else []
    )
    return EvalOut(
        model_id=model_id,
        runs=[
            {
                "eval_id": r.eval_id,
                "split_name": r.split_name,
                "period_start": r.period_start.isoformat() if r.period_start else None,
                "period_end": r.period_end.isoformat() if r.period_end else None,
                "metrics": r.metrics_json,
                "created_at": r.created_at.isoformat(),
            }
            for r in runs
        ],
        calibration=[
            {
                "eval_id": b.eval_id,
                "bin_lower": float(b.bin_lower),
                "bin_upper": float(b.bin_upper),
                "n": int(b.n),
                "mean_pred": float(b.mean_pred),
                "mean_actual": float(b.mean_actual),
            }
            for b in bins
        ],
    )
