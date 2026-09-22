"""Optional stateless HTTP adapter. Install with: pip install -e '.[http]'.

Prefer direct Python calls for the real-time path: JSON serializing [H,F]
features and transferring CUDA tensors to CPU both add overhead.
"""

from __future__ import annotations

import argparse

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .acc import ACCDetector
from .detector import combine_results
from .llmd import LLMDDetector


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TokenRequest(Request):
    tokens: list[list[float]]


class PairRequest(Request):
    previous_positions: list[list[float]]
    current_positions: list[list[float]]
    executed_steps: int = Field(ge=1)
    previous_smoothed_score: float = Field(ge=0)


class EvaluationRequest(Request):
    tokens: list[list[float]] | None = None
    positions: list[list[float]] | None = None
    previous_positions: list[list[float]] | None = None
    executed_steps: int | None = Field(default=None, ge=1)
    previous_smoothed_score: float | None = Field(default=None, ge=0)


def create_app(*, llmd: LLMDDetector | None = None, acc: ACCDetector | None = None) -> FastAPI:
    """ACC history is supplied by clients; concurrent environments never mix.

    Treat loaded detector configurations as read-only while serving requests.
    """
    if llmd is None and acc is None:
        raise ValueError("Configure at least one detector")
    app = FastAPI(title="VLA-FAIL scoring API", version="0.1.0")

    def checked(function):
        try:
            return function().to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/health")
    def health():
        return {
            "llmd_enabled": llmd is not None,
            "acc_enabled": acc is not None,
            "llmd_fitted": llmd is not None and llmd.fitted,
            "feature_shape": None if llmd is None else llmd.feature_shape,
            "llmd_threshold": None if llmd is None else llmd.threshold,
            "acc_threshold": None if acc is None else acc.threshold,
        }

    @app.post("/v1/llmd/score")
    def llmd_score(request: TokenRequest):
        if llmd is None:
            raise HTTPException(status_code=503, detail="LLMD is not configured")
        return checked(lambda: llmd.score(request.tokens))

    @app.post("/v1/acc/score")
    def acc_score(request: PairRequest):
        if acc is None:
            raise HTTPException(status_code=503, detail="ACC is not configured")
        return checked(
            lambda: acc.score_pair(
                request.previous_positions,
                request.current_positions,
                request.executed_steps,
                previous_smoothed_score=request.previous_smoothed_score,
            )
        )

    @app.post("/v1/intervention/evaluate")
    def evaluate(request: EvaluationRequest):
        def run():
            if llmd is not None and request.tokens is None:
                raise ValueError("tokens are required because LLMD is enabled")
            if llmd is None and request.tokens is not None:
                raise ValueError("LLMD is not configured")
            llmd_result = None if llmd is None else llmd.score(request.tokens)
            acc_result = None
            if acc is not None:
                if request.positions is None:
                    raise ValueError("positions are required because ACC is enabled")
                if request.previous_positions is None:
                    if request.executed_steps is not None or request.previous_smoothed_score is not None:
                        raise ValueError(
                            "First chunk: omit previous_positions, executed_steps and previous_smoothed_score"
                        )
                    # Local object only; there is no shared per-episode server state.
                    acc_result = ACCDetector(acc.alpha, acc.v_min, acc.threshold).update(request.positions)
                else:
                    if request.executed_steps is None or request.previous_smoothed_score is None:
                        raise ValueError("Provide executed_steps and previous_smoothed_score with previous_positions")
                    acc_result = acc.score_pair(
                        request.previous_positions,
                        request.positions,
                        request.executed_steps,
                        previous_smoothed_score=request.previous_smoothed_score,
                    )
            elif any(
                value is not None
                for value in (
                    request.positions,
                    request.previous_positions,
                    request.executed_steps,
                    request.previous_smoothed_score,
                )
            ):
                raise ValueError("ACC is not configured")
            return combine_results(llmd=llmd_result, acc=acc_result)

        return checked(run)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llmd-model", help="LLMD .npz artifact including its calibrated threshold")
    parser.add_argument("--acc", action="store_true", help="Enable stateless ACC scoring")
    parser.add_argument("--acc-threshold", type=float)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--v-min", type=float, default=1e-3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not args.llmd_model and not args.acc:
        parser.error("Provide --llmd-model and/or --acc")
    if args.acc_threshold is not None and not args.acc:
        parser.error("--acc-threshold requires --acc")
    llmd = LLMDDetector.load(args.llmd_model) if args.llmd_model else None
    acc = ACCDetector(args.alpha, args.v_min, args.acc_threshold) if args.acc else None
    import uvicorn

    uvicorn.run(create_app(llmd=llmd, acc=acc), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
