"""Stage 3: split the validated data.

Input:   data/interim/validated.parquet
Outputs: data/interim/splits/{train_real,val,test_golden,test_recent}.parquet,
         reports/split_report.json

The rules (from the design document):

* real rows only are split 70/15/15, stratified by churn, with a fixed seed;
  ``test_golden`` is fixed for the whole life of the project and is never trained on;
* the split happens BEFORE synthetic data is made, so no synthetic row can be derived
  from a customer in a test set;
* feedback rows: 30% of the most recent feedback (the newest batch, plus older batches until
  ``recent_window_min_rows`` rows are pooled) becomes ``test_recent`` (what the
  model must do well on NOW); the other 70% and all older feedback join the training rows;
* with no feedback yet, ``test_recent`` is simply empty (a valid, expected state).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from churnsense.config import Params, Paths
from churnsense.logging_utils import get_logger
from churnsense.tracking import StageError, StageReport, tracked_stage

log = get_logger(__name__)

SPLIT_NAMES = ("train_real", "val", "test_golden", "test_recent")


def _stratified_part(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (rest, part) with ``part`` being ``test_size`` of the rows."""
    rest, part = train_test_split(df, test_size=test_size, stratify=df["churn"], random_state=seed)
    return rest.sort_index(), part.sort_index()


def split_frames(df: pd.DataFrame, params: Params) -> dict[str, pd.DataFrame]:
    """Return the four splits of a validated frame."""
    seed = int(params.seed)
    val_frac, test_frac = float(params.split.val_frac), float(params.split.test_frac)
    real = df[df["source"] == "real"]
    feedback = df[df["source"] == "feedback"]

    held = val_frac + test_frac
    train_real, rest = _stratified_part(real, held, seed)
    val, golden = _stratified_part(rest, test_frac / held, seed)

    recent = df.iloc[0:0]  # empty frame with the right columns and types
    if len(feedback):
        # The "recent" window: the newest batch, plus older batches until there are enough rows.
        # Without this a tiny newest batch (a few outcomes reported since the last run) would
        # leave the recent test set too small for the gate to say anything.
        newest_first = feedback.groupby("batch_id")["event_ts"].max().sort_values(ascending=False)
        window_ids: list[str] = []
        pooled = 0
        for batch_id in newest_first.index:
            window_ids.append(batch_id)
            pooled += int((feedback["batch_id"] == batch_id).sum())
            if pooled >= int(params.split.get("recent_window_min_rows", 1000)):
                break
        in_window = feedback["batch_id"].isin(window_ids)
        newest, older = feedback[in_window], feedback[~in_window]
        fraction = float(params.split.recent_holdout_frac)
        if newest["churn"].nunique() > 1 and len(newest) >= 4:
            newest_train, recent = _stratified_part(newest, fraction, seed)
        else:  # tiny or one-class window: a plain random holdout
            recent = newest.sample(frac=fraction, random_state=seed)
            newest_train = newest.drop(index=recent.index)
        train_real = pd.concat([train_real, older, newest_train.sort_index()])

    frames = {"train_real": train_real, "val": val, "test_golden": golden, "test_recent": recent}
    return {name: frame.reset_index(drop=True) for name, frame in frames.items()}


def check_no_overlap(frames: dict[str, pd.DataFrame]) -> None:
    """Raise ``StageError`` if one customer appears in more than one split."""
    seen: dict[str, str] = {}
    for name, frame in frames.items():
        for customer in frame["customer_id"]:
            if customer in seen:
                raise StageError(
                    f"customer {customer!r} is in both {seen[customer]!r} and {name!r}"
                )
            seen[customer] = name


@tracked_stage("split", report="split_report.json")
def run(params: Params, paths: Paths) -> StageReport:
    """Run the split stage."""
    df = pd.read_parquet(paths.validated)
    frames = split_frames(df, params)
    check_no_overlap(frames)

    paths.ensure_parents(paths.split_file("train_real"))
    for name, frame in frames.items():
        frame.to_parquet(paths.split_file(name), index=False)

    real_rate = float(df.loc[df["source"] == "real", "churn"].mean())
    sizes = {name: len(frame) for name, frame in frames.items()}
    rates = {
        name: round(float(frame["churn"].mean()), 4) for name, frame in frames.items() if len(frame)
    }
    gap = max(
        (abs(rate - real_rate) for n, rate in rates.items() if n != "test_recent"), default=0.0
    )
    return StageReport(
        rows_in=len(df),
        rows_out=sum(sizes.values()),
        extra={
            "split_sizes": sizes,
            "churn_rate_by_split": rates,
            "churn_rate_real_overall": round(real_rate, 4),
            "max_stratification_gap": round(gap, 4),
            "seed": int(params.seed),
        },
    )


if __name__ == "__main__":
    run.main()
