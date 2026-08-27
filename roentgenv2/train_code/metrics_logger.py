import json
import os
from bisect import bisect_left
from pathlib import Path


class MetricsLogger:
    """Persist LoRA training metrics across checkpoint resumes."""

    def __init__(self, path, meta=None):
        self.path = os.fspath(path)
        self._current_meta = dict(meta or {})
        self.data = self._empty_data()

    def _empty_data(self):
        return {
            "meta": dict(self._current_meta),
            "train": {"step": [], "loss": [], "lr": []},
            "val": {"step": [], "loss": []},
        }

    @staticmethod
    def _trim_series(series_name, series, through_step, required_fields):
        if not isinstance(series, dict):
            raise ValueError(f"Invalid metrics file: {series_name!r} must be an object.")

        for field in required_fields:
            if not isinstance(series.get(field), list):
                raise ValueError(
                    f"Invalid metrics file: {series_name}.{field} must be a list."
                )

        steps = series["step"]
        for field, values in series.items():
            if isinstance(values, list) and len(values) != len(steps):
                raise ValueError(
                    f"Invalid metrics file: {series_name}.{field} has {len(values)} values "
                    f"for {len(steps)} steps."
                )

        try:
            normalized_steps = [int(step) for step in steps]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid metrics file: {series_name}.step must contain integers."
            ) from exc

        if len(set(normalized_steps)) != len(normalized_steps):
            raise ValueError(f"Invalid metrics file: {series_name}.step contains duplicates.")
        if normalized_steps != sorted(normalized_steps):
            raise ValueError(
                f"Invalid metrics file: {series_name}.step must be in ascending order."
            )

        keep = [index for index, step in enumerate(normalized_steps) if step <= through_step]
        trimmed = {}
        for field, values in series.items():
            if isinstance(values, list):
                trimmed[field] = [values[index] for index in keep]
            else:
                trimmed[field] = values
        trimmed["step"] = [normalized_steps[index] for index in keep]
        return trimmed

    def load_for_resume(self, through_step):
        """Load existing history and discard points newer than the checkpoint."""

        metrics_path = Path(self.path)
        if not metrics_path.is_file():
            return False

        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot resume from invalid metrics JSON: {metrics_path}") from exc

        if not isinstance(payload, dict):
            raise ValueError("Invalid metrics file: the top-level value must be an object.")
        previous_meta = payload.get("meta", {})
        if not isinstance(previous_meta, dict):
            raise ValueError("Invalid metrics file: 'meta' must be an object.")

        payload["meta"] = {**previous_meta, **self._current_meta}
        payload["train"] = self._trim_series(
            "train", payload.get("train"), int(through_step), ("step", "loss", "lr")
        )
        payload["val"] = self._trim_series(
            "val", payload.get("val"), int(through_step), ("step", "loss")
        )
        self.data = payload
        return True

    @staticmethod
    def _upsert(series, step, values):
        steps = series["step"]
        position = bisect_left(steps, step)
        if position < len(steps) and steps[position] == step:
            for field, value in values.items():
                if field not in series:
                    series[field] = [None] * len(steps)
                series[field][position] = value
            return

        existing_length = len(steps)
        steps.insert(position, step)
        for field, field_values in series.items():
            if field == "step" or not isinstance(field_values, list):
                continue
            field_values.insert(position, values.get(field))
        for field, value in values.items():
            if field not in series:
                field_values = [None] * existing_length
                field_values.insert(position, value)
                series[field] = field_values

    def log_train(self, step, loss, lr, text_encoder_lr=None):
        values = {"loss": float(loss), "lr": float(lr)}
        if text_encoder_lr is not None or "text_encoder_lr" in self.data["train"]:
            values["text_encoder_lr"] = (
                float(text_encoder_lr) if text_encoder_lr is not None else None
            )
        self._upsert(self.data["train"], int(step), values)

    def log_val(self, step, loss):
        self._upsert(self.data["val"], int(step), {"loss": float(loss)})

    def save(self):
        metrics_path = Path(self.path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metrics_path.with_name(f".{metrics_path.name}.tmp")
        temporary_path.write_text(
            json.dumps(self.data, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, metrics_path)
