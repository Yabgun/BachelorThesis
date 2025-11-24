import os
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class LargeScaleFeatureExtractor:
    def __init__(self, input_csv_path: str, output_dir_path: str, chunksize: int = 100000, id_columns: List[str] | None = None):
        self.input_csv_path = input_csv_path
        self.output_dir_path = output_dir_path
        self.chunksize = chunksize
        self.id_columns = set(id_columns or [])
        self.roles: Dict[str, str] = {}
        self.numeric_stats: Dict[str, Dict[str, float]] = {}
        self.categorical_counts: Dict[str, Dict[str, int]] = {}
        self.datetime_candidates: List[str] = []
        self.text_columns: List[str] = []

    def run(self) -> None:
        self._ensure_output_dir()
        self._detect_roles_and_collect_stats()
        self._second_pass_transform_and_write()

    def _ensure_output_dir(self) -> None:
        Path(self.output_dir_path).mkdir(parents=True, exist_ok=True)

    def _detect_roles_and_collect_stats(self) -> None:
        sample = pd.read_csv(self.input_csv_path, nrows=min(self.chunksize, 5000))
        self.roles = self._infer_roles_from_sample(sample)
        for c in sample.columns:
            if c.lower() in {"id", "patient_id", "uid"}:
                self.id_columns.add(c)
        for chunk in pd.read_csv(self.input_csv_path, chunksize=self.chunksize):
            self._update_numeric_stats(chunk)
            self._update_categorical_counts(chunk)

    def _infer_roles_from_sample(self, df: pd.DataFrame) -> Dict[str, str]:
        roles: Dict[str, str] = {}
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                roles[c] = "numeric"
                continue
            if pd.api.types.is_datetime64_any_dtype(s):
                roles[c] = "datetime"
                continue
            if s.dtype == object:
                non_null = s.dropna()
                if non_null.empty:
                    roles[c] = "ignore"
                    continue
                try:
                    parsed = pd.to_datetime(non_null.sample(min(500, len(non_null))), errors="coerce", utc=False)
                    rate = float(parsed.notna().mean())
                except Exception:
                    rate = 0.0
                if rate >= 0.6:
                    roles[c] = "datetime"
                    continue
                uniq_ratio = float(non_null.nunique(dropna=True)) / float(len(non_null))
                if uniq_ratio <= 0.2:
                    roles[c] = "categorical"
                else:
                    roles[c] = "text"
            else:
                roles[c] = "ignore"
        self.datetime_candidates = [c for c, r in roles.items() if r == "datetime"]
        self.text_columns = [c for c, r in roles.items() if r == "text"]
        return roles

    def _update_numeric_stats(self, chunk: pd.DataFrame) -> None:
        num_cols = [c for c, r in self.roles.items() if r == "numeric"]
        if not num_cols:
            return
        sub = chunk[num_cols].apply(pd.to_numeric, errors="coerce")
        sums = sub.sum(skipna=True)
        sumsq = (sub ** 2).sum(skipna=True)
        counts = sub.notna().sum()
        mins = sub.min(skipna=True)
        maxs = sub.max(skipna=True)
        for c in num_cols:
            st = self.numeric_stats.get(c, {"sum": 0.0, "sumsq": 0.0, "count": 0.0, "min": math.inf, "max": -math.inf})
            st["sum"] += float(sums.get(c, 0.0))
            st["sumsq"] += float(sumsq.get(c, 0.0))
            st["count"] += float(counts.get(c, 0.0))
            vmin = mins.get(c)
            vmax = maxs.get(c)
            if pd.notna(vmin):
                st["min"] = min(st["min"], float(vmin))
            if pd.notna(vmax):
                st["max"] = max(st["max"], float(vmax))
            self.numeric_stats[c] = st

    def _update_categorical_counts(self, chunk: pd.DataFrame) -> None:
        cat_cols = [c for c, r in self.roles.items() if r == "categorical"]
        for c in cat_cols:
            vc = chunk[c].astype(str).fillna("").value_counts()
            bucket = self.categorical_counts.get(c, {})
            for k, v in vc.items():
                if k in bucket:
                    bucket[k] += int(v)
                else:
                    bucket[k] = int(v)
            self.categorical_counts[c] = bucket

    def _numeric_means_stds(self) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for c, st in self.numeric_stats.items():
            cnt = max(float(st["count"]), 1.0)
            mean = float(st["sum"]) / cnt
            var = max(float(st["sumsq"]) / cnt - mean * mean, 0.0)
            std = math.sqrt(var) if var > 1e-18 else 0.0
            out[c] = (mean, std)
        return out

    def _cat_frequency_maps(self) -> Dict[str, Dict[str, float]]:
        maps: Dict[str, Dict[str, float]] = {}
        for c, d in self.categorical_counts.items():
            total = float(sum(int(v) for v in d.values()))
            if total <= 0:
                maps[c] = {}
            else:
                maps[c] = {k: float(v) / total for k, v in d.items()}
        return maps

    def _parse_datetime_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in self.datetime_candidates:
            if c in df.columns:
                dt = pd.to_datetime(df[c], errors="coerce", utc=False)
                df[f"{c}_year"] = dt.dt.year.astype("Int32")
                df[f"{c}_month"] = dt.dt.month.astype("Int8")
                df[f"{c}_day"] = dt.dt.day.astype("Int8")
                df[f"{c}_dow"] = dt.dt.dayofweek.astype("Int8")
                df[f"{c}_hour"] = dt.dt.hour.astype("Int8")
                df = df.drop(columns=[c])
        return df

    def _add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in self.text_columns:
            if c in df.columns:
                s = df[c].astype(str).fillna("")
                lengths = s.str.len().astype("Int32")
                words = s.str.split()
                word_counts = words.map(len).astype("Int32")
                avg_word_len = pd.to_numeric(np.where(word_counts > 0, lengths / word_counts, 0.0), errors="coerce")
                df[f"{c}_len"] = lengths
                df[f"{c}_wc"] = word_counts
                df[f"{c}_awl"] = avg_word_len.astype("float32")
                df = df.drop(columns=[c])
        return df

    def _second_pass_transform_and_write(self) -> None:
        num_stats = self._numeric_means_stds()
        cat_maps = self._cat_frequency_maps()
        part_idx = 0
        for chunk in pd.read_csv(self.input_csv_path, chunksize=self.chunksize):
            df = chunk.copy()
            for c, role in self.roles.items():
                if role == "numeric" and c in df.columns:
                    mean, std = num_stats.get(c, (0.0, 0.0))
                    col = pd.to_numeric(df[c], errors="coerce")
                    col = col.fillna(mean)
                    if std > 0:
                        df[f"{c}_z"] = ((col - mean) / std).astype("float32")
                    else:
                        df[f"{c}_z"] = (col * 0.0).astype("float32")
                    df = df.drop(columns=[c])
                elif role == "categorical" and c in df.columns:
                    fmap = cat_maps.get(c, {})
                    mapped = df[c].astype(str).map(lambda x: fmap.get(x, 0.0)).astype("float32")
                    df[f"{c}_freq"] = mapped
                    df = df.drop(columns=[c])
            df = self._parse_datetime_cols(df)
            df = self._add_text_features(df)
            keep_ids = [c for c in self.id_columns if c in chunk.columns]
            drop_cols = [c for c in df.columns if c not in keep_ids and c not in [col for col in df.columns if col.endswith("_z") or col.endswith("_freq") or col.endswith("_year") or col.endswith("_month") or col.endswith("_day") or col.endswith("_dow") or col.endswith("_hour") or col.endswith("_len") or col.endswith("_wc") or col.endswith("_awl")]]
            if drop_cols:
                df = df.drop(columns=drop_cols, errors="ignore")
            out_path = Path(self.output_dir_path) / f"part_{part_idx:05d}.parquet"
            df.to_parquet(out_path, index=False)
            part_idx += 1


def run_large_scale_features(input_csv_path: str, output_dir_path: str, chunksize: int = 100000, id_columns: List[str] | None = None) -> None:
    extractor = LargeScaleFeatureExtractor(input_csv_path=input_csv_path, output_dir_path=output_dir_path, chunksize=chunksize, id_columns=id_columns)
    extractor.run()



