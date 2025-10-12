#!/usr/bin/env python3
"""
Compare predictions for sample_test against the ground truth sample_test_out.
Reads predictions from outputs/sample_test_pred_improved.csv (or a provided path)
and compares with student_resource/dataset/sample_test_out.csv.
Prints summary metrics and saves detailed comparison to outputs/sample_test_comparison.csv.
"""
import os
import argparse
import numpy as np
import pandas as pd


def smape(y_true, y_pred, eps=1e-8):
	y_true = np.array(y_true, dtype=float)
	y_pred = np.array(y_pred, dtype=float)
	denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
	denom = np.where(denom == 0, eps, denom)
	return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--pred_csv", default=os.path.join("outputs", "sample_test_pred_improved.csv"))
	parser.add_argument("--truth_csv", default=os.path.join("student_resource", "dataset", "sample_test_out.csv"))
	parser.add_argument("--out_csv", default=os.path.join("outputs", "sample_test_comparison.csv"))
	args = parser.parse_args()

	if not os.path.exists(args.pred_csv):
		raise FileNotFoundError(f"Predictions not found: {args.pred_csv}")
	if not os.path.exists(args.truth_csv):
		raise FileNotFoundError(f"Ground truth not found: {args.truth_csv}")

	pred = pd.read_csv(args.pred_csv)
	truth = pd.read_csv(args.truth_csv)
	# Normalize column names if needed
	pred_cols = {c.lower(): c for c in pred.columns}
	truth_cols = {c.lower(): c for c in truth.columns}
	sid_col_pred = pred_cols.get("sample_id", None) or pred.columns[0]
	price_col_pred = pred_cols.get("price", None) or pred.columns[1]
	sid_col_truth = truth_cols.get("sample_id", None) or truth.columns[0]
	price_col_truth = truth_cols.get("price", None) or truth.columns[1]

	df = pred.rename(columns={sid_col_pred: "sample_id", price_col_pred: "price_pred"}) \
			 .merge(truth.rename(columns={sid_col_truth: "sample_id", price_col_truth: "price_actual"}),
					on="sample_id", how="inner")

	if df.empty:
		raise RuntimeError("No overlapping sample_id between predictions and truth.")

	df["abs_err"] = (df["price_pred"] - df["price_actual"]).abs()
	df["pct_err"] = df["abs_err"] / df["price_actual"].replace(0, np.nan)
	mae = df["abs_err"].mean()
	med_ae = df["abs_err"].median()
	s = smape(df["price_actual"], df["price_pred"])
	within_10 = (df["pct_err"] <= 0.10).mean()
	within_20 = (df["pct_err"] <= 0.20).mean()
	within_50 = (df["pct_err"] <= 0.50).mean()

	print("Summary metrics vs sample ground truth:")
	print(f"- Rows compared: {len(df)}")
	print(f"- MAE: {mae:.4f}")
	print(f"- Median AE: {med_ae:.4f}")
	print(f"- SMAPE: {s:.2f}%")
	print(f"- % within 10%: {within_10*100:.2f}%")
	print(f"- % within 20%: {within_20*100:.2f}%")
	print(f"- % within 50%: {within_50*100:.2f}%")

	print("\nTop 10 largest absolute errors:")
	print(df.sort_values("abs_err", ascending=False).head(10).to_string(index=False))

	os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
	df.to_csv(args.out_csv, index=False)
	print(f"\nSaved detailed comparison to {args.out_csv}")


if __name__ == "__main__":
	main()

