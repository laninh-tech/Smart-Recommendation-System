"""
Import e-commerce interaction CSV into SmartRec format.

Expected minimum columns:
- user_id
- product_id
- action (view | add_to_cart | purchase)

Optional columns:
- timestamp (ISO8601 or parseable datetime)
- rating (1..5)
- user_name, age_group, active_level
- title, category, price, thumbnail, brand

Output files:
- users.json
- products.json
- interactions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VALID_ACTIONS = {"view", "add_to_cart", "purchase"}


def _normalize_actions(df: pd.DataFrame) -> pd.DataFrame:
    action_alias = {
        "click": "view",
        "impression": "view",
        "cart": "add_to_cart",
        "add-cart": "add_to_cart",
        "buy": "purchase",
        "order": "purchase",
    }

    df = df.copy()
    df["action"] = (
        df["action"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(action_alias)
    )

    invalid = sorted(set(df[~df["action"].isin(VALID_ACTIONS)]["action"].unique().tolist()))
    if invalid:
        raise ValueError(
            f"Unsupported actions found: {invalid}. "
            f"Allowed actions: {sorted(VALID_ACTIONS)}"
        )

    return df


def _prepare_users(df: pd.DataFrame) -> list[dict]:
    users = []

    if "user_name" not in df.columns:
        df = df.assign(user_name=df["user_id"].astype(str))
    if "age_group" not in df.columns:
        df = df.assign(age_group="unknown")
    if "active_level" not in df.columns:
        df = df.assign(active_level="medium")

    # Infer top-2 category preferences from interaction frequency.
    cat_df = df.copy()
    if "category" not in cat_df.columns:
        cat_df = cat_df.assign(category="General")

    pref_map: dict[str, list[str]] = {}
    grouped = (
        cat_df.groupby(["user_id", "category"], dropna=False)
        .size()
        .reset_index(name="cnt")
        .sort_values(["user_id", "cnt"], ascending=[True, False])
    )

    for uid, g in grouped.groupby("user_id"):
        pref_map[str(uid)] = [str(x) for x in g["category"].head(2).tolist()]

    user_cols = ["user_id", "user_name", "age_group", "active_level"]
    for row in df[user_cols].drop_duplicates().itertuples(index=False):
        uid = str(row.user_id)
        users.append(
            {
                "user_id": uid,
                "name": str(row.user_name),
                "preferences": pref_map.get(uid, []),
                "age_group": str(row.age_group),
                "active_level": str(row.active_level),
            }
        )

    return users


def _prepare_products(df: pd.DataFrame) -> list[dict]:
    products = []

    defaults = {
        "title": "Unknown Product",
        "category": "General",
        "price": 0.0,
        "thumbnail": "https://dummyimage.com/200x200/4f46e5/ffffff.png&text=Product",
        "brand": "Unknown",
    }

    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    # Rating proxy derived from purchase event ratings when available.
    purchase = df[df["action"] == "purchase"].copy()
    if "rating" in purchase.columns:
        rating_by_product = (
            purchase.dropna(subset=["rating"])
            .groupby("product_id")["rating"]
            .mean()
            .to_dict()
        )
    else:
        rating_by_product = {}

    prod_cols = ["product_id", "title", "category", "price", "thumbnail", "brand"]
    for row in df[prod_cols].drop_duplicates(subset=["product_id"]).itertuples(index=False):
        pid = str(row.product_id)
        products.append(
            {
                "product_id": pid,
                "title": str(row.title),
                "category": str(row.category),
                "price": float(row.price),
                "rating": round(float(rating_by_product.get(pid, 3.0)), 2),
                "thumbnail": str(row.thumbnail),
                "brand": str(row.brand),
            }
        )

    return products


def _prepare_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        ts = pd.Timestamp.utcnow()

    interactions = pd.DataFrame(
        {
            "user_id": df["user_id"].astype(str),
            "product_id": df["product_id"].astype(str),
            "action": df["action"].astype(str),
            "rating": df["rating"] if "rating" in df.columns else None,
            "timestamp": ts,
        }
    )

    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="coerce")
    interactions = interactions.dropna(subset=["timestamp"])
    interactions = interactions.sort_values("timestamp")
    interactions["timestamp"] = interactions["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return interactions


def import_csv(input_csv: Path, output_dir: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required = {"user_id", "product_id", "action"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _normalize_actions(df)

    users = _prepare_users(df)
    products = _prepare_products(df)
    interactions = _prepare_interactions(df)

    with open(output_dir / "users.json", "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

    with open(output_dir / "products.json", "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    interactions.to_csv(output_dir / "interactions.csv", index=False)

    print("Import complete:")
    print(f"  users: {len(users)}")
    print(f"  products: {len(products)}")
    print(f"  interactions: {len(interactions)}")
    print(f"  output_dir: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import e-commerce CSV to SmartRec data format")
    parser.add_argument("--input", required=True, help="Path to raw interaction CSV")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output folder for users.json/products.json/interactions.csv",
    )
    args = parser.parse_args()

    import_csv(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
