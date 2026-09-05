import pandas as pd

def quick_profile(df: pd.DataFrame):
    profile = {}

    for col in df.columns:
        s = df[col]

        profile[col] = {
            "dtype": s.dtype,
            "n_missing": s.isna().sum(),
            "missing_%": round(s.isna().mean() * 100, 2),
            "n_unique": s.nunique(),
            "sample_values": s.dropna().unique()[:5].tolist(),
        }

        if pd.api.types.is_numeric_dtype(s):
            profile[col].update({
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max()
            })
        else:
            top = s.value_counts().head(3)
            profile[col]["top_values"] = top.to_dict()

    return pd.DataFrame(profile).T
