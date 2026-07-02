"""
Model Eğitim Script'i
Komut satırından çalıştırılarak modeli eğitir ve değerlendirir.

Kullanım:
    python train.py
    python train.py --optimize  # Optuna ile hiperparametre optimizasyonu
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw_data, get_data_info
from src.data.preprocessor import preprocess_pipeline
from src.models.xgboost_model import (
    train_xgboost, evaluate_model, optimize_hyperparams, save_model
)
from src.xai.shap_explainer import (
    get_shap_explainer, compute_shap_values, get_global_feature_importance, plot_summary
)


def main(optimize: bool = False):
    print("=" * 60)
    print("🧠 Predictive Retention AI — Model Eğitimi")
    print("=" * 60)

    # 1. Veri yükle
    print("\n📂 Veri yükleniyor...")
    df = load_raw_data()
    info = get_data_info(df)
    print(f"   Toplam müşteri: {info['n_rows']}")
    print(f"   Churn oranı: %{info.get('churn_rate', 0)*100:.1f}")

    # 2. Ön işleme
    print("\n⚙️ Veri ön işleniyor...")
    results = preprocess_pipeline(df, use_smote=True)
    X_train = results["X_train"]
    X_test = results["X_test"]
    y_train = results["y_train"]
    y_test = results["y_test"]
    feature_names = results["feature_names"]

    print(f"   Eğitim seti: {X_train.shape}")
    print(f"   Test seti: {X_test.shape}")

    # 3. Model eğitimi
    print("\n🤖 XGBoost modeli eğitiliyor...")
    if optimize:
        print("   🔍 Hiperparametre optimizasyonu başlatılıyor...")
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        best_params = optimize_hyperparams(X_tr, y_tr, X_val, y_val, n_trials=50)
        model = train_xgboost(X_train, y_train, params=best_params)
    else:
        model = train_xgboost(X_train, y_train)

    # 4. Değerlendirme
    print("\n📊 Model değerlendiriliyor...")
    metrics = evaluate_model(model, X_test, y_test)

    # 5. Modeli kaydet
    save_model(model)

    # 6. SHAP analizi
    print("\n🔍 SHAP analizi yapılıyor...")
    explainer = get_shap_explainer(model)
    # Test setinin küçük bir örneğinde SHAP hesapla
    sample_size = min(200, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    shap_values = compute_shap_values(explainer, X_sample)

    importance_df = get_global_feature_importance(shap_values, feature_names)
    print("\n📌 En Önemli 10 Özellik (SHAP):")
    for _, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']:<40} {row['mean_abs_shap']:.4f}")

    # SHAP grafiği kaydet
    try:
        plot_summary(shap_values, X_sample, save=True)
    except Exception as e:
        print(f"   ⚠️ SHAP grafiği kaydedilemedi: {e}")

    print("\n" + "="*60)
    print("✅ Eğitim tamamlandı!")
    print(f"   F1 Score : {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive Retention AI — Model Eğitimi")
    parser.add_argument("--optimize", action="store_true",
                        help="Optuna ile hiperparametre optimizasyonu yap")
    args = parser.parse_args()
    main(optimize=args.optimize)
