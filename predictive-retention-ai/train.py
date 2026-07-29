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
from src.data.preprocessor import preprocess_pipeline, apply_smote
from src.models.xgboost_model import (
    train_xgboost, evaluate_model, optimize_hyperparams, save_model,
    save_evaluation_report,
)
from src.models.model_comparison import compare_models, save_model_comparison
from src.models.calibration import (
    calibration_report, fit_oof_calibrator, save_calibration,
)
from src.monitoring.model_monitor import build_reference_profile, save_reference_profile
from src.xai.shap_explainer import (
    get_shap_explainer, compute_shap_values, get_global_feature_importance, plot_summary
)


def main(optimize: bool = False, compare: bool = False):
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
    results = preprocess_pipeline(df, use_smote=not optimize)
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
        best_params = optimize_hyperparams(X_train, y_train, n_trials=50)
        X_train_fit, y_train_fit = apply_smote(X_train, y_train)
        model = train_xgboost(X_train_fit, y_train_fit, params=best_params)
    else:
        model = train_xgboost(X_train, y_train)

    # 4. Değerlendirme
    print("\n📊 Model değerlendiriliyor...")
    metrics = evaluate_model(model, X_test, y_test)

    # 5. Modeli kaydet
    save_model(model, metrics=metrics)
    save_evaluation_report(metrics, model_params=model.get_params())

    print("\n🎯 Churn olasılıkları kalibre ediliyor...")
    calibrator, _ = fit_oof_calibrator(
        model,
        results["X_train_unbalanced"],
        results["y_train_unbalanced"],
    )
    calibrated_test = calibrator.predict(metrics["y_proba"])
    cal_report = calibration_report(
        results["y_test"], metrics["y_proba"], calibrated_test
    )
    save_calibration(calibrator, cal_report)
    save_reference_profile(
        build_reference_profile(results["raw_train"]),
        ROOT / "models" / "drift_reference.json",
    )
    print(
        "✅ Kalibrasyon tamamlandı. "
        f"Brier: {cal_report['raw']['brier_score']:.4f} → "
        f"{cal_report['calibrated']['brier_score']:.4f}"
    )

    if compare:
        print("\n⚖️ Baseline modeller karşılaştırılıyor...")
        comparison = compare_models(
            results["X_train_unbalanced"],
            results["y_train_unbalanced"],
        )
        save_model_comparison(comparison)
        print(comparison[["model", "f1", "recall", "roc_auc"]].to_string(index=False))

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
    parser.add_argument("--compare", action="store_true",
                        help="Baseline modelleri leakage-safe CV ile karşılaştır")
    args = parser.parse_args()
    main(optimize=args.optimize, compare=args.compare)
