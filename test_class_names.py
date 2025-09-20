"""クラス名参照機能のテストスクリプト"""

import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from image.core.datasets.wood_dataset import WoodDataset
from image.core.utils.checkpoint import CheckpointManager
from image.core.evaluation.evaluator import ModelEvaluator


def test_dataset_class_names():
    """データセットのクラス名取得をテスト"""
    print("=== データセットクラス名テスト ===")
    
    # テストデータセットのパスを確認
    dataset_path = project_root / "dataset" / "test_dataset"
    print(f"データセットパス: {dataset_path}")
    
    if not dataset_path.exists():
        print("テストデータセットが見つかりません")
        return
    
    try:
        # WoodDatasetインスタンスを作成
        dataset = WoodDataset(
            dataset_path=str(dataset_path),
            num_class=0,  # 自動検出
            img_size=128,
            augmentation=False
        )
        
        # クラス名を取得
        class_names = dataset.get_class_names()
        print(f"クラス数: {dataset.get_num_class()}")
        print(f"クラス名: {class_names}")
        
        # データセット作成順序が固定されているかチェック
        print(f"データセット作成順序で固定されたクラス名: {class_names}")
        
    except Exception as e:
        print(f"データセットテストでエラー: {e}")


def test_checkpoint_manager():
    """CheckpointManagerのクラス名保存・取得をテスト"""
    print("\n=== CheckpointManagerテスト ===")
    
    # テスト用のチェックポイントディレクトリ
    test_checkpoint_dir = project_root / "test_checkpoints"
    test_checkpoint_dir.mkdir(exist_ok=True)
    
    try:
        # CheckpointManagerインスタンスを作成
        checkpoint_manager = CheckpointManager(str(test_checkpoint_dir))
        
        # テスト用クラス名
        test_class_names = ["oak", "pine", "birch"]
        
        # クラス名を保存
        checkpoint_manager.save_class_names(test_class_names)
        print(f"保存したクラス名: {test_class_names}")
        
        # クラス名を取得
        loaded_class_names = checkpoint_manager.get_class_names()
        print(f"読み込んだクラス名: {loaded_class_names}")
        
        # 一致確認
        if test_class_names == loaded_class_names:
            print("✓ クラス名の保存・取得が正常に動作しています")
        else:
            print("✗ クラス名の保存・取得に問題があります")
        
    except Exception as e:
        print(f"CheckpointManagerテストでエラー: {e}")
    
    finally:
        # テストディレクトリを削除
        import shutil
        if test_checkpoint_dir.exists():
            shutil.rmtree(test_checkpoint_dir)


def test_evaluator_class_names():
    """Evaluatorのクラス名取得をテスト"""
    print("\n=== Evaluatorクラス名テスト ===")
    
    # 既存のチェックポイントを確認
    checkpoints_dir = project_root / "checkpoints"
    
    if not checkpoints_dir.exists():
        print("チェックポイントディレクトリが見つかりません")
        return
    
    # 最新のチェックポイントを探す
    checkpoint_subdirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
    
    if not checkpoint_subdirs:
        print("チェックポイントが見つかりません")
        return
    
    # 最新のチェックポイントを選択
    latest_checkpoint_dir = sorted(checkpoint_subdirs)[-1]
    checkpoint_files = list(latest_checkpoint_dir.glob("*.pth"))
    
    if not checkpoint_files:
        print("チェックポイントファイルが見つかりません")
        return
    
    try:
        # best_accuracy.pthまたはlatest.pthを優先
        checkpoint_path = None
        for filename in ["best_accuracy.pth", "latest.pth"]:
            potential_path = latest_checkpoint_dir / filename
            if potential_path.exists():
                checkpoint_path = potential_path
                break
        
        if not checkpoint_path:
            checkpoint_path = checkpoint_files[0]
        
        print(f"テスト用チェックポイント: {checkpoint_path}")
        
        # ModelEvaluatorインスタンスを作成
        evaluator = ModelEvaluator(
            checkpoint_path=str(checkpoint_path),
            device="cpu",  # CPUでテスト
            img_size=128
        )
        
        # クラス名を取得
        class_names = evaluator.get_class_names()
        print(f"Evaluatorから取得したクラス名: {class_names}")
        
        # クラス名による予測結果の表示テスト
        if class_names:
            for i, class_name in enumerate(class_names):
                print(f"クラス {i}: {class_name}")
                
            # インデックスからクラス名を取得するテスト
            test_index = 0
            class_name = evaluator.get_class_name_by_index(test_index)
            print(f"インデックス {test_index} のクラス名: {class_name}")
        
    except Exception as e:
        print(f"Evaluatorテストでエラー: {e}")


if __name__ == "__main__":
    print("クラス名参照機能のテストを開始します...\n")
    
    test_dataset_class_names()
    test_checkpoint_manager()
    test_evaluator_class_names()
    
    print("\nテスト完了")