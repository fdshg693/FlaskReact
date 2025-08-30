#!/usr/bin/env python3
"""
テストデータセット作成スクリプト
木材分類のテスト用にダミー画像を生成します
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image, ImageDraw
import random
from loguru import logger


def create_test_image(
    width: int = 128,
    height: int = 128,
    class_name: str = "",
    image_name: str = "",
    color_scheme: Optional[List[Tuple[int, int, int]]] = None,
) -> Image.Image:
    """テスト用のダミー画像を作成"""

    # クラスごとの色スキーム
    color_schemes = {
        "oak": [(139, 69, 19), (160, 82, 45), (205, 133, 63)],  # 茶色系（オーク）
        "pine": [(34, 139, 34), (46, 125, 50), (76, 175, 80)],  # 緑系（パイン）
        "birch": [
            (245, 245, 220),
            (255, 248, 220),
            (250, 235, 215),
        ],  # ベージュ系（バーチ）
    }

    colors = color_schemes.get(
        class_name, [(128, 128, 128), (160, 160, 160), (192, 192, 192)]
    )

    # 背景色をランダムに選択
    bg_color = random.choice(colors)

    # 画像作成
    image = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(image)

    # 木目のような模様を追加
    for _ in range(random.randint(5, 15)):
        # ランダムな線を描画（木目風）
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)

        line_color = tuple(
            max(0, min(255, c + random.randint(-30, 30))) for c in bg_color
        )
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=random.randint(1, 3))

    # ノイズ追加
    pixels = np.array(image)
    noise = np.random.normal(0, 10, pixels.shape).astype(np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)

    # テキスト追加（デバッグ用）
    draw = ImageDraw.Draw(image)
    text = f"{class_name}\n{image_name}"

    # テキストの色（背景と対比）
    text_color = (255, 255, 255) if sum(bg_color) < 384 else (0, 0, 0)

    # テキスト描画
    draw.text((5, 5), text, fill=text_color)

    return image


def create_test_dataset() -> None:
    """テストデータセット全体を作成"""

    # ベースディレクトリ
    base_dir = (
        Path(__file__).parent.parent.parent / "data/machineLearning/image/test_dataset"
    )

    # クラス定義
    classes = ["oak", "pine", "birch"]

    # 各クラスに画像を作成
    for class_name in classes:
        class_dir = base_dir / class_name

        # ディレクトリを作成（存在しない場合）
        class_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating images for class: {class_name}")

        # 各クラスに10枚の画像を作成
        for i in range(1, 11):
            image_name = f"image_{i:02d}.jpg"
            image_path = class_dir / image_name

            # 画像作成
            image = create_test_image(
                width=128,
                height=128,
                class_name=class_name,
                image_name=f"img_{i:02d}",
                color_scheme=None,
            )

            # 保存
            image.save(image_path, "JPEG", quality=85)
            logger.info(f"  Created: {image_path}")

    logger.success("\nテストデータセット作成完了！")
    logger.info(f"場所: {base_dir.absolute()}")
    logger.info("構造:")
    logger.info("dataset/")
    logger.info("└── test_dataset/")
    for class_name in classes:
        logger.info(f"    ├── {class_name}/")
        logger.info("    │   ├── image_01.jpg")
        logger.info("    │   ├── image_02.jpg")
        logger.info("    │   ├── ...")
        logger.info("    │   └── image_10.jpg")


if __name__ == "__main__":
    create_test_dataset()
