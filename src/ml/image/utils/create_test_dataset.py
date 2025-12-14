#!/usr/bin/env python3
"""
テストデータセット作成スクリプト
木材分類のテスト用にダミー画像を生成します
"""

import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw

from config import PATHS, ensure_path_exists

# TODO: tqdmによるプログレスバー・typerによるCLI強化
# PYTEST追加

# ===== 定数 =====
DEFAULT_IMAGE_WIDTH = 128
DEFAULT_IMAGE_HEIGHT = 128
DEFAULT_IMAGES_PER_CLASS = 10
JPEG_QUALITY = 85
NOISE_LEVEL = 10
TEXT_POSITION = (5, 5)
LINE_WIDTH_RANGE = (1, 3)
COLOR_VARIATION_RANGE = (-30, 30)
WOOD_GRAIN_LINES_RANGE = (5, 15)


class RGBColor(NamedTuple):
    """RGB色を表現するNamedTuple

    Attributes:
        red: 赤成分 (0-255)
        green: 緑成分 (0-255)
        blue: 青成分 (0-255)
    """

    red: int
    green: int
    blue: int

    def to_tuple(self) -> tuple[int, int, int]:
        """PIL互換のタプル形式に変換

        Returns:
            tuple[int, int, int]: (R, G, B)形式のタプル
        """
        return (self.red, self.green, self.blue)

    def adjust_brightness(self, min_val: int, max_val: int) -> "RGBColor":
        """明度をランダムに調整した新しい色を返す

        Args:
            min_val: 調整値の最小値
            max_val: 調整値の最大値

        Returns:
            RGBColor: 調整後の色
        """
        return RGBColor(
            red=max(0, min(255, self.red + random.randint(min_val, max_val))),
            green=max(0, min(255, self.green + random.randint(min_val, max_val))),
            blue=max(0, min(255, self.blue + random.randint(min_val, max_val))),
        )

    def luminance(self) -> int:
        """色の明度を計算

        Returns:
            int: RGB成分の合計値
        """
        return self.red + self.green + self.blue


def get_color_scheme_for_class(class_name: str) -> list[RGBColor]:
    """クラス名に対応する色スキームを取得

    Args:
        class_name: 木材のクラス名 ("oak", "pine", "birch" など)

    Returns:
        list[RGBColor]: クラスに対応する色のリスト

    Note:
        - oak (オーク): 濃い茶色系
        - pine (パイン): 緑系
        - birch (バーチ): 明るいベージュ系
        - その他: グレー系(デフォルト)
    """
    color_schemes: dict[str, list[RGBColor]] = {
        "oak": [
            RGBColor(139, 69, 19),  # サドルブラウン
            RGBColor(160, 82, 45),  # チョコレート
            RGBColor(205, 133, 63),  # ペルー
        ],
        "pine": [
            RGBColor(34, 139, 34),  # フォレストグリーン
            RGBColor(46, 125, 50),  # グリーン
            RGBColor(76, 175, 80),  # ライトグリーン
        ],
        "birch": [
            RGBColor(245, 245, 220),  # ベージュ
            RGBColor(255, 248, 220),  # コーンシルク
            RGBColor(250, 235, 215),  # アンティークホワイト
        ],
    }

    # デフォルト色(グレー系)
    default_colors = [
        RGBColor(128, 128, 128),
        RGBColor(160, 160, 160),
        RGBColor(192, 192, 192),
    ]

    return color_schemes.get(class_name, default_colors)


def add_wood_grain_texture(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    base_color: RGBColor,
    num_lines: int,
) -> None:
    """木目のようなテクスチャを画像に追加

    Args:
        draw: PIL ImageDrawオブジェクト
        width: 画像の幅
        height: 画像の高さ
        base_color: 基準となる色
        num_lines: 描画する線の数
    """
    for _ in range(num_lines):
        # ランダムな線を描画(木目風)
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)

        line_color = base_color.adjust_brightness(*COLOR_VARIATION_RANGE)
        line_width = random.randint(*LINE_WIDTH_RANGE)

        draw.line([(x1, y1), (x2, y2)], fill=line_color.to_tuple(), width=line_width)


def add_noise_to_image(image: Image.Image, noise_std: int = NOISE_LEVEL) -> Image.Image:
    """画像にガウシアンノイズを追加

    Args:
        image: 元画像
        noise_std: ノイズの標準偏差

    Returns:
        Image.Image: ノイズ追加後の画像
    """
    pixels = np.array(image)
    noise = np.random.normal(0, noise_std, pixels.shape).astype(np.int16)
    noisy_pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_pixels)


def add_debug_text(
    draw: ImageDraw.ImageDraw,
    class_name: str,
    image_name: str,
    bg_color: RGBColor,
) -> None:
    """デバッグ用テキストを画像に追加

    Args:
        draw: PIL ImageDrawオブジェクト
        class_name: クラス名
        image_name: 画像名
        bg_color: 背景色(テキスト色の決定に使用)
    """
    text = f"{class_name}\n{image_name}"

    # 背景色の明度に応じてテキスト色を決定
    # 明度が低い(暗い)場合は白、高い(明るい)場合は黒
    text_color = (
        RGBColor(255, 255, 255) if bg_color.luminance() < 384 else RGBColor(0, 0, 0)
    )

    draw.text(TEXT_POSITION, text, fill=text_color.to_tuple())


def create_test_image(
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    class_name: str = "",
    image_name: str = "",
) -> Image.Image:
    """テスト用のダミー画像を作成

    木材分類タスク用のテスト画像を生成します。
    各クラスに対応した色スキームを使用し、木目のようなテクスチャとノイズを追加します。

    Args:
        width: 画像の幅(ピクセル)
        height: 画像の高さ(ピクセル)
        class_name: 木材のクラス名("oak", "pine", "birch" など)
        image_name: 画像識別子(デバッグテキストに使用)

    Returns:
        Image.Image: 生成されたPIL画像オブジェクト

    Example:
        >>> img = create_test_image(128, 128, "oak", "img_01")
        >>> img.save("oak_test.jpg")
    """
    # クラスに対応する色スキームを取得
    colors = get_color_scheme_for_class(class_name)

    # 背景色をランダムに選択
    bg_color = random.choice(colors)

    # 画像作成
    image = Image.new("RGB", (width, height), bg_color.to_tuple())
    draw = ImageDraw.Draw(image)

    # 木目のようなテクスチャを追加
    num_grain_lines = random.randint(*WOOD_GRAIN_LINES_RANGE)
    add_wood_grain_texture(draw, width, height, bg_color, num_grain_lines)

    # ノイズ追加
    image = add_noise_to_image(image)

    # デバッグ用テキスト追加
    draw = ImageDraw.Draw(image)
    add_debug_text(draw, class_name, image_name, bg_color)

    return image


def create_test_dataset(
    base_dir: Path | None = PATHS.ml_image_data,
    classes: list[str] | None = ["oak", "pine", "birch"],
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    overwrite: bool = False,
) -> None:
    """テストデータセット全体を作成

    指定されたクラスごとにダミー画像を生成し、ディレクトリ構造を作成します。

    Args:
        base_dir: データセットの保存先ディレクトリ
        classes: 木材クラスのリスト
        images_per_class: 各クラスで生成する画像数
        image_width: 生成する画像の幅
        image_height: 生成する画像の高さ
        overwrite: 既存ファイルがある場合に上書きするかどうか
                   False の場合、既存ファイルが検出されると処理を中断

    Raises:
        OSError: ディレクトリの作成に失敗した場合
        FileExistsError: overwrite=False かつ既存ファイルが存在する場合

    Example:
        >>> create_test_dataset(
        ...     classes=["oak", "pine"],
        ...     images_per_class=5
        ... )
        >>> create_test_dataset(
        ...     classes=["oak", "pine"],
        ...     images_per_class=5,
        ...     overwrite=True
        ... )
    """
    if overwrite:
        logger.warning(
            "overwrite=True が指定されました。既存ファイルがあっても上書きされます。"
        )
    # 既存ファイルのチェック
    else:
        existing_files: list[Path] = []
        for class_name in classes:
            class_dir = base_dir / class_name
            if class_dir.exists():
                # ディレクトリ配下のファイルのみを検出（ディレクトリは無視）
                for item in class_dir.iterdir():
                    if item.is_file():
                        existing_files.append(item)

        if existing_files:
            logger.warning(f"既存ファイルが {len(existing_files)} 個見つかりました:")
            for file_path in existing_files[:5]:  # 最初の5件のみ表示
                logger.warning(f"  - {file_path}")
            if len(existing_files) > 5:
                logger.warning(f"  ... 他 {len(existing_files) - 5} 個")
            error_msg = (
                f"base_dir ({base_dir}) 配下に既存ファイルが存在します。"
                "上書きする場合は overwrite=True を指定してください。"
            )
            logger.error(error_msg)
            raise FileExistsError(error_msg)
    # 各クラスに画像を作成
    for class_name in classes:
        class_dir = base_dir / class_name

        # ディレクトリを作成(存在しない場合)
        try:
            ensure_path_exists(class_dir)
        except OSError as e:
            logger.error(f"Failed to create directory {class_dir}: {e}")
            raise

        logger.info(f"Creating images for class: {class_name}")

        # 各クラスに指定枚数の画像を作成
        for i in range(1, images_per_class + 1):
            image_name = f"image_{i:02d}.jpg"
            image_path = class_dir / image_name

            # 画像作成
            try:
                image = create_test_image(
                    width=image_width,
                    height=image_height,
                    class_name=class_name,
                    image_name=f"img_{i:02d}",
                )

                # 保存
                image.save(image_path, "JPEG", quality=JPEG_QUALITY)
                logger.debug(f"  Created: {image_path}")
            except Exception as e:
                logger.error(f"Failed to create image {image_path}: {e}")
                continue

    # 結果サマリー
    logger.success("\nテストデータセット作成完了！")
    logger.info(f"場所: {base_dir.absolute()}")
    logger.info(f"クラス数: {len(classes)}")
    logger.info(f"クラスあたりの画像数: {images_per_class}")
    logger.info(f"合計画像数: {len(classes) * images_per_class}")
    logger.info(f"画像サイズ: {image_width}x{image_height}")
    logger.info("\n構造:")
    logger.info("test_dataset/")
    for idx, class_name in enumerate(classes):
        is_last = idx == len(classes) - 1
        prefix = "└──" if is_last else "├──"
        logger.info(f"{prefix} {class_name}/")
        sub_prefix = "    " if is_last else "│   "
        logger.info(f"{sub_prefix}├── image_01.jpg")
        logger.info(f"{sub_prefix}├── image_02.jpg")
        logger.info(f"{sub_prefix}├── ...")
        logger.info(f"{sub_prefix}└── image_{images_per_class:02d}.jpg")


if __name__ == "__main__":
    create_test_dataset(base_dir=PATHS.ml_image_data, images_per_class=3)
