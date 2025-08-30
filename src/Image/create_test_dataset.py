#!/usr/bin/env python3
"""
テストデータセット作成スクリプト
木材分類のテスト用にダミー画像を生成します
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_test_image(width=128, height=128, class_name="", image_name="", color_scheme=None):
    """テスト用のダミー画像を作成"""
    
    # クラスごとの色スキーム
    color_schemes = {
        'oak': [(139, 69, 19), (160, 82, 45), (205, 133, 63)],      # 茶色系（オーク）
        'pine': [(34, 139, 34), (46, 125, 50), (76, 175, 80)],      # 緑系（パイン）
        'birch': [(245, 245, 220), (255, 248, 220), (250, 235, 215)] # ベージュ系（バーチ）
    }
    
    colors = color_schemes.get(class_name, [(128, 128, 128), (160, 160, 160), (192, 192, 192)])
    
    # 背景色をランダムに選択
    bg_color = random.choice(colors)
    
    # 画像作成
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # 木目のような模様を追加
    for _ in range(random.randint(5, 15)):
        # ランダムな線を描画（木目風）
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        
        line_color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in bg_color)
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=random.randint(1, 3))
    
    # ノイズ追加
    pixels = np.array(image)
    noise = np.random.normal(0, 10, pixels.shape).astype(np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    
    # テキスト追加（デバッグ用）
    draw = ImageDraw.Draw(image)
    try:
        # フォント設定（システムフォントを使用）
        font_size = 12
        text = f"{class_name}\n{image_name}"
        
        # テキストの色（背景と対比）
        text_color = (255, 255, 255) if sum(bg_color) < 384 else (0, 0, 0)
        
        # テキスト描画
        draw.text((5, 5), text, fill=text_color)
        
    except:
        # フォントが利用できない場合はスキップ
        pass
    
    return image

def create_test_dataset():
    """テストデータセット全体を作成"""
    
    # ベースディレクトリ
    base_dir = "./dataset/test_dataset"
    
    # クラス定義
    classes = ['oak', 'pine', 'birch']
    
    # 各クラスに画像を作成
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        
        print(f"Creating images for class: {class_name}")
        
        # 各クラスに10枚の画像を作成
        for i in range(1, 11):
            image_name = f"image_{i:02d}.jpg"
            image_path = os.path.join(class_dir, image_name)
            
            # 画像作成
            image = create_test_image(
                width=128, 
                height=128, 
                class_name=class_name,
                image_name=f"img_{i:02d}",
                color_scheme=None
            )
            
            # 保存
            image.save(image_path, 'JPEG', quality=85)
            print(f"  Created: {image_path}")
    
    print(f"\nテストデータセット作成完了！")
    print(f"場所: {os.path.abspath(base_dir)}")
    print(f"構造:")
    print(f"dataset/")
    print(f"└── test_dataset/")
    for class_name in classes:
        print(f"    ├── {class_name}/")
        print(f"    │   ├── image_01.jpg")
        print(f"    │   ├── image_02.jpg")
        print(f"    │   ├── ...")
        print(f"    │   └── image_10.jpg")

if __name__ == "__main__":
    create_test_dataset()
