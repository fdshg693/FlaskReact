# pyrightの使い方

## 参考リンク
- [Github]https://github.com/microsoft/pyright
- [コマンド] https://github.com/microsoft/pyright/blob/main/docs/command-line.md

## VSCODE拡張機能
1. Pylance（推奨）
MicrosoftがPyrightをベースに開発した公式Python言語サーバーです。Pyrightの型チェック機能に加え、IntelliSense、自動補完、インポート整理などが統合されています。
拡張機能ID: ms-python.vscode-pylance
Pylanceをインストールすると、Python拡張機能（ms-python.python）も自動的に入ります。型チェックの厳格さは settings.json で調整できます：
```json
{
  "python.analysis.typeCheckingMode": "basic"  // "off", "basic", "standard", "strict"
}
```

2. Pyright単体

Pylanceの追加機能が不要で、純粋な型チェッカーだけ欲しい場合はこちら。OSSなので中身を確認したい場合にも適しています。
拡張機能ID: ms-pyright.pyright

## 型情報の与え方
`py,typed` ファイルをパッケージディレクトリに配置することで、そのパッケージが型情報を提供していることを示せます。これにより、Pyrightはそのパッケージ内の型ヒントを認識し、型チェックに利用します。
