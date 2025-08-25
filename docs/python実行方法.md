UVを使って実行する
IMPORTは全て絶対インポートを使う
SRC配下にpythonスクリプトを集める

以下の内容を`.vscode/launch.json`に追加することで、F5実行で絶対インポートが効くようになる。
```json
{
    "name": "Python: Current File (src layout)",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",                 // “開いているファイル”が走る
    "cwd": "${workspaceFolder}",
    "env": { "PYTHONPATH": "${workspaceFolder}/src" },
    "console": "integratedTerminal",
    "justMyCode": true
}
```