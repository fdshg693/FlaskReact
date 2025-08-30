UVを使って実行する（F5で実行するので、VSCODEの右下のPYthonファイルのパスが仮想環境になっている限り関係ない）
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
`.vscode/sample_launch.json`を参考にすること
