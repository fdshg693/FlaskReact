---
description: 特定のフォルダ専用のREADME.mdをAIに作成させるためのプロンプトテンプレートです。target_directoryに目的のフォルダパスを指定してください。
name: Generate-folder-specific-readme
argument-hint: target_directory=目的のフォルダパス
agent: agent
model: Claude Sonnet 4.5
tools: ['edit/createFile', 'edit/createDirectory', 'edit/editFiles', 'search', 'usages', 'todos', 'runSubagent']
---
### ${input:target_directory:"ルートディレクトリ"}/README.mdを新規作成または更新してください。
- ${input:target_directory:"ルートディレクトリ"}/.github/copilot-instructions.mdを読んでフォルダの概要を理解してください。
- 以下の内容を含むREADME.mdを簡潔に作成してください。
最大200行以内でお願いします。日本語で記述してください。コードスニペットなどは含めずに、あくまで文章で概要を説明してください。
  1. フォルダの目的と概要
  2. 主な機能と特徴
  3. 簡単なファイル構成
  4. トラブルシューティングやよくある質問
- コマンドから行数を取得して、200行に収まっているか確認してください。