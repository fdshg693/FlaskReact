---
description: 特定のフォルダ専用の.github/copilot-instructions.mdをAIに作成させるためのプロンプトテンプレートです。target_directoryに目的のフォルダパスを指定してください。
name: Generate-folder-specific-copilot-instructions
argument-hint: target_directory=目的のフォルダパス
agent: agent
model: Claude Sonnet 4.5
tools: ['edit/createFile', 'edit/createDirectory', 'edit/editFiles', 'search', 'usages', 'todos', 'runSubagent']
---
### ${input:target_directory:"ルートディレクトリ"}/.github/copilot-instructions.mdを新規作成または更新してください。
- まず、コードベース全体を調査してください。スコープが広い調査などは#tool:runSubagentを利用してサブエージェントに任せてください。
- ${input:target_directory:"ルートディレクトリ"}配下のコードベースに特化したGitHub Copilot向けのフォルダ概要を100~200行で記述してください。
- ${input:target_directory:"ルートディレクトリ"}以外のファイルに関しては、必要最小限の詳細に踏み込まない記述にとどめてください。
- 英語で記述してください。