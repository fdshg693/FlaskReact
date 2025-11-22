---
description: 特定のフォルダ専用のレビューをAIに依頼するためのプロンプトテンプレートです。target_directoryに目的のフォルダパスを指定してください。
name: Generate-folder-specific-review
argument-hint: target_directory=目的のフォルダパス
agent: Folder-Specific-Agent
model: Claude Sonnet 4.5
tools: ['edit/createFile', 'edit/createDirectory', 'edit/editFiles', 'search', 'usages', 'todos', 'runSubagent']
---
### ${input:target_directory:"ルートディレクトリ"}配下のレビューを行なってください。
- ${input:target_directory:"ルートディレクトリ"}/.github/review_guideline.mdを読んで、レビューの方針を理解してください。
- 観点ごとにタスクを分割して、#tool:runSubagentを利用してサブエージェントにレビューを依頼してください。
- サブエージェントの結果を元に、冗長にならないようにして、10点満点の点数＋優先度の高い改善点に絞った上で、${input:target_directory:"ルートディレクトリ"}/.github/reviewフォルダの配下に、prompts_review.mdとしてレビュー結果をまとめてください。（すでに存在する場合は、一旦ファイルを削除してから新規作成してください）1
- 日本語で記述してください。