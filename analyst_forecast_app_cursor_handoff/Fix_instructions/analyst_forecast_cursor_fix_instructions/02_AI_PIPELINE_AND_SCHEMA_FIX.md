# 02 — AI処理分離・Schema・独立レビュー修正

## 目的

P05、P08、P11、P12、必要時P13を別工程として記録し、「予想なし」「対象解決不能」を捏造なしで処理できるようにする。

## コピペ用依頼文

```text
REMEDIATION_PLAN.mdを確認し、AI処理パイプラインとSchemaを修正してください。対象外の市場評価や対話CLIはまだ変更しないでください。

現在の問題は、P08の1つのJSONへ予想抽出と対象解決・独立レビューを混在させ、review_resultという文字列の存在だけで検証済み扱いしていることです。P11・P12の実出力をPythonが独立して検証・記録・統合できる構造へ変更してください。

最低限、次の論理成果物を分離してください。
1. P05 YouTube整理・話者推定結果
2. P08 予想抽出結果
3. P11 予測対象候補・代理指標提案
4. P12 独立レビュー結果
5. P13 不一致時だけの裁定結果

物理テーブル数やクラス名は既存設計に合わせて決めて構いませんが、各成果物には以下を保存してください。
- 独立ID
- run_id、source_id、対象forecast/componentへの参照
- prompt_id、prompt_version、environment、model、executed_at
- input hash、output hash、Schema version
- confidence、根拠、knowledge_cutoff
- classificationとresolution status
- proposal/review/adjudicationの参照関係

P12はP11と別のprompt executionでなければverifiedにならないよう、Pythonで検査してください。同じJSON内にreview_result文字列があるだけではverifiedにしないでください。別実行であることを保証しつつ、高性能モデルを使用する方針を維持してください。人間承認は追加しません。

P11では最大3候補を表現できるようにしてください。発言者が明示した対象を最優先にし、テーマ・業種では指数、ETF、固定バスケット等の候補、ticker、取引所、通貨、ウェイト、存在時点、根拠を保存します。P12は各候補を独立にaccept / correct / reject / unresolvedで判定し、不一致時だけP13へ進めます。

次のケースを正式に表現できるSchemaへ変更してください。
- 情報源を処理したが評価対象予想が0件：forecasts=[]を許可し、processed_no_forecast等の状態を保存する。
- 対象は抽出できたが適切な評価用symbolがない：symbolとcurrencyをnullableにし、mapping_status=unresolvable、unevaluable_reasonを必須にする。
- 複数代理指標：1～3候補とweights、または感度分析対象を保存する。
- 判断不能な話者：unknownを許可し、対象者本人として強制登録しない。
- 低確信度または高重要度：AIレビュー待ちへ送る。高重要度の理由を保存する。

P08は予想抽出に集中させ、P11/P12で確定すべきsymbolやverified状態をP08へ要求しないでください。P08は、明示されたraw target labelと対象解決待ち状態までを出力できればよい構造にします。

P05用Schemaと取込みコマンドを追加し、rawを変更せず、processed側へセグメント、normalized_text、speaker_candidate、speaker_confidence、根拠、raw offsetを保存してください。P05完了後にP08がその成果物を入力として使えるようにします。YouTube以外はP07相当の整理へ拡張可能な境界にしてください。

既存Schema 1.0.0は既存fixtureの読込み互換を可能な範囲で維持してください。新しいSchema versionを導入し、必要なAlembic migrationを追加してください。既存データを捨てたり、DBを作り直したりしないでください。

案件生成プロンプトも修正してください。
- 各プロンプトへ具体的入力パスと出力パスを書く。
- P08、P11、P12の出力形式を個別Schemaへ固定する。
- P12へ市場結果を渡さない。
- P11とP12の出力を手作業でP08へ貼り戻させない。
- 各プロンプトはコピペだけで実行可能な依頼文にする。

必須テスト：
- 予想0件の情報源を正常取込みできる。
- unresolvable targetをdummy symbolなしで登録できる。
- P11単独ではmappingがverifiedにならない。
- P12の別prompt executionがない場合は拒否する。
- P11とP12が一致した場合だけmappingをlockできる。
- 不一致時はP13待ちになる。
- 最大3候補と複数代理指標を保存できる。
- unknown speakerを本人発言として登録しない。
- 低確信度または高重要度がAIレビュー待ちになる。
- 発言日時より後のknowledge_cutoffを拒否する。
- 市場結果フィールドが対象解決入力に混入した場合は拒否または監査警告にする。
- 既存Schema 1.0.0 fixtureの互換試験。

Ruff、mypy、通常pytest、Alembic upgrade/checkを実行し、IMPLEMENTATION_STATUS.mdとREMEDIATION_PLAN.mdを更新してください。完了後もcommitやpushは行わないでください。
```

