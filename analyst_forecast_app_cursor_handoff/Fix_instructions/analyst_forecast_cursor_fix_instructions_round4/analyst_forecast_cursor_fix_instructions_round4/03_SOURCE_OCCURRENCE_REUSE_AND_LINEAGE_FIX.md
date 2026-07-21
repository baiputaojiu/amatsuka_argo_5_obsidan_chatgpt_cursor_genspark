# 03 — 別source occurrenceへの安全な前処理再利用

## 目的

同一rawの別URL/source occurrenceでP05/P07を再利用済みと表示しながら、P08ではsource不一致として拒否する矛盾を解消します。raw同一性とsource occurrenceの違いを保ち、再利用先に適用可能な上流artifactを明示します。

## Cursorへ渡す依頼文

```text
00～02に従い、P05/P07再利用とP08 upstream検証を修正してください。commit・pushはしないでください。

raw artifactとsource occurrenceを混同しないでください。

- raw artifact: bytes/hashで識別する不変の原文実体。
- source occurrence: URL、媒体、公開日時、取得日時、分析対象者との関係を持つ観測事実。
- AI preprocessing artifact: 特定条件でrawを解釈した成果物。
- artifact applicability: その成果物をどのrun/source occurrenceでP08の上流として使えるか。

実装は、次のどちらか一貫した方式にしてください。

A. 推奨: 再利用先source occurrenceに紐づくderived preprocessing artifactを作る。
- target run_id/source_idを持つ。
- reused_from_artifact_id、raw_artifact_id、original output hash、reuse条件、作成日時を持つ。
- segmentをtarget artifactへコピーまたはimmutable segmentへのassociationで参照し、target側stable segment IDをP08から使える。
- AIは再実行しないが、由来とpayload hashを監査できる。

B. association方式:
- artifact本体をsource-neutralにし、artifactとrun/source occurrenceの多対多associationを正本にする。
- P08 upstream検証はartifact.source_idの単純一致ではなく、有効なassociationを検証する。
- segment/evidenceもどのsource occurrenceに適用されたか追跡できる。

RunSourceRecord.latest_ai_artifact_idへ別sourceのartifact IDを代入するだけの実装は不可です。workflowがacceptedと判断する条件とP08がupstreamを認める条件を同じserviceで判定してください。

再利用を許可する最小条件:

- raw_artifact_idまたはraw hashが完全一致する。
- analyst_idが一致する。
- mediumが一致し、YouTube=P05、非YouTube=P07の規則が一致する。
- prompt family、固定Schema major version、意味処理model/config/versionが互換である。
- original artifactがacceptedかつ未supersededである。
- original knowledge_cutoffがtarget sourceのallowed boundaryを越えない。
- source metadataに依存するsegment time/speaker判定がtarget occurrenceでも矛盾しない。
- target sourceに独自のaccepted preprocessingがある場合は、それを勝手に上書きしない。

URL、retrieved_at、published_atが違うこと自体はsource occurrenceとして保持してください。同じrawでもtarget boundaryがoriginal cutoffより早い場合は再利用不可とし、P05/P07を新規実行させてください。

P08検証:

- upstream artifactがtarget run/sourceへ明示的にapplicableであること。
- raw hash、analyst、medium、prompt、accepted statusを再確認すること。
- evidence offset/segment/speaker検証はtarget source rawに対して通常どおり行うこと。
- 別sourceのartifactをIDだけ差し替えた偽装入力を拒否すること。
- target occurrenceに生成されたderived/association済みartifactなら正常受理すること。

再利用は冪等にしてください。同じsourceへ再度reuseを要求してもderived artifact/association/segmentを増やさず、既存の有効なlineageを返してください。
```

## 必須テスト

1. 同一raw・同一analyst・同一medium・別URLでsource 2へ前処理を再利用できる。
2. source 2のP08が再利用artifactを参照してacceptedになる。
3. source 2のevidence/segment/raw lineageをsource 2から原artifactまで追跡できる。
4. 同じreuse操作を2回行ってもartifact/association/segmentが増えない。
5. raw hash違いでは再利用しない。
6. analyst違いでは再利用しない。
7. medium違いでは再利用しない。
8. prompt/schema/model互換性違いでは再利用しない。
9. source 2 boundaryより後のcutoffを持つartifactを再利用しない。
10. superseded/rejected preprocessing artifactを再利用しない。
11. target sourceに独自accepted artifactがある場合に上書きしない。
12. associationなしの別source artifact IDをP08へ渡すと拒否する。
13. workflowのpreprocess済み判定とP08 upstream判定が一致する。
14. 別runでも条件を満たせば安全に再利用でき、run/source lineageを保持する。

## 完了条件

- workflow上の「再利用済み」がP08でも実際に利用可能である。
- raw重複排除のためにURL/source occurrenceを失わない。
- 再利用可否がhashと明示条件で決まり、AI出力の自己申告に依存しない。

