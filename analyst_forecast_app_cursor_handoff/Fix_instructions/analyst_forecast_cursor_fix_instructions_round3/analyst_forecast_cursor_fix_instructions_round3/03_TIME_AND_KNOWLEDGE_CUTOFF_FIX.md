# 03 — 発言時点とknowledge cutoffの逆流防止

## 目的

AI実行日、取得日、公開後の情報、市場結果を発言時点以前の情報として扱うことを防ぎます。P05～P13と修正候補を同じ時点規則で検証します。

## Cursorへ渡す依頼文

    00_START_HERE.md、01、02に従い、source time、forecast time、knowledge cutoffの検証を一元化してください。commit・pushはしないでください。

    時刻の意味を混同しないでください。

    - recorded_at: 実際に収録・発言した時刻
    - published_at: 一般公開された時刻
    - retrieved_at: システムが取得した時刻
    - made_at: 予想が実際に表明された時刻。分からなければ未知のまま。
    - publicly_available_at: 一般利用可能になった時刻
    - executed_at: AI処理を実行した時刻。後日でよい。
    - knowledge_cutoff: 意味判断に使用した外部知識の上限時刻

    sourceごとにPythonがallowed knowledge boundaryを決める共通関数を作ってください。

    - YouTube: recorded_atが分かればrecorded_at。なければpublished_atを代替利用した事実をbasisとして保存する。
    - 本人執筆blog/X/Web: 原則published_at。本文でそれ以前の発言時刻が明示され、原文根拠を保存した場合だけmade_atを別に持てる。
    - 第三者記事の直接引用: 引用の発言時刻が明示されればその時刻。明示されなければpublished_atをpublic boundaryとするが、実発言時刻を捏造しない。
    - 日時情報がない場合、今日、取得日、system defaultを発言者明示値として入れない。未解決・評価不能・要レビューのいずれかにする。

    P05/P07:

    - payload knowledge_cutoffがallowed boundaryを超えたらrejectする。
    - segmentごとに異なる発言時刻を使う場合、根拠offsetとtime basisを保存する。

    P06/P09:

    - review knowledge_cutoffがreview対象sourceのallowed boundaryを超えたらrejectする。
    - corrected_payload内部のknowledge_cutoff、made_at、publicly_available_atも完全検証する。
    - review実行日が現在でもよいが、後発情報を根拠に使ってはならない。

    P08:

    - made_at <= publicly_available_atを維持する。
    - made_at_source=source_metadataならsourceのrecorded_at/published_atと決定的に整合させる。
    - made_at_source=context_inferredなら根拠segment、推定理由、time precisionを必須にして別AI reviewへ回す。
    - made_at_source=unknownでは勝手なdatetimeを埋めない。Schema/DB上でnullableまたは明示的unknownを表現し、正式な時期評価・対象解決へ進ませない。
    - 公開時刻を実発言時刻として保存する場合は、made_atではなくpublicly_available_atに保存する。

    P11/P12/P13:

    - payload knowledge_cutoff <= verified made_atまたは明示した保守的boundary。
    - P11全candidateのknowledge_cutoffとexisted_atを検証する。
    - P12 corrected_candidateのknowledge_cutoffとexisted_atも同じ検証を行う。
    - P13がP12 correctionを選ぶ場合も再検証する。
    - made_atが未解決なら、後発情報を使わず対象解決不能またはAI再レビューへ進める。

    市場値・評価結果のファイル、table、cache pathをP05～P13のprompt inputへ含めないことを自動testで確認してください。

## 必須テスト

1. P05 cutoff > recorded_atを拒否する。
2. P07 cutoff > published_atを拒否する。
3. P06 cutoff > source boundaryを拒否する。
4. P09 cutoff > source boundaryを拒否する。
5. P08 made_at > publicly_available_atを拒否する。
6. source_metadataのmade_atがsource metadataと矛盾したら拒否する。
7. unknown日時をtodayやretrieved_atで確定しない。
8. P11 cutoff > made_atを拒否する。
9. P11 candidate existed_at > made_atを拒否する。
10. P12 corrected candidate cutoff > made_atを拒否する。
11. P12 corrected candidate existed_at > made_atを拒否する。
12. P13で未来情報を含むP12 correctionを選べない。
13. 後日公開された録画でrecorded_atとpublished_atを別々に保持する。
14. prompt snapshotへ市場結果・evaluation cacheが入力されない。

## 完了条件

- P05～P13と修正版で同じ時点規則が使われる。
- executed_atが後日でも、知識上限は発言時点以前に固定される。
- 日時不明を偽の確定日へ変換しない。

