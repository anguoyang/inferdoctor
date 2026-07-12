# Dify Local / Private RAG クイックスタート

このドキュメントは、InferDoctor v0.7 開発版の Dify 連携を使って、ローカルまたはプライベートなモデルエンドポイントにつながる RAG アプリを準備するための短い手順です。

InferDoctor は Dify を置き換えるものではありません。Dify は Chatflow、Workflow、Knowledge、API、DSL import/export を担当します。InferDoctor は、スターターキットの検証、スモークテスト、TTFT と総レイテンシの確認、ベースライン比較、最適化ガイドを担当します。

## オフラインでキットを作成する

```bash
inferdoctor dify template export local-private-rag --output ./dify-local-private-rag
inferdoctor dify validate ./dify-local-private-rag
inferdoctor dify smoke --kit ./dify-local-private-rag --dry-run
```

この段階では Dify には接続しません。DSL の自動インポート、Knowledge Base の作成、文書アップロード、モデルダウンロード、ランタイムのインストールは行いません。

## Dify で手動設定する

1. `dify_app.yaml` を確認します。
2. Dify ワークスペースで手動でインポート、または同じ構成を再作成します。
3. ローカル / LAN / プライベートのモデルプロバイダーを設定します。
4. Knowledge Base を選択します。
5. アプリを Publish します。
6. アプリ API Key を作成します。

## API Key を安全に設定する

```bash
export DIFY_API_BASE_URL=http://127.0.0.1:5001/v1
export DIFY_APP_API_KEY=your-app-api-key
```

API Key をコマンドライン引数として直接渡さないでください。シェル履歴に残る可能性があります。

## ライブのスモークテスト

```bash
inferdoctor dify check --base-url "$DIFY_API_BASE_URL"
inferdoctor dify smoke --base-url "$DIFY_API_BASE_URL"
inferdoctor dify perf --base-url "$DIFY_API_BASE_URL" --runs 2 --warmup 1 --format json --output dify-perf.json
```

LAN / プライベートエンドポイントでは `--allow-non-local` を明示してください。Dify Cloud などの公開エンドポイントでは、意図して安全な短いクエリを送る場合だけ `--allow-public` を使います。

## 比較と最適化

```bash
inferdoctor perf baseline create --report dify-perf.json --name dify-before
inferdoctor perf compare before.json after.json
inferdoctor dify optimize --report after.json --kit ./dify-local-private-rag
```

これらは正式なベンチマークではありません。モデル品質、同時実行、長時間負荷、Dify 内部ワーカーの詳細なプロファイルは測定しません。

秘密情報や社内文書をスモークテストに使わないでください。
