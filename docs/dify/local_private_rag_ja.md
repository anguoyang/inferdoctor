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


## Self-host Reliability Doctor

v0.7 開発版では、Dify の self-host 環境を読み取り専用で確認する診断コマンドも追加されています。

```bash
inferdoctor dify selfhost preflight --compose-file ./docker-compose.yaml
inferdoctor dify selfhost inspect --compose-file ./docker-compose.yaml
inferdoctor dify connectivity check --compose-file ./docker-compose.yaml --endpoint http://192.168.1.20:8000/v1 --runtime openai-compatible --role chat --allow-non-local
inferdoctor dify evidence collect --compose-file ./docker-compose.yaml --since 10m --output evidence.json
inferdoctor dify evidence explain evidence.json --format markdown --output diagnosis.md
```

Dify はアプリケーションを構築・実行する基盤です。InferDoctor は、ホスト、Docker Compose、Plugin Daemon、Sandbox、SSRF Proxy、モデルエンドポイント、Knowledge 関連コンポーネントが連携できているかを確認します。

これらの診断は読み取り専用です。Dify のインストール、コンテナの起動・停止・再起動、Docker イメージの取得、`.env` の変更、秘密情報の収集、完全なログの保存は行いません。

Cloud 版 Dify では、明示的に指定された App API に対する外部観測はできます。ただし、Plugin Daemon や SSRF Proxy などの深い原因切り分けには self-host 側の証拠が必要です。

これは正式なベンチマークではありません。スモークテストと根本原因候補の整理を目的とした、安全で範囲を限定した診断です。
