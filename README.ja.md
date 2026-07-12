# InferDoctor 日本語クイックスタート

[English README](https://github.com/anguoyang/inferdoctor/blob/main/README.md)

**InferDoctor は、ローカルまたはセルフホスト AI アプリの診断、セットアップ、性能 UX 改善を支援する軽量 CLI です。**

InferDoctor は、Ollama、vLLM、SGLang、Xinference、Dify、CUDA、NVIDIA ドライバ、llama.cpp server、LM Studio、Open WebUI などを安全に確認し、ローカル AI スタックがなぜ動かないのか、次に何を試すべきかを示します。

```bash
inferdoctor
```

モデルを選ぶだけのツールではありません。InferDoctor は、診断、スタック計画、starter template、テンプレート検証、軽量な性能 smoke test、baseline 保存、最適化前後の比較、TTFT / streaming / RAG UX の改善アドバイスまでをつなげます。

## インストール

PyPI からインストールできます。

```bash
pip install inferdoctor
```

開発版を GitHub から試す場合:

```bash
python -m pip install "git+https://github.com/anguoyang/inferdoctor.git@dev"
```

ローカル開発用に clone する場合:

```bash
git clone https://github.com/anguoyang/inferdoctor.git
cd inferdoctor
python -m pip install -e ".[dev]"
```

## クイックスタート

```bash
inferdoctor                                      # 全体のヘルスチェック
inferdoctor --language ja                       # 日本語ヘルスダッシュボード
inferdoctor template list                       # starter template の一覧
inferdoctor quickstart customer-service          # guided setup plan
inferdoctor stack plan --goal customer-service  # 目的に合う手順を表示
inferdoctor template create customer-service --output ./customer-service-demo
inferdoctor template validate ./customer-service-demo
inferdoctor template smoke-test ./customer-service-demo
```

生成されたテンプレートは、OpenAI 互換のローカル endpoint を前提にしています。`config.yaml` または `.env` で Ollama、LM Studio、vLLM、SGLang などの endpoint を指定できます。

## 性能 UX smoke test

ローカル AI アプリでは、endpoint が到達可能なだけでは不十分です。ユーザー体験では、最初の文字が早く出るか、streaming が有効か、RAG の検索中に進捗が見えるか、cold start がデモを遅くしないかが重要です。

```bash
inferdoctor perf endpoint --endpoint http://127.0.0.1:8000/v1 --model local-model
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --runs 2 --warmup 1
inferdoctor perf baseline create --report before.json --name before
inferdoctor perf compare before.json after.json
inferdoctor optimize plan --report after.json --goal customer-service
inferdoctor optimize endpoint --runtime vllm --vram 24 --model-size 14b --streaming
inferdoctor optimize rag --top-k 8 --ttft 2.5 --streaming
```

InferDoctor v0.6 では、次のような性能 UX 情報を扱います。

- TTFT: time to first token。最初のユーザー可視テキストが出るまでの時間です。
- total latency: レスポンス完了までの総時間です。
- generation duration: 最初の出力から完了までの生成時間です。
- TPS: tokens per second。API usage が信頼できる場合は exact、なければ estimated と明示します。
- bounded runs / warmup: 最大実行回数を制限し、cold / warm の差を軽く確認します。
- baseline: sanitized な smoke test 結果を保存します。
- compare: 最適化前後の TTFT、latency、TPS、成功率を比較します。
- optimization plan: 観測値に基づいて検証可能な次の手順を出します。
- streaming check: stream=true が実際に user-visible content を返すか確認します。

これらは formal benchmark ではありません。短いプロンプトで行う timeout-bounded smoke test です。モデル品質、長時間負荷、最大スループット、実運用 SLA は測定しません。

## RAG と endpoint の最適化アドバイス

```bash
inferdoctor optimize endpoint --runtime ollama --streaming --ttft 1.5 --tps 40
inferdoctor optimize endpoint --runtime vllm --vram 24 --model-size 14b --quant q4
inferdoctor optimize rag --docs 1000 --chunks 5000 --top-k 8 --ttft 2.5 --streaming
```

アドバイスはヒューリスティックです。InferDoctor は、与えられた観測値から、streaming、warmup、context 長、model size、quantization、RAG の top_k、rerank latency、retrieval progress などを見直すための実用的な次の手順を提示します。

## first-step i18n

v0.5 以降では、ヘルスダッシュボードと `inferdoctor check` の console summary について、英語・中国語・日本語の first-step i18n を提供します。

```bash
inferdoctor --language auto
inferdoctor --language zh
inferdoctor --language ja
inferdoctor check --language en
```

対象外の command、生成テンプレート、JSON schema、Markdown report の機械可読フィールド名は英語のまま残る場合があります。これは自動化や issue report の互換性を保つためです。

## 対応しているチェック

| 対象 | 内容 |
| --- | --- |
| System | OS、Python、CPU アーキテクチャ、メモリ |
| NVIDIA | `nvidia-smi`、GPU 名、VRAM、ドライバ |
| CUDA | `nvcc`、CUDA toolkit、関連環境変数 |
| Ollama | CLI と `/api/tags` |
| vLLM / SGLang | OpenAI 互換 `/v1/models` |
| llama.cpp server / LM Studio | OpenAI 互換 API の疎通 |
| Xinference / Dify | SDK なしの軽量 HTTP 診断 |
| Open WebUI | Web endpoint の疎通 |
| Docker | CLI と daemon の疎通。コンテナは起動しません |

## テンプレート機能

InferDoctor は、次のような starter project を生成できます。

- customer-service: FAQ 対応チャットボット
- restaurant-ordering: メニューと注文ポリシーを使う接客アシスタント
- local-doc-qa: Markdown ドキュメントの軽量 Q&A

テンプレート生成は指定した出力ディレクトリにファイルを書くだけです。依存関係のインストール、モデルのダウンロード、推論実行は行いません。

## ローカル / LAN / プライベート endpoint

InferDoctor は localhost だけでなく、自分で管理する LAN または private OpenAI-compatible endpoint も扱えます。live smoke test で非ローカル endpoint に短いプロンプトを送る場合は、明示的に `--allow-non-local` を付けます。public endpoint へ自動的に送信することはありません。endpoint URL 内の credential や key らしき query は report や baseline で redaction されます。

## 安全ポリシー

InferDoctor は軽量で、デフォルトでは読み取り専用です。

- AI runtime、CUDA、GPU framework をインストールしません。
- モデルをダウンロードしません。
- systemd、Docker container、OS 設定を変更しません。
- 長時間 benchmark や負荷試験を実行しません。
- GPU がない CPU-only マシンでも動作します。
- live endpoint smoke test は短く、timeout-bounded です。

## 次に読むもの

- [English README](https://github.com/anguoyang/inferdoctor/blob/main/README.md)
- [Getting Started](https://github.com/anguoyang/inferdoctor/blob/main/docs/getting_started.md)
- [Templates](https://github.com/anguoyang/inferdoctor/blob/main/docs/templates.md)
- [Recommendations](https://github.com/anguoyang/inferdoctor/blob/main/docs/recommendations.md)
- [Model Fit Advisor](https://github.com/anguoyang/inferdoctor/blob/main/docs/model_fit.md)
- [Performance Metric Definitions](https://github.com/anguoyang/inferdoctor/blob/main/docs/performance/metric_definitions.md)
