from __future__ import annotations

import os
from typing import Dict, Mapping, Optional

SUPPORTED_LANGUAGES = ("auto", "en", "zh", "ja")

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "dashboard_title": "InferDoctor - Local AI Stack Health Check",
        "dashboard_health": "Overall Health: {score} / 100  ({label})",
        "dashboard_scores": "PASS 100 | WARN 60 | FAIL 0 | SKIP 85  (heuristic)",
        "dashboard_header": "Component   Status   Summary",
        "dashboard_divider": "----------- -------- --------------------------------------------------",
        "dashboard_top_fixes": "Top recommended fixes (most useful first):",
        "dashboard_no_fixes": "No immediate fixes recommended. Your detected stack looks healthy.",
        "dashboard_fix": "{index}. {component}: {issue}",
        "dashboard_likely_cause": "   Likely cause: {cause}",
        "dashboard_impact": "   Impact: {impact}",
        "dashboard_try": "   Try: {command}",
        "dashboard_config": "   {hint_label}: {config_hint}",
        "dashboard_hint_config": "Config",
        "dashboard_hint_note": "Note",
        "dashboard_detail": "Detailed diagnostics:",
        "dashboard_stack_summary": "Stack Summary: {pass_count} working | {warn_count} needs attention | {skip_count} optional/offline | {fail_count} failed",
        "dashboard_doctor_read_fail": "Doctor's read: At least one required check failed. Start with the first fix below.",
        "dashboard_doctor_read_warn": "Doctor's read: Some components need attention. Start with the first fix below.",
        "dashboard_doctor_read_skip": "Doctor's read: No hard failures detected. Skipped components are optional unless you use them.",
        "dashboard_doctor_read_pass": "Doctor's read: All detected checks passed. Your local AI stack looks ready.",
        "console_status": "[{status:<4}] {name:<10} {summary}",
        "console_detail": "       - {detail}",
        "console_suggestion": "       suggestion: {suggestion}",
        "console_raw_data": "       raw_data:",
        "console_status_pass": "PASS",
        "console_status_warn": "WARN",
        "console_status_fail": "FAIL",
        "console_status_skip": "SKIP",
        "console_component_system": "System",
        "dashboard_row": "{name:<11} {status:<8} {summary}",
        "markdown_title": "# InferDoctor Report",
        "markdown_generated": "Generated: `{generated_at}`",
        "markdown_header": "| Check | Status | Summary |",
        "markdown_details": "**Details**",
        "markdown_suggestions": "**Suggestions**",
        "markdown_raw_data": "**Raw data**",
        "profile_title": "# InferDoctor Safe Diagnostic Profile",
        "profile_generated": "Generated: `{generated_at}`",
        "profile_note": "> Safe to share by default: secrets, endpoint credentials, query strings, and home paths are redacted.",
        "profile_section_system": "## System",
        "profile_section_gpus": "## GPUs",
        "profile_section_commands": "## Commands",
        "profile_section_endpoints": "## Configured Endpoints",
        "profile_section_checker_summary": "## Checker Summary",
        "profile_section_top_fixes": "## Top Fixes",
        "profile_system_header": "| Field | Value |",
        "profile_gpus_header": "| GPU | VRAM | Driver |",
        "profile_commands_header": "| Command | Available | Path |",
        "profile_endpoints_header": "| Name | URL |",
        "profile_available_yes": "yes",
        "profile_available_no": "no",
        "profile_no_gpus": "No NVIDIA GPU data was detected.",
        "profile_likely_cause": "   - Likely cause: {cause}",
        "profile_try": "   - Try: `{command}`",
        "profile_config_hint": "   - Config hint: `{config_hint}`",
        "profile_no_fixes": "No immediate fixes recommended.",
    },
    "zh": {
        "dashboard_title": "InferDoctor - 本地 AI 堆栈健康检查",
        "dashboard_health": "整体健康度：{score} / 100  ({label})",
        "dashboard_scores": "通过 100 | 警告 60 | 失败 0 | 跳过 85  （启发式评分）",
        "dashboard_header": "组件         状态     摘要",
        "dashboard_divider": "----------- -------- --------------------------------------------------",
        "dashboard_top_fixes": "建议修复（按优先级排序）：",
        "dashboard_no_fixes": "没有发现紧急修复建议。检测到的堆栈看起来很健康。",
        "dashboard_fix": "{index}. {component}: {issue}",
        "dashboard_likely_cause": "   可能原因：{cause}",
        "dashboard_impact": "   影响：{impact}",
        "dashboard_try": "   尝试：{command}",
        "dashboard_config": "   {hint_label}：{config_hint}",
        "dashboard_hint_config": "配置",
        "dashboard_hint_note": "说明",
        "dashboard_detail": "详细诊断信息：",
        "dashboard_stack_summary": "堆栈摘要：{pass_count} 个正常 | {warn_count} 个需注意 | {skip_count} 个可选/离线 | {fail_count} 个失败",
        "dashboard_doctor_read_fail": "诊断结论：至少一个必需检查失败。建议先从下面的第一项开始排查。",
        "dashboard_doctor_read_warn": "诊断结论：某些组件需要注意。建议先从下面的第一项开始排查。",
        "dashboard_doctor_read_skip": "诊断结论：未检测到严重失败。被跳过的组件通常是可选项，除非您的应用依赖它们。",
        "dashboard_doctor_read_pass": "诊断结论：所有检测通过。您的本地 AI 堆栈看起来已准备就绪。",
        "console_status": "[{status:<4}] {name:<10} {summary}",
        "console_detail": "       - {detail}",
        "console_suggestion": "       建议：{suggestion}",
        "console_raw_data": "       原始数据：",
        "console_status_pass": "通过",
        "console_status_warn": "警告",
        "console_status_fail": "失败",
        "console_status_skip": "跳过",
        "console_component_system": "系统",
        "dashboard_row": "{name:<11} {status:<8} {summary}",
        "markdown_title": "# InferDoctor 报告",
        "markdown_generated": "生成时间：`{generated_at}`",
        "markdown_header": "| 检查项 | 状态 | 摘要 |",
        "markdown_details": "**详情**",
        "markdown_suggestions": "**建议**",
        "markdown_raw_data": "**原始数据**",
        "profile_title": "# InferDoctor 安全诊断档案",
        "profile_generated": "生成时间：`{generated_at}`",
        "profile_note": "> 默认可安全分享：密钥、端点凭据、查询字符串和主目录路径均已脱敏。",
        "profile_section_system": "## 系统",
        "profile_section_gpus": "## GPU 信息",
        "profile_section_commands": "## 命令",
        "profile_section_endpoints": "## 已配置的端点",
        "profile_section_checker_summary": "## 检查项摘要",
        "profile_section_top_fixes": "## 建议修复",
        "profile_system_header": "| 字段 | 值 |",
        "profile_gpus_header": "| GPU | 显存 | 驱动 |",
        "profile_commands_header": "| 命令 | 可用 | 路径 |",
        "profile_endpoints_header": "| 名称 | URL |",
        "profile_available_yes": "是",
        "profile_available_no": "否",
        "profile_no_gpus": "未检测到 NVIDIA GPU 数据。",
        "profile_likely_cause": "   - 可能原因：{cause}",
        "profile_try": "   - 尝试：`{command}`",
        "profile_config_hint": "   - 配置提示：`{config_hint}`",
        "profile_no_fixes": "没有发现紧急修复建议。",
    },
    "ja": {
        "dashboard_title": "InferDoctor - ローカルAIスタックヘルスチェック",
        "dashboard_health": "全体の健全性：{score} / 100  ({label})",
        "dashboard_scores": "成功 100 | 注意 60 | 失敗 0 | スキップ 85  （ヒューリスティック）",
        "dashboard_header": "コンポーネント   ステータス   概要",
        "dashboard_divider": "----------- -------- --------------------------------------------------",
        "dashboard_top_fixes": "推奨される対応（優先順）：",
        "dashboard_no_fixes": "すぐに必要な修正はありません。検出されたスタックは健康に見えます。",
        "dashboard_fix": "{index}. {component}: {issue}",
        "dashboard_likely_cause": "   考えられる原因: {cause}",
        "dashboard_impact": "   影響: {impact}",
        "dashboard_try": "   試す: {command}",
        "dashboard_config": "   {hint_label}: {config_hint}",
        "dashboard_hint_config": "設定",
        "dashboard_hint_note": "注記",
        "dashboard_detail": "詳細な診断情報：",
        "dashboard_stack_summary": "スタック概要：{pass_count} 件成功 | {warn_count} 件注意 | {skip_count} 件オプション/オフライン | {fail_count} 件失敗",
        "dashboard_doctor_read_fail": "診断結果：必要なチェックが少なくとも1つ失敗しました。まず下の最初の項目から確認してください。",
        "dashboard_doctor_read_warn": "診断結果：いくつかのコンポーネントに注意が必要です。まず下の最初の項目から確認してください。",
        "dashboard_doctor_read_skip": "診断結果：重大な失敗は検出されませんでした。スキップされたコンポーネントは、使わない場合は通常オプションです。",
        "dashboard_doctor_read_pass": "診断結果：検出されたチェックはすべて成功しました。ローカルAIスタックは準備できているようです。",
        "console_status": "[{status:<4}] {name:<10} {summary}",
        "console_detail": "       - {detail}",
        "console_suggestion": "       提案：{suggestion}",
        "console_raw_data": "       生データ：",
        "console_status_pass": "成功",
        "console_status_warn": "注意",
        "console_status_fail": "失敗",
        "console_status_skip": "スキップ",
        "console_component_system": "システム",
        "dashboard_row": "{name:<11} {status:<8} {summary}",
        "markdown_title": "# InferDoctor レポート",
        "markdown_generated": "生成日時：`{generated_at}`",
        "markdown_header": "| チェック | ステータス | 概要 |",
        "markdown_details": "**詳細**",
        "markdown_suggestions": "**提案**",
        "markdown_raw_data": "**生データ**",
        "profile_title": "# InferDoctor 安全診断プロファイル",
        "profile_generated": "生成日時：`{generated_at}`",
        "profile_note": "> デフォルトで安全に共有可能：シークレット、エンドポイント認証情報、クエリ文字列、ホームパスは伏せ字化されています。",
        "profile_section_system": "## システム",
        "profile_section_gpus": "## GPU 情報",
        "profile_section_commands": "## コマンド",
        "profile_section_endpoints": "## 設定済みエンドポイント",
        "profile_section_checker_summary": "## チェッカー概要",
        "profile_section_top_fixes": "## 推奨される対応",
        "profile_system_header": "| フィールド | 値 |",
        "profile_gpus_header": "| GPU | VRAM | ドライバ |",
        "profile_commands_header": "| コマンド | 利用可能 | パス |",
        "profile_endpoints_header": "| 名前 | URL |",
        "profile_available_yes": "はい",
        "profile_available_no": "いいえ",
        "profile_no_gpus": "NVIDIA GPU データは検出されませんでした。",
        "profile_likely_cause": "   - 考えられる原因: {cause}",
        "profile_try": "   - 試す: `{command}`",
        "profile_config_hint": "   - 設定のヒント: `{config_hint}`",
        "profile_no_fixes": "すぐに必要な修正はありません。",
    },
}


def _language_from_locale(value: str) -> Optional[str]:
    normalized = value.strip().split(".")[0].replace("-", "_").lower()
    if not normalized or normalized in {"c", "posix"}:
        return None
    if normalized.startswith("zh"):
        return "zh"
    if normalized.startswith("ja"):
        return "ja"
    if normalized.startswith("en"):
        return "en"
    return None


def detect_system_language(env: Optional[Mapping[str, str]] = None) -> str:
    environment = os.environ if env is None else env
    for key in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        raw_value = environment.get(key, "")
        for candidate in raw_value.split(":"):
            detected = _language_from_locale(candidate)
            if detected:
                return detected
    return "en"


def normalize_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in SUPPORTED_LANGUAGES:
        return normalized
    if normalized.startswith("zh"):
        return "zh"
    if normalized.startswith("ja"):
        return "ja"
    if normalized.startswith("en"):
        return "en"
    raise ValueError("Unsupported language: {0}".format(language))


def t(key: str, language: str = "auto", **kwargs: object) -> str:
    if language == "auto":
        language = detect_system_language()
    try:
        normalized_language = normalize_language(language)
    except ValueError:
        normalized_language = "en"
    translation = TRANSLATIONS.get(normalized_language, TRANSLATIONS["en"])
    template = translation.get(key, TRANSLATIONS["en"].get(key, key))
    return template.format(**kwargs)
