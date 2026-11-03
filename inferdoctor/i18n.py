from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

SUPPORTED_LANGUAGES = ("auto", "en", "zh", "ja")

# Nested translation dictionary.
# Use dot-path keys with t() for lookup, e.g. t("dashboard.title", language).
TRANSLATIONS: Dict[str, Dict[str, Any]] = {
    "en": {
        "dashboard": {
            "title": "InferDoctor - Local AI Stack Health Check",
            "health": "Overall Health: {score} / 100  ({label})",
            "scores": "PASS 100 | WARN 60 | FAIL 0 | SKIP 85  (heuristic)",
            "header": "Component   Status   Summary",
            "divider": "----------- -------- --------------------------------------------------",
            "top_fixes": "Top recommended fixes (most useful first):",
            "no_fixes": "No immediate fixes recommended. Your detected stack looks healthy.",
            "fix": "{index}. {component}: {issue}",
            "likely_cause": "   Likely cause: {cause}",
            "impact": "   Impact: {impact}",
            "try": "   Try: {command}",
            "config": "   {hint_label}: {config_hint}",
            "hint_config": "Config",
            "hint_note": "Note",
            "detail": "Detailed diagnostics:",
            "stack_summary": "Stack Summary: {pass_count} working | {warn_count} needs attention | {skip_count} optional/offline | {fail_count} failed",
            "doctor_read_fail": "Doctor's read: At least one required check failed. Start with the first fix below.",
            "doctor_read_warn": "Doctor's read: Some components need attention. Start with the first fix below.",
            "doctor_read_skip": "Doctor's read: No hard failures detected. Skipped components are optional unless you use them.",
            "doctor_read_pass": "Doctor's read: All detected checks passed. Your local AI stack looks ready.",
        },
        "console": {
            "status": "[{status:<4}] {name:<10} {summary}",
            "detail": "       - {detail}",
            "suggestion": "       suggestion: {suggestion}",
            "raw_data": "       raw_data:",
        },
        "health": {
            "label_healthy": "Healthy",
            "label_mostly_healthy": "Mostly healthy",
            "label_needs_attention": "Needs attention",
            "label_unhealthy": "Unhealthy",
            "impact_fail": "Likely blocking for this component until fixed.",
            "impact_warn": "Needs attention if your app depends on {service_name}.",
            "impact_skip": "Optional unless {service_name} is part of your local stack.",
            "cause_endpoint_404": "The service responded, but /v1/models was not found. The base URL may be missing or duplicating /v1.",
            "cause_endpoint_auth": "The endpoint requires authentication or the proxy blocks this route.",
            "cause_endpoint_invalid_json": "The URL may point to a web UI or a non-compatible API route.",
            "cause_endpoint_dify": "Dify is optional. If you run it locally, the service may be stopped or mapped to another port.",
            "cause_endpoint_generic": "The service may not be running, or it may be listening on another port.",
            "cause_nvidia": "The NVIDIA driver is absent, unhealthy, or nvidia-smi is not on PATH.",
            "cause_nvidia_noaction": "No action is needed on a CPU-only or non-NVIDIA machine.",
            "impact_nvidia": "Required only for NVIDIA GPU inference or CUDA-backed runtimes.",
            "cause_cuda_driver_available": "NVIDIA driver is available, but CUDA toolkit was not found. This may be OK if you only use prebuilt runtimes such as Ollama.",
            "cause_cuda_no_driver": "The CUDA toolkit is absent or nvcc is not on PATH.",
            "config_hint_cuda": "Install CUDA toolkit only if you need to compile CUDA workloads or use runtimes that require nvcc.",
            "impact_cuda": "Optional for CPU-only and many prebuilt runtimes; required for CUDA compilation.",
            "cause_docker": "The Docker CLI is missing, the daemon is stopped, or the current user cannot reach the Docker daemon.",
            "config_hint_docker": "InferDoctor only checks Docker status; it never starts containers.",
            "impact_docker": "Required only if your local AI stack runs services in containers.",
            "cause_generic": "The diagnostic could not confirm this component is healthy.",
            "impact_generic": "Review this component if it is part of your local AI stack.",
            "config_hint_endpoint": "endpoints.{name}: {endpoint}",
        },
        "checker": {
            "system": {
                "ok": "System information collected",
                "detail_os": "OS: {value}",
                "detail_python": "Python: {value}",
                "detail_arch": "Architecture: {value}",
                "detail_memory": "Available memory: {value}",
                "detail_memory_unavailable": "Available memory: unavailable",
            },
            "nvidia": {
                "not_found": "nvidia-smi was not found",
                "timed_out": "nvidia-smi timed out",
                "exec_error": "nvidia-smi could not be executed",
                "reported_error": "nvidia-smi reported an error",
                "no_gpu_data": "nvidia-smi ran but no GPU data was parsed",
                "detected": "{count} NVIDIA GPU(s) detected",
            },
            "cuda": {
                "not_found": "CUDA compiler was not found",
                "timed_out": "nvcc timed out",
                "exec_error": "nvcc could not be executed",
                "reported_error": "nvcc reported an error",
                "version_detected": "CUDA toolkit {version} detected",
                "version_not_parsed": "nvcc is available but its CUDA version was not parsed",
            },
            "docker": {
                "not_found": "Docker CLI was not found",
                "timed_out": "Docker daemon check timed out",
                "exec_error": "Docker CLI could not be executed",
                "daemon_not_reachable": "Docker CLI is installed but the daemon is not reachable",
                "reachable": "Docker CLI and daemon are reachable",
            },
            "ollama": {
                "not_found_not_reachable": "Ollama was not found and its API is not reachable",
                "installed_not_reachable": "Ollama is installed but its API is not reachable",
                "http_error": "Ollama API responded with HTTP {status}",
                "api_reachable_cli_missing": "Ollama API is reachable but the CLI is not on PATH",
                "ready": "Ollama CLI and API are available",
            },
            "endpoint": {
                "not_reachable": "{service_label} endpoint is not reachable",
                "reachable": "{service_label} endpoint is reachable",
                "requires_auth": "{service_label} endpoint requires authentication",
                "http_status": "{service_label} responded with HTTP {status}",
            },
            "openai": {
                "not_reachable": "{service_label} endpoint is not reachable",
                "requires_auth": "{service_label} is reachable but requires authentication",
                "not_found": "{service_label} models route returned HTTP 404",
                "http_status": "{service_label} returned HTTP {status}",
                "invalid_json": "{service_label} returned invalid JSON",
                "not_compatible": "{service_label} response is not OpenAI-compatible",
                "ready": "{service_label} OpenAI-compatible API is ready",
            },
        },
        "markdown": {
            "title": "InferDoctor Report",
            "generated_at": "Generated: `{datetime}`",
            "table_header": "| Check | Status | Summary |",
            "table_divider": "| --- | --- | --- |",
            "table_row": "| {name} | **{status}** | {summary} |",
            "section_title": "## {name}",
            "details_title": "**Details**",
            "suggestions_title": "**Suggestions**",
            "raw_data_title": "**Raw data**",
            "list_item": "- {item}",
        },
        "explain": {
            "title": "InferDoctor Explain: {topic}",
            "what_it_means": "What it means:",
            "common_causes": "Common causes:",
            "what_to_try_next": "What to try next:",
            "related_command": "Related InferDoctor command:",
        },
        "capacity": {
            "title": "InferDoctor Capacity Preview",
            "hardware_detected": "Detected hardware:",
            "workload_readiness": "Workload readiness:",
            "no_gpu": "No GPU detected on this machine.",
            "vram": "VRAM: {vram} GiB",
            "gpu": "GPU: {name}",
            "cpu_only": "CPU-only mode",
            "ready": "Ready",
            "possible": "Possible",
            "tight": "Tight",
            "unlikely": "Unlikely",
        },
        "optimize": {
            "title": "InferDoctor Optimization Advice",
            "current_situation": "Current situation",
            "possible_bottlenecks": "Possible bottlenecks",
            "quick_wins": "Suggested quick wins",
            "safe_next_command": "Safe next command",
        },
        "profile": {
            "title": "# InferDoctor Safe Diagnostic Profile",
            "section_system": "## System",
            "section_gpu": "## GPU",
            "section_commands": "## Commands",
            "section_endpoints": "## Endpoints",
        },
        "recommendation": {
            "title": "InferDoctor Stack Recommendation",
            "goal": "Goal: {goal}",
            "preference": "Preference: {preference}",
            "recommendation": "Recommendation:",
            "runtime_path": "Runtime paths:",
        },
        "setup": {
            "title": "InferDoctor Guided Setup",
            "plain_english": "Plain-English recommendation:",
            "step_by_step": "Step-by-step next actions:",
            "no_recommendation": "Not enough information to recommend a setup. Provide a --goal or answer the interactive prompts.",
            "what_to_build": "What do you want to build? [chatbot/document-qa/customer-service/restaurant-ordering/local-api/not-sure]: ",
            "what_preference": "What do you prefer? [easiest/performance/cpu/gpu]: ",
            "existing_runtime": "Existing runtime? [ollama/vllm/sglang/xinference/not-sure]: ",
        },
        "stack_plan": {
            "title": "InferDoctor Local AI Stack Plan",
            "required_components": "Required components:",
            "optional_components": "Optional components:",
            "step_by_step": "Step-by-step next actions:",
            "no_plan": "Not enough information to build a stack plan.",
        },
        "templates": {
            "title": "InferDoctor Local AI App Templates",
            "what_it_builds": "What it builds",
            "who_its_for": "Who it's for",
            "stack_fit": "Stack fit",
            "no_templates": "No templates are available.",
            "created_summary": "Created {count} file(s) for {template} at {output}",
            "compose_created_summary": "Created {count} Docker Compose file(s) for {template} at {output}",
            "build_what": "What to build:",
            "for_who": "For who:",
            "suitable_stack": "Suitable stack:",
        },
        "template_validation": {
            "title": "InferDoctor Template Validation",
            "checks": "Checks:",
            "passed": "All checks passed",
            "failed": "{fail} of {total} checks failed",
            "top_fixes": "Top fixes:",
            "next_command": "Next command to try:",
            "smoke_title": "InferDoctor Template Smoke Test",
            "smoke_results": "Smoke test results:",
            "all_passed": "All smoke tests passed",
        },
        "model_fit": {
            "title": "InferDoctor Model Fit Advisor",
            "request": "Request: {description}",
            "estimate": "Estimate: {description}",
            "runtime_guidance": "Runtime guidance:",
            "no_estimate": "Cannot estimate fit for the given parameters.",
        },
        "scenarios": {
            "title": "InferDoctor Scenario Readiness",
            "status": "Status: {status}",
            "reason": "Reason: {reason}",
            "next_step": "Next step: {next}",
            "no_scenarios": "No scenarios are available.",
        },
        "perf": {
            "smoke_title": "InferDoctor Performance UX Smoke Test",
            "mode": "Mode: {mode}",
            "endpoint": "Endpoint: {endpoint}",
            "model": "Model: {model}",
            "warmup": "Warmup: {count}",
            "runs": "Runs: {count}",
            "warning": "Warning: {warning}",
            "error": "Error: {error}",
            "result": "Results:",
            "ttft": "TTFT: {value}s",
            "tps": "TPS: {value}",
            "latency": "Latency: {value}s",
            "no_results": "No results to display.",
        },
        "runner": {
            "checker_failed": "Checker failed unexpectedly",
        },
    },
    "zh": {
        "dashboard": {
            "title": "InferDoctor - 本地 AI 堆栈健康检查",
            "health": "整体健康度：{score} / 100  ({label})",
            "scores": "PASS 100 | WARN 60 | FAIL 0 | SKIP 85  （启发式评分）",
            "header": "组件         状态     摘要",
            "divider": "----------- -------- --------------------------------------------------",
            "top_fixes": "建议修复（按优先级排序）：",
            "no_fixes": "没有发现紧急修复建议。检测到的堆栈看起来很健康。",
            "fix": "{index}. {component}: {issue}",
            "likely_cause": "   可能原因：{cause}",
            "impact": "   影响：{impact}",
            "try": "   尝试：{command}",
            "config": "   {hint_label}：{config_hint}",
            "hint_config": "配置",
            "hint_note": "说明",
            "detail": "详细诊断信息：",
            "stack_summary": "堆栈摘要：{pass_count} 个正常 | {warn_count} 个需注意 | {skip_count} 个可选/离线 | {fail_count} 个失败",
            "doctor_read_fail": "诊断结论：至少一个必需检查失败。建议先从下面的第一项开始排查。",
            "doctor_read_warn": "诊断结论：某些组件需要注意。建议先从下面的第一项开始排查。",
            "doctor_read_skip": "诊断结论：未检测到严重失败。被跳过的组件通常是可选项，除非您的应用依赖它们。",
            "doctor_read_pass": "诊断结论：所有检测通过。您的本地 AI 堆栈看起来已准备就绪。",
        },
        "console": {
            "status": "[{status:<4}] {name:<10} {summary}",
            "detail": "       - {detail}",
            "suggestion": "       建议：{suggestion}",
            "raw_data": "       原始数据：",
        },
        "health": {
            "label_healthy": "健康",
            "label_mostly_healthy": "大致健康",
            "label_needs_attention": "需要注意",
            "label_unhealthy": "不健康",
            "impact_fail": "此组件可能因此阻塞，直到问题修复。",
            "impact_warn": "如果您的应用依赖 {service_name}，则需要注意。",
            "impact_skip": "除非 {service_name} 是您本地堆栈的一部分，否则可选。",
            "cause_endpoint_404": "服务已响应，但未找到 /v1/models。基础 URL 可能缺少或重复了 /v1。",
            "cause_endpoint_auth": "端点需要身份验证，或者代理阻止了此路由。",
            "cause_endpoint_invalid_json": "URL 可能指向 Web UI 或不兼容的 API 路由。",
            "cause_endpoint_dify": "Dify 是可选的。如果您在本地运行它，服务可能已停止或映射到其他端口。",
            "cause_endpoint_generic": "服务可能未运行，或者正在监听其他端口。",
            "cause_nvidia": "NVIDIA 驱动程序缺失、异常或 nvidia-smi 不在 PATH 中。",
            "cause_nvidia_noaction": "在纯 CPU 或非 NVIDIA 机器上无需操作。",
            "impact_nvidia": "仅 NVIDIA GPU 推理或 CUDA 运行时需要。",
            "cause_cuda_driver_available": "NVIDIA 驱动程序可用，但未找到 CUDA 工具包。如果只使用像 Ollama 这样的预构建运行时，这可能没问题。",
            "cause_cuda_no_driver": "CUDA 工具包缺失或 nvcc 不在 PATH 中。",
            "config_hint_cuda": "仅在需要编译 CUDA 工作负载或使用需要 nvcc 的运行时才安装 CUDA 工具包。",
            "impact_cuda": "CPU 和许多预构建运行时可选；CUDA 编译需要。",
            "cause_docker": "Docker CLI 缺失、守护进程已停止或当前用户无法访问 Docker 守护进程。",
            "config_hint_docker": "InferDoctor 只检查 Docker 状态，从不启动容器。",
            "impact_docker": "仅当本地 AI 堆栈在容器中运行服务时需要。",
            "cause_generic": "诊断无法确认此组件是否正常。",
            "impact_generic": "如果此组件是本地 AI 堆栈的一部分，请检查它。",
            "config_hint_endpoint": "endpoints.{name}: {endpoint}",
        },
        "checker": {
            "system": {
                "ok": "系统信息已收集",
                "detail_os": "OS：{value}",
                "detail_python": "Python：{value}",
                "detail_arch": "架构：{value}",
                "detail_memory": "可用内存：{value}",
                "detail_memory_unavailable": "可用内存：不可用",
            },
            "nvidia": {
                "not_found": "未找到 nvidia-smi",
                "timed_out": "nvidia-smi 超时",
                "exec_error": "无法执行 nvidia-smi",
                "reported_error": "nvidia-smi 报告了错误",
                "no_gpu_data": "nvidia-smi 已运行但未解析到 GPU 数据",
                "detected": "检测到 {count} 个 NVIDIA GPU",
            },
            "cuda": {
                "not_found": "未找到 CUDA 编译器",
                "timed_out": "nvcc 超时",
                "exec_error": "无法执行 nvcc",
                "reported_error": "nvcc 报告了错误",
                "version_detected": "检测到 CUDA 工具包 {version}",
                "version_not_parsed": "nvcc 可用但无法解析 CUDA 版本",
            },
            "docker": {
                "not_found": "未找到 Docker CLI",
                "timed_out": "Docker 守护进程检查超时",
                "exec_error": "无法执行 Docker CLI",
                "daemon_not_reachable": "Docker CLI 已安装但守护进程不可达",
                "reachable": "Docker CLI 和守护进程可达",
            },
            "ollama": {
                "not_found_not_reachable": "未找到 Ollama 且 API 不可达",
                "installed_not_reachable": "Ollama 已安装但 API 不可达",
                "http_error": "Ollama API 响应了 HTTP {status}",
                "api_reachable_cli_missing": "Ollama API 可达但 CLI 不在 PATH 中",
                "ready": "Ollama CLI 和 API 可用",
            },
            "endpoint": {
                "not_reachable": "{service_label} 端点不可达",
                "reachable": "{service_label} 端点可达",
                "requires_auth": "{service_label} 端点需要身份验证",
                "http_status": "{service_label} 响应了 HTTP {status}",
            },
            "openai": {
                "not_reachable": "{service_label} 端点不可达",
                "requires_auth": "{service_label} 可达但需要身份验证",
                "not_found": "{service_label} models 路由返回了 HTTP 404",
                "http_status": "{service_label} 返回了 HTTP {status}",
                "invalid_json": "{service_label} 返回了无效的 JSON",
                "not_compatible": "{service_label} 响应不是 OpenAI 兼容格式",
                "ready": "{service_label} OpenAI 兼容 API 已就绪",
            },
        },
        "markdown": {
            "title": "InferDoctor 报告",
            "generated_at": "生成时间：`{datetime}`",
            "table_header": "| 检查项 | 状态 | 摘要 |",
            "table_divider": "| --- | --- | --- |",
            "table_row": "| {name} | **{status}** | {summary} |",
            "section_title": "## {name}",
            "details_title": "**详细信息**",
            "suggestions_title": "**建议**",
            "raw_data_title": "**原始数据**",
            "list_item": "- {item}",
        },
        "explain": {
            "title": "InferDoctor 解释：{topic}",
            "what_it_means": "含义：",
            "common_causes": "常见原因：",
            "what_to_try_next": "下一步尝试：",
            "related_command": "相关 InferDoctor 命令：",
        },
        "capacity": {
            "title": "InferDoctor 容量预览",
            "hardware_detected": "检测到的硬件：",
            "workload_readiness": "工作负载就绪性：",
            "no_gpu": "未检测到 GPU。",
            "vram": "VRAM：{vram} GiB",
            "gpu": "GPU：{name}",
            "cpu_only": "仅 CPU 模式",
            "ready": "就绪",
            "possible": "可行",
            "tight": "紧张",
            "unlikely": "不太可能",
        },
        "optimize": {
            "title": "InferDoctor 优化建议",
            "current_situation": "当前情况",
            "possible_bottlenecks": "可能的瓶颈",
            "quick_wins": "建议的快速改进",
            "safe_next_command": "安全的下一个命令",
        },
        "profile": {
            "title": "# InferDoctor 安全诊断配置文件",
            "section_system": "## 系统",
            "section_gpu": "## GPU",
            "section_commands": "## 命令",
            "section_endpoints": "## 端点",
        },
        "recommendation": {
            "title": "InferDoctor 堆栈推荐",
            "goal": "目标：{goal}",
            "preference": "偏好：{preference}",
            "recommendation": "推荐：",
            "runtime_path": "运行时路径：",
        },
        "setup": {
            "title": "InferDoctor 引导设置",
            "plain_english": "简明推荐：",
            "step_by_step": "分步操作：",
            "no_recommendation": "信息不足，无法推荐设置。请提供 --goal 或回答交互提示。",
            "what_to_build": "你想构建什么？[chatbot/document-qa/customer-service/restaurant-ordering/local-api/not-sure]：",
            "what_preference": "你的偏好是什么？[easiest/performance/cpu/gpu]：",
            "existing_runtime": "已有的运行时？[ollama/vllm/sglang/xinference/not-sure]：",
        },
        "stack_plan": {
            "title": "InferDoctor 本地 AI 堆栈计划",
            "required_components": "必需组件：",
            "optional_components": "可选组件：",
            "step_by_step": "分步操作：",
            "no_plan": "信息不足，无法制定堆栈计划。",
        },
        "templates": {
            "title": "InferDoctor 本地 AI 应用模板",
            "what_it_builds": "构建内容",
            "who_its_for": "适用人群",
            "stack_fit": "堆栈适配",
            "no_templates": "没有可用的模板。",
            "created_summary": "已为 {template} 在 {output} 创建了 {count} 个文件",
            "compose_created_summary": "已为 {template} 在 {output} 创建了 {count} 个 Docker Compose 文件",
            "build_what": "构建内容：",
            "for_who": "适用对象：",
            "suitable_stack": "适配堆栈：",
        },
        "template_validation": {
            "title": "InferDoctor 模板验证",
            "checks": "检查项：",
            "passed": "所有检查通过",
            "failed": "{fail} / {total} 检查失败",
            "top_fixes": "主要修复：",
            "next_command": "下一步命令：",
            "smoke_title": "InferDoctor 模板冒烟测试",
            "smoke_results": "冒烟测试结果：",
            "all_passed": "所有冒烟测试通过",
        },
        "model_fit": {
            "title": "InferDoctor 模型适配顾问",
            "request": "请求：{description}",
            "estimate": "估算：{description}",
            "runtime_guidance": "运行时指导：",
            "no_estimate": "无法估算给定参数的适配情况。",
        },
        "scenarios": {
            "title": "InferDoctor 场景就绪性",
            "status": "状态：{status}",
            "reason": "原因：{reason}",
            "next_step": "下一步：{next}",
            "no_scenarios": "没有可用的场景。",
        },
        "perf": {
            "smoke_title": "InferDoctor 性能 UX 冒烟测试",
            "mode": "模式：{mode}",
            "endpoint": "端点：{endpoint}",
            "model": "模型：{model}",
            "warmup": "预热：{count}",
            "runs": "运行次数：{count}",
            "warning": "警告：{warning}",
            "error": "错误：{error}",
            "result": "结果：",
            "ttft": "TTFT：{value}秒",
            "tps": "TPS：{value}",
            "latency": "延迟：{value}秒",
            "no_results": "没有可显示的结果。",
        },
        "runner": {
            "checker_failed": "检查器意外失败",
        },
    },
    "ja": {
        "dashboard": {
            "title": "InferDoctor - ローカルAIスタックヘルスチェック",
            "health": "全体の健全性：{score} / 100  ({label})",
            "scores": "PASS 100 | WARN 60 | FAIL 0 | SKIP 85  （ヒューリスティック）",
            "header": "コンポーネント   ステータス   概要",
            "divider": "----------- -------- --------------------------------------------------",
            "top_fixes": "推奨される対応（優先順）：",
            "no_fixes": "すぐに必要な修正はありません。検出されたスタックは健康に見えます。",
            "fix": "{index}. {component}: {issue}",
            "likely_cause": "   考えられる原因: {cause}",
            "impact": "   影響: {impact}",
            "try": "   試す: {command}",
            "config": "   {hint_label}: {config_hint}",
            "hint_config": "設定",
            "hint_note": "注記",
            "detail": "詳細な診断情報：",
            "stack_summary": "スタック概要：{pass_count} 件成功 | {warn_count} 件注意 | {skip_count} 件オプション/オフライン | {fail_count} 件失敗",
            "doctor_read_fail": "診断結果：必要なチェックが少なくとも1つ失敗しました。まず下の最初の項目から確認してください。",
            "doctor_read_warn": "診断結果：いくつかのコンポーネントに注意が必要です。まず下の最初の項目から確認してください。",
            "doctor_read_skip": "診断結果：重大な失敗は検出されませんでした。スキップされたコンポーネントは、使わない場合は通常オプションです。",
            "doctor_read_pass": "診断結果：検出されたチェックはすべて成功しました。ローカルAIスタックは準備できているようです。",
        },
        "console": {
            "status": "[{status:<4}] {name:<10} {summary}",
            "detail": "       - {detail}",
            "suggestion": "       提案：{suggestion}",
            "raw_data": "       生データ：",
        },
        "health": {
            "label_healthy": "健全",
            "label_mostly_healthy": "おおむね健全",
            "label_needs_attention": "注意が必要",
            "label_unhealthy": "不健全",
            "impact_fail": "修正されるまでこのコンポーネントはブロックされる可能性があります。",
            "impact_warn": "アプリが {service_name} に依存している場合は注意が必要です。",
            "impact_skip": "{service_name} がローカルスタックの一部でない限りオプションです。",
            "cause_endpoint_404": "サービスは応答しましたが、/v1/models が見つかりませんでした。ベースURLが欠落しているか、/v1 が重複している可能性があります。",
            "cause_endpoint_auth": "エンドポイントは認証が必要か、プロキシがこのルートをブロックしています。",
            "cause_endpoint_invalid_json": "URLがWeb UIまたは互換性のないAPIルートを指している可能性があります。",
            "cause_endpoint_dify": "Difyはオプションです。ローカルで実行している場合、サービスが停止しているか別のポートにマッピングされている可能性があります。",
            "cause_endpoint_generic": "サービスが実行されていないか、別のポートで待機している可能性があります。",
            "cause_nvidia": "NVIDIAドライバーがないか異常があるか、nvidia-smiがPATHにありません。",
            "cause_nvidia_noaction": "CPU専用または非NVIDIAマシンでは操作は必要ありません。",
            "impact_nvidia": "NVIDIA GPU推論またはCUDA対応ランタイムにのみ必要です。",
            "cause_cuda_driver_available": "NVIDIAドライバーは利用可能ですが、CUDAツールキットが見つかりませんでした。Ollamaなどのプリビルトランタイムのみを使用する場合は問題ない可能性があります。",
            "cause_cuda_no_driver": "CUDAツールキットがないか、nvccがPATHにありません。",
            "config_hint_cuda": "CUDAワークロードをコンパイルするか、nvccを必要とするランタイムを使用する場合のみCUDAツールキットをインストールしてください。",
            "impact_cuda": "CPUおよび多くのプリビルトランタイムではオプション。CUDAコンパイルには必要です。",
            "cause_docker": "Docker CLIがないか、デーモンが停止しているか、現在のユーザーがDockerデーモンにアクセスできません。",
            "config_hint_docker": "InferDoctorはDockerのステータスのみをチェックし、コンテナを起動することはありません。",
            "impact_docker": "ローカルAIスタックがコンテナ内でサービスを実行している場合にのみ必要です。",
            "cause_generic": "診断でこのコンポーネントが正常であることを確認できませんでした。",
            "impact_generic": "このコンポーネントがローカルAIスタックの一部である場合は確認してください。",
            "config_hint_endpoint": "endpoints.{name}: {endpoint}",
        },
        "checker": {
            "system": {
                "ok": "システム情報が収集されました",
                "detail_os": "OS：{value}",
                "detail_python": "Python：{value}",
                "detail_arch": "アーキテクチャ：{value}",
                "detail_memory": "利用可能メモリ：{value}",
                "detail_memory_unavailable": "利用可能メモリ：なし",
            },
            "nvidia": {
                "not_found": "nvidia-smi が見つかりませんでした",
                "timed_out": "nvidia-smi がタイムアウトしました",
                "exec_error": "nvidia-smi を実行できませんでした",
                "reported_error": "nvidia-smi がエラーを報告しました",
                "no_gpu_data": "nvidia-smi は実行されましたがGPUデータが解析できませんでした",
                "detected": "{count} 個の NVIDIA GPU を検出",
            },
            "cuda": {
                "not_found": "CUDA コンパイラが見つかりませんでした",
                "timed_out": "nvcc がタイムアウトしました",
                "exec_error": "nvcc を実行できませんでした",
                "reported_error": "nvcc がエラーを報告しました",
                "version_detected": "CUDA ツールキット {version} を検出",
                "version_not_parsed": "nvcc は利用可能ですが CUDA バージョンが解析できませんでした",
            },
            "docker": {
                "not_found": "Docker CLI が見つかりませんでした",
                "timed_out": "Docker デーモンチェックがタイムアウトしました",
                "exec_error": "Docker CLI を実行できませんでした",
                "daemon_not_reachable": "Docker CLI はインストールされていますがデーモンに到達できません",
                "reachable": "Docker CLI とデーモンが利用可能です",
            },
            "ollama": {
                "not_found_not_reachable": "Ollama が見つからず、API にも到達できません",
                "installed_not_reachable": "Ollama はインストールされていますが API に到達できません",
                "http_error": "Ollama API が HTTP {status} を返しました",
                "api_reachable_cli_missing": "Ollama API には到達できますが CLI が PATH にありません",
                "ready": "Ollama CLI と API が利用可能です",
            },
            "endpoint": {
                "not_reachable": "{service_label} エンドポイントに到達できません",
                "reachable": "{service_label} エンドポイントは到達可能です",
                "requires_auth": "{service_label} エンドポイントは認証が必要です",
                "http_status": "{service_label} が HTTP {status} を返しました",
            },
            "openai": {
                "not_reachable": "{service_label} エンドポイントに到達できません",
                "requires_auth": "{service_label} は到達可能ですが認証が必要です",
                "not_found": "{service_label} models ルートが HTTP 404 を返しました",
                "http_status": "{service_label} が HTTP {status} を返しました",
                "invalid_json": "{service_label} が無効な JSON を返しました",
                "not_compatible": "{service_label} の応答は OpenAI 互換ではありません",
                "ready": "{service_label} OpenAI 互換 API の準備ができています",
            },
        },
        "markdown": {
            "title": "InferDoctor レポート",
            "generated_at": "生成日時：`{datetime}`",
            "table_header": "| チェック | ステータス | 概要 |",
            "table_divider": "| --- | --- | --- |",
            "table_row": "| {name} | **{status}** | {summary} |",
            "section_title": "## {name}",
            "details_title": "**詳細**",
            "suggestions_title": "**提案**",
            "raw_data_title": "**生データ**",
            "list_item": "- {item}",
        },
        "explain": {
            "title": "InferDoctor 説明：{topic}",
            "what_it_means": "意味：",
            "common_causes": "一般的な原因：",
            "what_to_try_next": "次に試すこと：",
            "related_command": "関連 InferDoctor コマンド：",
        },
        "capacity": {
            "title": "InferDoctor 容量プレビュー",
            "hardware_detected": "検出されたハードウェア：",
            "workload_readiness": "ワークロードの準備状況：",
            "no_gpu": "GPUが検出されませんでした。",
            "vram": "VRAM：{vram} GiB",
            "gpu": "GPU：{name}",
            "cpu_only": "CPUのみモード",
            "ready": "準備完了",
            "possible": "可能",
            "tight": "厳しい",
            "unlikely": "難しい",
        },
        "optimize": {
            "title": "InferDoctor 最適化アドバイス",
            "current_situation": "現在の状況",
            "possible_bottlenecks": "可能性のあるボトルネック",
            "quick_wins": "推奨される迅速な改善",
            "safe_next_command": "安全な次のコマンド",
        },
        "profile": {
            "title": "# InferDoctor 安全診断プロファイル",
            "section_system": "## システム",
            "section_gpu": "## GPU",
            "section_commands": "## コマンド",
            "section_endpoints": "## エンドポイント",
        },
        "recommendation": {
            "title": "InferDoctor スタック推奨",
            "goal": "目標：{goal}",
            "preference": "優先：{preference}",
            "recommendation": "推奨：{value}",
            "runtime_path": "ランタイムパス：{path}",
        },
        "setup": {
            "title": "InferDoctor ガイド付きセットアップ",
            "plain_english": "わかりやすい推奨：",
            "step_by_step": "ステップバイステップの次のアクション：",
            "no_recommendation": "セットアップを推奨するのに十分な情報がありません。--goal を指定するか、インタラクティブなプロンプトに回答してください。",
            "what_to_build": "何を構築しますか？[chatbot/document-qa/customer-service/restaurant-ordering/local-api/not-sure]：",
            "what_preference": "どのような優先度ですか？[easiest/performance/cpu/gpu]：",
            "existing_runtime": "既存のランタイム？[ollama/vllm/sglang/xinference/not-sure]：",
        },
        "stack_plan": {
            "title": "InferDoctor ローカルAIスタック計画",
            "required_components": "必要なコンポーネント：",
            "optional_components": "オプションのコンポーネント：",
            "step_by_step": "ステップバイステップの次のアクション：",
            "no_plan": "スタック計画を立てるのに十分な情報がありません。",
        },
        "templates": {
            "title": "InferDoctor ローカルAIアプリテンプレート",
            "what_it_builds": "構築するもの",
            "who_its_for": "対象者",
            "stack_fit": "スタック適合",
            "no_templates": "利用可能なテンプレートはありません。",
            "created_summary": "{template} の {count} ファイルを {output} に作成しました",
            "compose_created_summary": "{template} の {count} Docker Compose ファイルを {output} に作成しました",
            "build_what": "構築内容：",
            "for_who": "対象者：",
            "suitable_stack": "適合スタック：",
        },
        "template_validation": {
            "title": "InferDoctor テンプレート検証",
            "checks": "チェック：",
            "passed": "すべてのチェックが合格しました",
            "failed": "{fail} / {total} チェックが失敗しました",
            "top_fixes": "主な修正：",
            "next_command": "次に試すコマンド：",
            "smoke_title": "InferDoctor テンプレートスモークテスト",
            "smoke_results": "スモークテスト結果：",
            "all_passed": "すべてのスモークテストに合格しました",
        },
        "model_fit": {
            "title": "InferDoctor モデル適合アドバイザー",
            "request": "リクエスト：{description}",
            "estimate": "見積もり：{description}",
            "runtime_guidance": "ランタイムガイダンス：",
            "no_estimate": "指定されたパラメータでは適合を見積もれません。",
        },
        "scenarios": {
            "title": "InferDoctor シナリオ準備状況",
            "status": "ステータス：{status}",
            "reason": "理由：{reason}",
            "next_step": "次のステップ：{next}",
            "no_scenarios": "利用可能なシナリオはありません。",
        },
        "perf": {
            "smoke_title": "InferDoctor パフォーマンスUXスモークテスト",
            "mode": "モード：{mode}",
            "endpoint": "エンドポイント：{endpoint}",
            "model": "モデル：{model}",
            "warmup": "ウォームアップ：{count}",
            "runs": "実行回数：{count}",
            "warning": "警告：{warning}",
            "error": "エラー：{error}",
            "result": "結果：",
            "ttft": "TTFT：{value}秒",
            "tps": "TPS：{value}",
            "latency": "レイテンシ：{value}秒",
            "no_results": "表示する結果がありません。",
        },
        "runner": {
            "checker_failed": "チェッカーが予期せず失敗しました",
        },
    },
}


def _nested_get(d: Dict[str, Any], key: str) -> Optional[str]:
    """Resolve a dot-path key against a nested dictionary.

    Returns None if any segment of the path is missing or the terminal value is not a string.
    """
    parts = key.split(".")
    current: Any = d
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current if isinstance(current, str) else None


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
    """Look up a dot-path translation key in the nested TRANSLATIONS dict.

    Falls back to English if the key is not found in the target language.
    If still not found, returns the key itself as a last resort.
    """
    if language == "auto":
        language = detect_system_language()
    try:
        normalized_language = normalize_language(language)
    except ValueError:
        normalized_language = "en"

    lang_dict = TRANSLATIONS.get(normalized_language, TRANSLATIONS["en"])
    template = _nested_get(lang_dict, key)

    # Fallback to English
    if template is None:
        template = _nested_get(TRANSLATIONS["en"], key)

    if template is None:
        return key

    return template.format(**kwargs)
