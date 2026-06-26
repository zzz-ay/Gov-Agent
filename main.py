import streamlit as st
import html
import json
import os
import time
import uuid
from dotenv import load_dotenv

from app.agents.business_flow import classify_business_flow
from app.tools.vectorstore_admin import (
    get_policy_library_status,
    rebuild_vectorstore,
    save_uploaded_policy_file,
)
from app.storage.jsonl_store import save_chat_log
from app.knowledge_sources import get_enabled_knowledge_sources


DEFAULT_SLOTS = {
    "region": "未明确",
    "hukou_region": "未明确",
    "current_city": "未明确",
    "graduation_year": "未明确",
    "education_level": "未明确",
    "is_fresh_graduate": None,
    "employment_status": "unknown",
    "employer_has_archive_authority": None,
    "archive_need": "unknown",
    "hukou_need": "unknown",
    "subsidy_interest": None,
    "social_security_interest": None,
    "entrepreneurship_interest": None,
}


def build_context_from_slots(slots: dict) -> str:
    meaningful = []
    for key, value in (slots or {}).items():
        if value in [None, "", "unknown", "未明确"]:
            continue
        meaningful.append(f"{key}: {value}")

    if not meaningful:
        return ""

    return "用户已补充信息：\n" + "\n".join(f"- {item}" for item in meaningful)


# =========================
# 0. 环境变量
# =========================

load_dotenv()

ENABLE_ADMIN_PANEL = os.getenv("ENABLE_ADMIN_PANEL", "false").lower() == "true"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", "18"))
STREAM_DELAY = float(os.getenv("STREAM_DELAY", "0.003"))
MAX_RENDERED_MESSAGES = int(os.getenv("MAX_RENDERED_MESSAGES", "20"))
ENABLE_SLOT_AGENT = os.getenv("ENABLE_SLOT_AGENT", "true").lower() == "true"


SOURCE_LABELS = {
    "gov_policy": "政策文件",
    "laws": "法律法规",
    "legal_cases": "类案参考",
    "public_notices": "公告通知",
}

SOURCE_DESCRIPTIONS = {
    "gov_policy": "办事条件、流程、材料和政策口径",
    "laws": "法律条文、法规依据和合规边界",
    "legal_cases": "相似事实、裁判观点和风险参照",
    "public_notices": "名单公示、结果公告和通知通告",
}

SOURCE_BADGE_CLASS = {
    "gov_policy": "badge-policy",
    "laws": "badge-law",
    "legal_cases": "badge-case",
    "public_notices": "badge-notice",
}


# =========================
# 1. 页面级配置
# =========================

st.set_page_config(
    page_title="政策法规 RAG Agent 平台",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 2. 页面样式
# =========================

st.markdown(
    """
<style>
    .stApp {
        background-color: #F7F8FA;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        box-shadow: 2px 0 8px rgba(0,0,0,0.04);
    }

    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"] {
        background-color: #165DFF !important;
        color: white !important;
        border: none;
        box-shadow: 0 2px 4px rgba(22, 93, 255, 0.2);
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #4080FF !important;
    }

    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E6EB;
        color: #4E5969;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }

    div.stButton > button[kind="secondary"]:hover {
        border-color: #165DFF;
        color: #165DFF;
        background-color: #F0F7FF;
    }

    [data-testid="stChatMessageAvatar"] {
        background-color: transparent !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #FFF7F7;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #FCE5E5;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #F0F7FF;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #C9E1FF;
        border-left: 4px solid #165DFF;
    }

    .gov-header {
        background-color: #FFFFFF;
        border: 1px solid #E5E6EB;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }

    .gov-title {
        font-size: 22px;
        font-weight: 700;
        color: #1D2129;
        margin-bottom: 6px;
    }

    .gov-subtitle {
        font-size: 14px;
        color: #4E5969;
        line-height: 1.6;
    }

    .capability-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E6EB;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 13px;
        color: #4E5969;
    }

    .slot-box {
        background-color: #F7F8FA;
        border: 1px solid #E5E6EB;
        border-radius: 8px;
        padding: 10px;
        font-size: 13px;
        color: #4E5969;
        line-height: 1.7;
    }

    .source-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E6EB;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }

    .source-title {
        font-weight: 600;
        color: #1D2129;
        margin-bottom: 4px;
    }

    .source-meta {
        font-size: 12px;
        color: #86909C;
        margin-bottom: 8px;
    }

    .source-preview {
        font-size: 13px;
        color: #4E5969;
        line-height: 1.6;
    }

    .gov-tips {
        font-size: 12px;
        color: #86909C;
        margin-top: 12px;
        padding-top: 8px;
        border-top: 1px dashed #D8DCE3;
    }

    .small-note {
        font-size: 12px;
        color: #86909C;
        line-height: 1.6;
    }

    .debug-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E6EB;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        font-size: 13px;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.82);
        backdrop-filter: blur(10px);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #E6EAF0;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: #3F4A5A;
    }

    .enterprise-shell {
        background: #FFFFFF;
        border: 1px solid #E1E7EF;
        border-radius: 8px;
        padding: 20px 22px 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }

    .enterprise-kicker {
        color: #516070;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .enterprise-title {
        color: #172033;
        font-size: 26px;
        line-height: 1.32;
        font-weight: 760;
        margin-bottom: 8px;
    }

    .enterprise-subtitle {
        color: #4B5565;
        font-size: 14px;
        line-height: 1.75;
        max-width: 860px;
    }

    .source-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }

    .source-badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 650;
        border: 1px solid #D9E0EA;
        background: #FFFFFF;
        color: #374151;
    }

    .badge-policy { border-color: #B9D7CC; background: #F2FAF7; color: #17624A; }
    .badge-law { border-color: #C8D7F2; background: #F4F7FE; color: #244C8E; }
    .badge-case { border-color: #E8D4AA; background: #FFF9EC; color: #7A5514; }
    .badge-notice { border-color: #D7C7EB; background: #F8F4FD; color: #5A3E7A; }

    .sidebar-brand {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 14px 12px 14px;
        background: #F8FAFC;
        margin-bottom: 14px;
    }

    .sidebar-title {
        color: #111827;
        font-size: 16px;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .sidebar-caption {
        color: #5B6573;
        font-size: 12px;
        line-height: 1.65;
    }

    .question-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 12px;
    }

    .question-card {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 12px 13px;
        min-height: 78px;
    }

    .question-card-title {
        color: #172033;
        font-size: 13px;
        font-weight: 720;
        margin-bottom: 6px;
    }

    .question-card-copy {
        color: #667085;
        font-size: 12px;
        line-height: 1.55;
    }

    .source-card {
        box-shadow: none;
    }

    [data-testid="stChatMessage"] {
        border-radius: 8px;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 3px solid #64748B;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #FFFFFF;
        border: 1px solid #DDE5F2;
        border-left: 3px solid #2F6BFF;
    }

    @media (max-width: 900px) {
        .question-grid {
            grid-template-columns: 1fr;
        }

        .enterprise-title {
            font-size: 22px;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# 3. 初始化状态
# =========================

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "您好，我是政策法规 RAG Agent 平台。"
        "您可以咨询政策办理、法规依据、企业合规风险和类案参考等问题。"
        "系统会结合已接入的政策文件、法律法规和案例数据，生成可追溯的参考回答。"
    ),
    "sources": [],
}

def new_welcome_message() -> dict:
    return {
        "role": WELCOME_MESSAGE["role"],
        "content": WELCOME_MESSAGE["content"],
        "sources": [],
    }


def create_conversation_record(title: str = "新对话") -> dict:
    return {
        "id": f"chat-{uuid.uuid4().hex[:8]}",
        "title": title,
        "messages": [new_welcome_message()],
        "user_slots": dict(DEFAULT_SLOTS),
        "slot_summary": "",
        "last_slot_context": "",
        "pending_question": "",
    }


def get_active_conversation() -> dict:
    conversations = st.session_state.get("conversations", [])
    active_id = st.session_state.get("active_conversation_id", "")
    for conversation in conversations:
        if conversation.get("id") == active_id:
            return conversation
    return conversations[0] if conversations else create_conversation_record()


def build_conversation_title(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = str(message.get("content", "")).strip()
            return content[:18] + ("..." if len(content) > 18 else "")
    return "新对话"


def sync_active_conversation():
    conversation = get_active_conversation()
    conversation["messages"] = st.session_state.get("messages", [new_welcome_message()])
    conversation["user_slots"] = st.session_state.get("user_slots", dict(DEFAULT_SLOTS))
    conversation["slot_summary"] = st.session_state.get("slot_summary", "")
    conversation["last_slot_context"] = st.session_state.get("last_slot_context", "")
    conversation["pending_question"] = st.session_state.get("pending_question", "")
    conversation["title"] = build_conversation_title(conversation["messages"])


def load_conversation(conversation_id: str):
    st.session_state.active_conversation_id = conversation_id
    conversation = get_active_conversation()
    st.session_state.messages = [dict(message) for message in conversation.get("messages", [new_welcome_message()])]
    st.session_state.user_slots = dict(conversation.get("user_slots", DEFAULT_SLOTS))
    st.session_state.slot_summary = conversation.get("slot_summary", "")
    st.session_state.last_slot_context = conversation.get("last_slot_context", "")
    st.session_state.pending_question = conversation.get("pending_question", "")
    st.session_state.quick_question = None


if "conversations" not in st.session_state:
    first_conversation = create_conversation_record()
    st.session_state.conversations = [first_conversation]
    st.session_state.active_conversation_id = first_conversation["id"]

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = st.session_state.conversations[0]["id"]

load_conversation(st.session_state.active_conversation_id)

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "fast_mode" not in st.session_state:
    st.session_state.fast_mode = True

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "Fast Chat"

if "admin_unlocked" not in st.session_state:
    st.session_state.admin_unlocked = False

if "admin_password_input" not in st.session_state:
    st.session_state.admin_password_input = ""


# =========================
# 4. 工具函数
# =========================

def create_new_conversation():
    sync_active_conversation()
    conversation = create_conversation_record()
    st.session_state.conversations.append(conversation)
    load_conversation(conversation["id"])


def switch_conversation(conversation_id: str):
    if conversation_id == st.session_state.active_conversation_id:
        return
    sync_active_conversation()
    load_conversation(conversation_id)


def render_sources(sources):
    if not sources:
        return

    with st.expander("查看参考依据", expanded=False):
        for idx, source in enumerate(sources, start=1):
            title = html.escape(source.get("title", "未知标题"))
            source_org = html.escape(source.get("source_org", "未知发布机构"))
            region = html.escape(source.get("region", "未知地区"))
            publish_date = source.get("publish_date", "")
            topic = html.escape(source.get("topic", ""))
            knowledge_source_id = source.get("knowledge_source_id", "")
            knowledge_source_type = source.get("knowledge_source_type", "")
            source_label = SOURCE_LABELS.get(knowledge_source_id, knowledge_source_type or "参考资料")
            source_url = source.get("source_url", "")
            content = html.escape(source.get("content", ""))

            meta_parts = [
                f"发布机构：{source_org}",
                f"地区：{region}",
            ]

            if publish_date:
                meta_parts.append(f"发布时间：{publish_date}")

            if topic:
                meta_parts.append(f"主题：{topic}")

            preview = content[:220] + "..." if content and len(content) > 220 else content
            badge_class = SOURCE_BADGE_CLASS.get(knowledge_source_id, "")

            st.markdown(
                f"""
<div class="source-card">
    <div class="source-title">资料 {idx}：{title}</div>
    <div class="source-meta">{" ｜ ".join(meta_parts)}</div>
    <div class="source-strip"><span class="source-badge {badge_class}">{source_label}</span></div>
    <div class="source-preview">{preview or "暂无内容预览"}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            if source_url:
                st.caption(f"来源链接：{source_url}")


def render_source_badges(enabled_sources):
    badges = []
    for source in enabled_sources:
        label = SOURCE_LABELS.get(source.id, source.name)
        badge_class = SOURCE_BADGE_CLASS.get(source.id, "")
        badges.append(f'<span class="source-badge {badge_class}">{label}</span>')
    return "\n".join(badges)


def render_workspace_header(mode_text: str):
    st.markdown(
        """
<div class="enterprise-shell">
    <div class="enterprise-kicker">Policy, Law and Public Service Intelligence</div>
    <div class="enterprise-title">政策法规与公共服务知识库</div>
    <div class="enterprise-subtitle">
        面向业务用户的政策法规问答工作台，支持政策办理、法规依据、合规风险和类案参考。
        系统会根据问题自动选择合适的知识源，并返回带参考依据的回答。
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_question_showcase():
    cards = [
        ("政策办理", "就业补贴怎么申请？需要满足哪些条件？"),
        ("法规依据", "劳动合同试用期怎么规定？"),
        ("合规风险", "虚假申请就业补贴有什么风险？"),
        ("类案参考", "类似诈骗罪法院一般怎么判？"),
    ]
    items = "\n".join(
        f"""
<div class="question-card">
    <div class="question-card-title">{title}</div>
    <div class="question-card-copy">{copy}</div>
</div>
"""
        for title, copy in cards
    )
    st.markdown(f'<div class="question-grid">{items}</div>', unsafe_allow_html=True)


def get_current_business_flow(question: str) -> dict:
    enabled_source_ids = [source.id for source in get_enabled_knowledge_sources()]
    return classify_business_flow(
        question=question,
        enabled_source_ids=enabled_source_ids,
    )


def render_business_flow_card(business_flow: dict):
    if not st.session_state.admin_unlocked or not st.session_state.debug_mode:
        return

    if not business_flow:
        return

    source_ids = ", ".join(business_flow.get("source_ids", [])) or "未指定"
    output_focus = " / ".join(business_flow.get("output_focus", [])) or "综合问答"
    steps = business_flow.get("steps", [])

    with st.expander("业务流程与知识源路由", expanded=False):
        st.markdown(f"**流程：** {business_flow.get('flow_name', '')}")
        st.caption(business_flow.get("description", ""))
        st.markdown(f"**本轮知识源：** `{source_ids}`")
        st.markdown(f"**输出重点：** {output_focus}")

        if steps:
            st.markdown("**执行步骤**")
            for idx, step in enumerate(steps, start=1):
                st.write(f"{idx}. {step}")


def update_slots_with_lightweight_rules(user_input: str):
    text = user_input or ""
    slots = dict(st.session_state.user_slots)
    updates = []

    if "石家庄" in text:
        slots["region"] = "河北省石家庄市"
        slots["current_city"] = "河北省石家庄市"
        updates.append("地区：河北省石家庄市")
    elif "保定" in text:
        slots["region"] = "河北省保定市"
        slots["current_city"] = "河北省保定市"
        updates.append("地区：河北省保定市")
    elif "河北" in text:
        slots["region"] = "河北省"
        updates.append("地区：河北省")

    if "灵活就业" in text:
        slots["employment_status"] = "flexible_employment"
        updates.append("就业状态：灵活就业")

    if "就业补贴" in text or "补贴" in text or "补助" in text:
        slots["subsidy_interest"] = True
        updates.append("关注补贴")

    if "社保" in text or "医保" in text:
        slots["social_security_interest"] = True
        updates.append("关注社保医保")

    if updates:
        st.session_state.user_slots = slots
        st.session_state.slot_summary = "；".join(updates)


def is_short_clarification_reply(user_input: str) -> bool:
    text = (user_input or "").strip()
    if not text:
        return False
    if len(text) <= 12:
        return True
    keywords = ["石家庄", "保定", "河北", "就业补贴", "补贴", "社保", "灵活就业"]
    return any(keyword in text for keyword in keywords) and len(text) <= 24


def build_enhanced_question(user_input: str) -> str:
    slot_context = build_context_from_slots(st.session_state.user_slots)
    st.session_state.last_slot_context = slot_context
    pending_question = st.session_state.get("pending_question", "")

    if pending_question and is_short_clarification_reply(user_input):
        parts = [f"用户原始问题：{pending_question}"]
        if slot_context:
            parts.append(slot_context)
        parts.append(f"用户本轮补充信息：{user_input}")
        return "\n\n".join(parts)

    if slot_context:
        return f"{slot_context}\n\n用户当前问题：{user_input}"

    return user_input


def stream_markdown(
    text: str,
    delay: float = STREAM_DELAY,
    chunk_size: int = STREAM_CHUNK_SIZE,
):
    placeholder = st.empty()
    rendered = ""

    for idx in range(0, len(text), chunk_size):
        rendered += text[idx : idx + chunk_size]
        placeholder.markdown(rendered, unsafe_allow_html=True)
        if delay > 0:
            time.sleep(delay)

    return rendered


def run_fast_streaming_answer(user_input: str):
    from app.rag.streaming_answer import build_streaming_context, stream_answer_from_context

    context = build_streaming_context(user_input)
    placeholder = st.empty()
    answer = ""

    for chunk in stream_answer_from_context(context):
        answer += chunk
        placeholder.markdown(answer, unsafe_allow_html=True)

    policy_sources = context.get("policy_result", {}).get("policy_sources", [])

    return {
        "final_answer": answer,
        "sources": policy_sources,
        "source_router_result": context.get("source_router_result", {}),
        "evidence_result": context.get("policy_result", {}).get("evidence_result", {}),
        "risk_level": "not_checked",
        "verification_score": 0,
        "streaming_mode": "fast_policy_answer",
        "direct_answer": bool(context.get("direct_answer", "")),
    }


def save_streamlit_chat_log(
    question: str,
    answer: str,
    result: dict,
    latency_ms: float,
    chat_mode: str,
    debug_mode: bool,
    business_flow: dict = None,
):
    from app.rag.citation import citations_from_sources

    sources = result.get("sources", [])
    citations = citations_from_sources(sources)

    return save_chat_log(
        {
            "session_id": "streamlit",
            "question": question,
            "answer": answer,
            "source_count": len(sources),
            "sources": [
                {
                    "title": source.get("title", ""),
                    "knowledge_source_id": source.get("knowledge_source_id", ""),
                    "knowledge_source_name": source.get("knowledge_source_name", ""),
                    "knowledge_source_type": source.get("knowledge_source_type", ""),
                    "source_file": source.get("source_file", ""),
                    "source_url": source.get("source_url", ""),
                    "region": source.get("region", ""),
                    "topic": source.get("topic", ""),
                }
                for source in sources
            ],
            "citations": citations,
            "risk_level": result.get("risk_level"),
            "verification_score": result.get("verification_score"),
            "debug_mode": debug_mode,
            "latency_ms": latency_ms,
            "metadata": {
                "entrypoint": "streamlit",
                "chat_mode": chat_mode,
                "streaming_mode": result.get("streaming_mode", ""),
                "business_flow": business_flow or {},
            },
            "source_router_result": result.get("source_router_result", {}),
            "evidence_result": result.get("evidence_result", {}),
            "source_distribution": result.get("evidence_result", {}).get("source_distribution", {}),
            "selected_source_ids": result.get("source_router_result", {}).get("source_ids", []),
        }
    )


def render_source_router_card(source_router_result: dict):
    if not st.session_state.admin_unlocked or not st.session_state.debug_mode:
        return

    if not source_router_result:
        return

    source_ids = ", ".join(source_router_result.get("source_ids", [])) or "none"
    requested = ", ".join(source_router_result.get("requested_source_ids", [])) or "none"
    strategy = source_router_result.get("strategy", "")
    reasons = source_router_result.get("routing_reasons", [])

    with st.expander("知识源路由结果", expanded=False):
        st.markdown(f"**实际检索知识源：** `{source_ids}`")
        st.markdown(f"**候选/请求知识源：** `{requested}`")
        if strategy:
            st.markdown(f"**路由策略：** `{strategy}`")
        if reasons:
            st.markdown("**路由原因：**")
            for reason in reasons:
                st.write(f"- {reason}")


def render_agent_debug(result: dict):
    """
    管理员模式下展示多智能体中间结果。
    """
    if not st.session_state.admin_unlocked:
        return

    if not st.session_state.debug_mode:
        return

    with st.expander("查看多智能体执行过程", expanded=False):
        st.caption("当前为管理员调试模式，会展示各 Agent 的中间输出。")

        debug_items = [
            ("Intent Agent", result.get("intent_result", {})),
            ("Source Router Agent", result.get("source_router_result", {})),
            ("Service Recommendation Agent", result.get("service_recommendation_result", {})),
            ("Policy Agent", result.get("policy_result", {})),
            ("Policy Validity Agent", result.get("policy_validity_result", {})),
            ("Agency Agent", result.get("agency_result", {})),
            ("Eligibility Agent", result.get("eligibility_result", {})),
            ("Procedure Agent", result.get("procedure_result", {})),
            ("Material Agent", result.get("material_result", {})),
            ("Answer Agent", result.get("answer_result", {})),
            ("Verification Agent", result.get("verify_result", {})),
        ]

        for title, data in debug_items:
            with st.expander(title, expanded=False):
                try:
                    st.json(data)
                except Exception:
                    st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")


def render_policy_library_admin():
    """
    管理员模式下的政策库管理面板。
    """
    if not st.button("Load policy library admin", use_container_width=True, type="secondary"):
        st.caption("Click to load policy files, upload controls, and vectorstore rebuild tools.")
        return

    status = get_policy_library_status()

    file_status = status.get("file_status", {})
    vector_status = status.get("vector_status", {})

    raw_docs_exists = file_status.get("raw_docs_exists", False)
    vectorstore_exists = vector_status.get("vectorstore_exists", False)

    if not raw_docs_exists:
        st.warning("未找到 data/raw_docs 政策资料目录。")
    else:
        st.markdown(
            f"""
<div class="capability-card">
📁 原始资料目录：<br>
{status.get("raw_docs_dir", "")}<br><br>
📤 管理员上传目录：<br>
{status.get("admin_uploads_dir", "")}<br><br>
📄 政策资料文件：{file_status.get("doc_files", 0)} 个<br>
- TXT 文件：{file_status.get("txt_files", 0)} 个<br>
- PDF 文件：{file_status.get("pdf_files", 0)} 个<br>
- Meta 文件：{file_status.get("meta_files", 0)} 个<br>
- 管理员上传文件：{file_status.get("uploaded_files", 0)} 个
</div>
""",
            unsafe_allow_html=True,
        )

    if vectorstore_exists:
        st.success("向量库状态：已构建")
    else:
        st.warning("向量库状态：未检测到")

    st.caption(f"向量库路径：{vector_status.get('vectorstore_path', '')}")
    st.caption(f"向量库文件数量：{vector_status.get('file_count', 0)}")

    with st.expander("查看政策文件列表", expanded=False):
        files = file_status.get("files", [])

        if files:
            for file in files:
                st.text(file)
        else:
            st.caption("当前未检测到政策文件。")

    st.divider()

    st.markdown("#### 上传政策文件")
    st.caption("支持上传 txt、pdf、md、docx 和 meta.txt 文件。上传后请重新构建向量库。")

    uploaded_files = st.file_uploader(
        "选择政策文件",
        type=["txt", "pdf", "md", "docx"],
        accept_multiple_files=True,
        help="如果是 PDF 文件，建议同时上传同名 .meta.txt 元数据文件。",
    )

    meta_file = st.file_uploader(
        "可选：上传 meta.txt 元数据文件",
        type=["txt"],
        accept_multiple_files=True,
        help="用于补充 PDF 或政策文件的标题、发布机构、发布时间、来源链接等元数据。文件名建议为 xxx.meta.txt。",
        key="meta_file_uploader",
    )

    files_to_save = []

    if uploaded_files:
        files_to_save.extend(uploaded_files)

    if meta_file:
        files_to_save.extend(meta_file)

    if files_to_save:
        if st.button("保存上传文件", use_container_width=True, type="secondary"):
            success_count = 0
            fail_count = 0

            for uploaded_file in files_to_save:
                result = save_uploaded_policy_file(uploaded_file)

                if result.get("success"):
                    success_count += 1
                    st.success(f"保存成功：{result.get('filename')}")
                    st.caption(result.get("saved_path", ""))
                else:
                    fail_count += 1
                    st.error(f"保存失败：{result.get('filename')}")
                    st.caption(result.get("message", ""))

            st.info(f"上传完成：成功 {success_count} 个，失败 {fail_count} 个。请重新构建向量库后生效。")

    st.divider()

    st.markdown("#### 向量库重建")

    if st.button("重新构建向量库", use_container_width=True, type="secondary"):
        with st.spinner("正在重建向量库，请不要关闭页面..."):
            result = rebuild_vectorstore()

        if result.get("success"):
            st.success("向量库重建完成")
        else:
            st.error("向量库重建失败")

        st.caption(f"执行命令：{result.get('command', '')}")
        st.caption(f"返回码：{result.get('returncode', '')}")

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if stdout:
            with st.expander("查看构建日志 stdout", expanded=True):
                st.code(stdout, language="text")

        if stderr:
            with st.expander("查看错误日志 stderr", expanded=True):
                st.code(stderr, language="text")


def render_runtime_admin():
    """
    Runtime observability panel for enterprise RAG operations.
    """
    if not st.button("Refresh runtime data", use_container_width=True, type="secondary"):
        st.caption("Click to refresh runtime logs, feedback, eval runs, and model config.")
        return

    from app.tools.runtime_admin import get_runtime_admin_status

    status = get_runtime_admin_status(limit=20)
    counts = status.get("counts", {})
    model_runtime = status.get("model_runtime", {})

    st.caption(f"Runtime DB: {status.get('runtime_db_path', '')}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Chat logs", counts.get("chat_logs", 0))
    col2.metric("Feedback", counts.get("feedback", 0))
    col3.metric("Eval runs", counts.get("eval_runs", 0))

    with st.expander("Model runtime", expanded=False):
        st.json(model_runtime)

    tab_chat, tab_feedback, tab_eval = st.tabs(
        ["Chat logs", "Feedback", "Eval runs"]
    )

    with tab_chat:
        chat_logs = status.get("chat_logs", [])
        if not chat_logs:
            st.caption("No chat logs yet.")
        for item in chat_logs:
            title = item.get("question", "")[:80] or item.get("id", "")
            with st.expander(title, expanded=False):
                st.caption(f"id: {item.get('id', '')}")
                st.caption(f"created_at: {item.get('created_at', '')}")
                st.caption(f"latency_ms: {item.get('latency_ms', '')}")
                st.markdown("**Question**")
                st.write(item.get("question", ""))
                st.markdown("**Answer**")
                st.write(item.get("answer", ""))
                st.markdown("**Citations**")
                st.json(item.get("citations", []))

    with tab_feedback:
        feedback_items = status.get("feedback", [])
        if not feedback_items:
            st.caption("No feedback yet.")
        for item in feedback_items:
            title = f"{item.get('rating', '')} | {item.get('question', '')[:60]}"
            with st.expander(title, expanded=False):
                st.caption(f"id: {item.get('id', '')}")
                st.caption(f"created_at: {item.get('created_at', '')}")
                st.caption(f"issue_type: {item.get('issue_type', '')}")
                st.markdown("**Question**")
                st.write(item.get("question", ""))
                st.markdown("**Comment**")
                st.write(item.get("comment", ""))

    with tab_eval:
        eval_runs = status.get("eval_runs", [])
        if not eval_runs:
            st.caption("No eval runs yet.")
        for item in eval_runs:
            title = (
                f"{item.get('eval_type', '')} | "
                f"rate={item.get('hit_rate', 0):.3f} | "
                f"mrr={item.get('mrr', 0):.3f}"
            )
            with st.expander(title, expanded=False):
                st.caption(f"id: {item.get('id', '')}")
                st.caption(f"created_at: {item.get('created_at', '')}")
                st.caption(f"mode: {item.get('mode', '')}")
                st.caption(f"top_k: {item.get('top_k', '')}")
                st.caption(f"avg_latency_ms: {item.get('avg_latency_ms', '')}")
                st.markdown("**Result**")
                st.json(item.get("result", {}))


def render_admin_login():
    """
    管理员入口。
    """
    if not ENABLE_ADMIN_PANEL:
        return

    with st.expander("管理员入口", expanded=False):
        if st.session_state.admin_unlocked:
            st.success("管理员模式已开启")

            if st.button("退出管理员模式", use_container_width=True, type="secondary"):
                st.session_state.admin_unlocked = False
                st.session_state.debug_mode = False
                st.rerun()

            return

        if not ADMIN_PASSWORD:
            st.warning("已开启管理员入口，但 .env 中未配置 ADMIN_PASSWORD。")
            return

        password = st.text_input(
            "管理员密码",
            type="password",
            key="admin_password_input",
        )

        if st.button("进入管理员模式", use_container_width=True, type="secondary"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_unlocked = True
                st.success("管理员验证成功")
                st.rerun()
            else:
                st.error("管理员密码错误")


# =========================
# 5. 左侧边栏
# =========================

with st.sidebar:
    st.markdown(
        """
<div class="sidebar-brand">
    <div class="sidebar-title">政策法规知识库</div>
    <div class="sidebar-caption">面向政策咨询、法规依据、合规风险和类案参考的智能问答工作台。</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button("新建对话", type="primary", use_container_width=True):
        create_new_conversation()
        st.rerun()

    st.markdown("#### 对话")
    active_conversation_id = st.session_state.active_conversation_id

    for conversation in reversed(st.session_state.conversations):
        conversation_id = conversation.get("id", "")
        title = conversation.get("title", "新对话")
        is_active = conversation_id == active_conversation_id
        button_label = f"当前：{title}" if is_active else title

        if st.button(
            button_label,
            key=f"conversation_{conversation_id}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
            disabled=is_active,
        ):
            switch_conversation(conversation_id)
            st.rerun()

    st.write("")
    mode_options = ["Fast Chat", "Agent Chat"]
    if st.session_state.admin_unlocked:
        mode_options.append("Debug")

    if st.session_state.chat_mode not in mode_options:
        st.session_state.chat_mode = "Agent Chat"

    st.session_state.chat_mode = st.selectbox(
        "工作模式",
        options=mode_options,
        index=mode_options.index(st.session_state.chat_mode),
        help="快速问答适合常规咨询；深度分析适合复杂法规、案例和合规问题；Debug 仅管理员可见。",
    )
    st.session_state.fast_mode = st.session_state.chat_mode == "Fast Chat"
    st.session_state.debug_mode = st.session_state.chat_mode == "Debug"

    if st.session_state.chat_mode == "Fast Chat":
        st.caption("当前：快速问答，优先响应速度。")
    elif st.session_state.chat_mode == "Agent Chat":
        st.caption("当前：深度分析，适合复杂政策、法规和案例问题。")
    else:
        st.caption("当前：开发者调试模式。")

    st.write("")
    st.markdown("#### 常用场景")

    quick_qs = [
        "政策办理",
        "法规解释",
        "合规风险",
        "类案参考",
        "居住证材料",
        "个人信息保护",
    ]

    full_qs = {
        "政策办理": "就业补贴怎么申请？需要满足哪些条件？",
        "法规解释": "劳动合同试用期怎么规定？",
        "合规风险": "虚假申请就业补贴有什么风险？",
        "类案参考": "类似诈骗罪法院一般怎么判？",
        "居住证材料": "办理居住证一般需要满足什么条件、准备哪些材料？",
        "个人信息保护": "企业收集员工个人信息需要注意哪些合规要求？",
    }

    for short_q in quick_qs:
        if st.button(short_q, use_container_width=True, type="secondary"):
            st.session_state.quick_question = full_qs[short_q]

    st.write("")

    with st.expander("已识别上下文", expanded=False):
        slot_context = build_context_from_slots(st.session_state.user_slots)

        if slot_context:
            st.markdown(
                f"<div class='slot-box'>{slot_context.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="slot-box">
当前还没有识别到明确的个人信息。<br>
你可以补充：地区、主体身份、业务场景、争议事实、企业行业、办理事项或合规主题等。
</div>
""",
                unsafe_allow_html=True,
            )

        if st.session_state.slot_summary:
            st.caption(f"最近识别：{st.session_state.slot_summary}")

    with st.expander("当前系统能力", expanded=False):
        st.markdown(
            """
<div class="capability-card">
 政策文件、法律法规、类案参考三类知识源已启用<br>
 公告通知源已预留，当前 Demo 暂不启用<br>
 支持政策办理、法规解释、风险合规、类案参考<br>
 支持参考依据展开、问答日志、反馈和评测接口
</div>
""",
            unsafe_allow_html=True,
        )

    render_admin_login()

    if st.session_state.admin_unlocked:
        with st.expander("开发者选项", expanded=False):
            st.caption("开发者调试由侧边栏“工作模式 -> Debug”统一控制。")
            if st.session_state.chat_mode == "Debug":
                st.info("当前为 Debug 模式，将运行完整链路并展示中间结果。")
            else:
                st.success(f"当前为 {st.session_state.chat_mode} 模式。")

        with st.expander("知识库管理", expanded=False):
            render_policy_library_admin()

        with st.expander("运行状态", expanded=False):
            render_runtime_admin()

    with st.expander("使用说明", expanded=False):
        st.markdown(
            """
<div class="small-note">
本系统基于已接入的真实公开政策、法律法规和法律案例生成回答。<br><br>
回答仅作为政策理解、法律依据检索和合规分析参考；具体办事条件、执法口径和司法判断，请以主管部门、正式法律文书和专业法律意见为准。
</div>
""",
            unsafe_allow_html=True,
        )


# =========================
# 6. 右侧主区域
# =========================

if st.session_state.admin_unlocked and st.session_state.debug_mode:
    mode_text = "开发者模式"
else:
    mode_text = "普通模式"

render_workspace_header(mode_text)
render_question_showcase()


# =========================
# 7. 渲染历史对话
# =========================

messages_to_render = st.session_state.messages[-MAX_RENDERED_MESSAGES:]
hidden_message_count = max(0, len(st.session_state.messages) - len(messages_to_render))

if hidden_message_count:
    st.caption(f"已隐藏较早的 {hidden_message_count} 条历史消息，以提升页面渲染速度。")

for msg in messages_to_render:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        render_business_flow_card(msg.get("business_flow", {}))
        render_source_router_card(msg.get("source_router_result", {}))
        render_sources(msg.get("sources", []))

        if msg.get("debug_result"):
            render_agent_debug(msg["debug_result"])


# =========================
# 8. 输入与执行
# =========================

user_input = st.chat_input("请输入政策、法规、合规或案例问题...")

if st.session_state.quick_question:
    user_input = st.session_state.quick_question
    st.session_state.quick_question = None

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    sync_active_conversation()

    with st.chat_message("assistant"):
        actual_debug_mode = st.session_state.admin_unlocked and st.session_state.debug_mode
        started_at = time.perf_counter()
        business_flow = get_current_business_flow(user_input)

        spinner_text = "正在识别..."

        with st.spinner(spinner_text):
            update_slots_with_lightweight_rules(user_input)

            if not st.session_state.fast_mode:
                from app.agents.slot_agent import slot_agent

                slot_result = slot_agent(
                    question=user_input,
                    existing_slots=st.session_state.user_slots,
                )

                st.session_state.user_slots = slot_result.get(
                    "slots",
                    st.session_state.user_slots,
                )

                st.session_state.slot_summary = slot_result.get("summary", "")
            else:
                if not st.session_state.slot_summary:
                    st.session_state.slot_summary = "快速问答模式已跳过本轮槽位识别。"

            enhanced_question = build_enhanced_question(user_input)

            if st.session_state.fast_mode and not actual_debug_mode:
                result = run_fast_streaming_answer(enhanced_question)
            else:
                from app.graph.gov_graph import run_gov_graph

                result = run_gov_graph(
                    enhanced_question,
                    debug_mode=actual_debug_mode,
                )

            if result.get("direct_answer"):
                if not st.session_state.pending_question:
                    st.session_state.pending_question = user_input
            else:
                st.session_state.pending_question = ""

        final_answer = result.get(
            "final_answer",
            "抱歉，当前系统暂时无法生成回答，请稍后再试。",
        )
        latency_ms = (time.perf_counter() - started_at) * 1000

        disclaimer = (
            "\n\n<div class='gov-tips'>"
            "提示：本系统回答仅供政策理解、法规检索和合规分析参考。"
            "具体办事条件、监管口径和司法判断，请以主管部门、正式法律文书和专业意见为准。"
            "</div>"
        )

        answer_to_display = final_answer + disclaimer

        if result.get("streaming_mode") == "fast_policy_answer":
            st.markdown(disclaimer, unsafe_allow_html=True)
        else:
            stream_markdown(answer_to_display)

        render_business_flow_card(business_flow)
        render_source_router_card(result.get("source_router_result", {}))
        render_sources(result.get("sources", []))
        render_agent_debug(result)

        try:
            saved_log = save_streamlit_chat_log(
                question=enhanced_question,
                answer=final_answer,
                result=result,
                latency_ms=latency_ms,
                chat_mode=st.session_state.chat_mode,
                debug_mode=actual_debug_mode,
                business_flow=business_flow,
            )
            if st.session_state.admin_unlocked:
                st.caption(f"Chat log saved: {saved_log.get('log_id', '')}")
        except Exception as exc:
            if st.session_state.admin_unlocked:
                st.warning(f"Chat log save failed: {exc}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_to_display,
                "business_flow": business_flow,
                "source_router_result": result.get("source_router_result", {}),
                "sources": result.get("sources", []),
                "debug_result": result if actual_debug_mode else None,
            }
        )
        sync_active_conversation()
