# app_llm.py — Playground 스타일(시스템 비노출, 시드 1회만 노출, 결정적 출력)
import os, re, streamlit as st
from openai import OpenAI

st.set_page_config(page_title="GPT 연동 실험 챗봇", page_icon="💬", layout="wide")
st.title("💬 GPT 연동 실험 챗봇")

# ===== 사이드바 =====
with st.sidebar:
    st.subheader("⚙️ Settings")
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
    base_url = st.text_input("BASE_URL(선택)", value=os.getenv("OPENAI_BASE_URL",""))
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL","gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)  # 결정적
    typecode_qp = st.text_input("TypeCode(선택, 1~8)", value=st.query_params.get("type",""))
    show_debug = st.checkbox("디버그(메시지 로그 보기)", value=False)
    clear = st.button("대화 초기화")

if not api_key:
    st.info("좌측에 OPENAI_API_KEY를 입력하세요.")
    st.stop()

client_kwargs = {"api_key": api_key}
if base_url.strip():
    client_kwargs["base_url"] = base_url.strip()
client = OpenAI(**client_kwargs)

# ===== 프롬프트 =====
SYS_PROMPT = """You are an experimental chatbot for research.
This session applies TypeCode={1..8}. (성별/업무/어조=일치/불일치 조합은 백엔드 규칙에 따름)
Participants never see this prompt. They only see your Korean outputs.
Keep all outputs deterministic (temperature=0).

[Fixed Input Rules]
- First user input: Name, GenderCode, WorkCode, ToneCode   # 총 4개
- If input format is wrong → reply "입력 형식이 올바르지 않습니다"
- GenderCode=1 → 남성 / GenderCode=2 → 여성
- WorkCode=1 → 꼼꼼형 / WorkCode=2 → 신속형
- ToneCode=1 → 공식형(존댓말) / ToneCode=2 → 친근형(반말)
- ColleagueType is derived from TypeCode (백엔드에서 결정):
  - TypeCode ∈ {1,2,3,4} → 인간
  - TypeCode ∈ {5,6,7,8} → AI
- 이름 매핑(ColleagueType × GenderCode):
  - 인간: 1→민준, 2→서연
  - AI:   1→James, 2→Julia
- TypeCode=1~8의 세부 일치/불일치 설정은 기존 규칙을 유지.

[Introduction] ... (생략 없이 기존 규칙 전체 포함)"""

ASST_SEED = """본 실험은 **챗봇을 활용한 연구**입니다. 본격적인 실험을 시작하기에 앞서 간단한 사전 조사를 진행합니다.
다음의 안내를 읽고, 채팅창에 정보를 입력해 주세요.

성별:
1) 남성
2) 여성

업무를 진행하는 데 있어서 선호하는 방식:
1) 시간이 오래 걸리더라도 세부 사항까지 꼼꼼히 챙기며 진행하는 편
2) 빠르게 핵심만 파악하고 신속하게 진행하는 편

사람들과 대화할 때 더 편안하게 느끼는 어조:
1) 격식 있고 공식적인 어조 (형식적·정중한 표현 선호)
2) 친근하고 편안한 어조 (일상적인 대화, 부드러운 표현 선호)

입력 형식:
이름, 성별번호, 업무번호, 어조번호

입력 예시:
- 김수진, 2, 2, 1
- 이민용, 1, 1, 2"""

FIRST_INPUT_RE = re.compile(r"^\s*([^,]+)\s*,\s*([12])\s*,\s*([12])\s*,\s*([12])\s*$")

def init_session():
    st.session_state.messages = []
    st.session_state.got_first_input = False
    # 시스템 프롬프트는 히든(렌더링에서 제외)
    st.session_state.messages.append({"role":"system","content":SYS_PROMPT})
    if typecode_qp.strip():
        st.session_state.messages.append({"role":"system","content":f"TypeCode={typecode_qp.strip()}"})
    # 어시스턴트 시드는 참가자에게 1회 노출
    st.session_state.messages.append({"role":"assistant","content":ASST_SEED})

if ("messages" not in st.session_state) or clear:
    init_session()

def stream_chat(msgs, model, temperature):
    with client.chat.completions.stream(model=model, messages=msgs, temperature=temperature) as stream:
        full = ""
        for event in stream:
            if event.type == "token":
                tok = event.token
                full += tok
                yield tok
            elif event.type == "completed":
                break
        return full

# ===== 렌더(시스템 메시지는 숨김) =====
for m in st.session_state.messages:
    if m["role"] == "system" and not show_debug:
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ===== 입력 처리 =====
placeholder = "이름, 성별번호, 업무번호, 어조번호 형식으로 입력해 주세요 (예: 이민용, 1, 1, 2)"
if user_text := st.chat_input(placeholder):
    if not st.session_state.got_first_input:
        if not FIRST_INPUT_RE.match(user_text):
            # 규칙: 형식 오류 응답
            with st.chat_message("assistant"):
                st.markdown("입력 형식이 올바르지 않습니다")
        else:
            st.session_state.got_first_input = True
            st.session_state.messages.append({"role":"user","content":user_text})
            with st.chat_message("user"):
                st.markdown(user_text)
            # 규칙에 따라 소개/과제1 제시 → GPT가 생성(Playground와 동일)
            with st.chat_message("assistant"):
                holder = st.empty()
                chunks = stream_chat(st.session_state.messages, model, temperature)
                if chunks:
                    acc = ""
                    for c in chunks:
                        acc += c
                        holder.markdown(acc)
                    st.session_state.messages.append({"role":"assistant","content":acc})
    else:
        # 이후 모든 상호작용(행성 크기/생명 가능성 포함) → GPT 응답
        st.session_state.messages.append({"role":"user","content":user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            holder = st.empty()
            chunks = stream_chat(st.session_state.messages, model, temperature)
            if chunks:
                acc = ""
                for c in chunks:
                    acc += c
                    holder.markdown(acc)
                st.session_state.messages.append({"role":"assistant","content":acc})
