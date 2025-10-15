import re, os, streamlit as st
from openai import OpenAI

# ---------- 기본 설정 ----------
st.set_page_config(page_title="GPT 연동 실험 챗봇", page_icon="💬", layout="wide")
st.title("💬 GPT 연동 실험 챗봇")

# ---------- 사이드바 ----------
with st.sidebar:
    st.subheader("⚙️ Settings")
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
    base_url = st.text_input("BASE_URL(선택)", value=os.getenv("OPENAI_BASE_URL",""))
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL","gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)  # 실험 요구: 결정적 출력
    typecode_qp = st.text_input("TypeCode(선택, 1~8)", value=st.query_params.get("type", ""))  # ?type=5 같이 링크로 넘길 수 있음
    clear = st.button("대화 초기화")

# ---------- OpenAI 클라이언트 ----------
if not api_key:
    st.info("좌측에 OPENAI_API_KEY를 입력하세요.")
    st.stop()

client_kwargs = {"api_key": api_key}
if base_url.strip():
    client_kwargs["base_url"] = base_url.strip()
client = OpenAI(**client_kwargs)

# ---------- 고정 System / Assistant Seed ----------
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

[Introduction]
- Use (GenderCode × ColleagueType) to decide 이름/역할.
- Use selected Tone for self-introduction:

  * 친근형(Tone=2):
    - 인간: "안녕 {사용자이름}! 반가워. 나는 {사용자이름} 널 도와줄 친구 {민준/서연}이야."
    - AI:   "안녕 {사용자이름}! 반가워. 나는 {사용자이름} 널 도와줄 AI 비서 {James/Julia}야."

  * 공식형(Tone=1):
    - 인간: "만나서 반갑습니다. 저는 {사용자이름} 님을 도와드릴 동료 {민준/서연}입니다."
    - AI:   "만나서 반갑습니다. 저는 {사용자이름} 님을 도와드릴 AI 비서 {James/Julia}입니다."

- Then show **과제1만 제시** in same tone:

  * 친근형:
    "과제1: 다음 태양계 행성들을 크기(직경)가 큰 순서대로 나열해 줘.
     보기: 수성, 금성, 지구, 화성, 목성, 토성, 천왕성, 해왕성
     모르는 건 나한테 물어봐.
     모든 질문이 끝나면 아래 형식으로 정답을 입력해 줘.
     정답: 행성1 행성2 행성3 행성4 행성5 행성6 행성7 행성8"

  * 공식형:
    "과제1: 다음 태양계 행성들을 크기(직경)가 큰 순서대로 나열해 주십시오.
     보기: 수성, 금성, 지구, 화성, 목성, 토성, 천왕성, 해왕성
     필요한 정보가 있으면 저에게 질문해 주시기 바랍니다.
     모든 질문이 끝나면 아래 형식으로 정답을 입력해 주십시오.
     정답: 행성1 행성2 행성3 행성4 행성5 행성6 행성7 행성8"

[Answer Handling]
- If input starts with "정답:" and lists 8 planets →
  * 공식형: "답안을 제출하셨습니다. 연구자가 확인할 예정입니다. 이어서 다음 과제를 드리겠습니다."
  * 친근형: "답안 잘 제출했어. 연구자가 확인할 거야. 이제 다음 과제를 줄게."
  → Then present 과제2:

  * 친근형:
    "과제2: 지구 말고 다른 행성 중에서 생명체가 살 수 있을 것 같은 곳을 하나 고르고, 그렇게 생각한 이유를 자유롭게 말해줘.
     답변: 자유 서술"

  * 공식형:
    "과제2: 지구를 제외했을 때, 태양계 행성 중에서 생명체가 존재할 가능성이 가장 높다고 생각하는 행성을 고르고, 그렇게 판단한 근거를 자유롭게 설명해 주십시오.
     답변: 자유 서술"

- If input starts with "답변:" (자유 서술) →
  * 공식형: "답안을 제출하셨습니다. 연구자가 확인할 예정입니다."
  * 친근형: "답안 잘 제출했어. 연구자가 확인할 거야."

- Otherwise → treat as question, follow Work Style + Tone.

[Work Style Guidelines]
- 꼼꼼형: 길고 정교한 설명(맥락·근거 제시)
- 신속형: 짧고 핵심만 전달

[Tone Rules]
- 친근형: 반말 only, 사용자이름 1회 언급, 짧은 격려 1회
- 공식형: 존댓말 only, 이름 재언급 없음, 정중·중립

[Consistency]
- Always follow TypeCode mapping (1~4=인간, 5~8=AI) and existing mismatch rules.
- Same input → same output. No randomness.
"""

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

# ---------- 세션 초기화 ----------
def init_session():
    st.session_state.messages = []
    st.session_state.got_first_input = False  # 첫 입력 검증용
    # System + (선택) TypeCode 고지 + Assistant seed
    st.session_state.messages.append({"role":"system","content":SYS_PROMPT})
    if typecode_qp.strip():
        st.session_state.messages.append({"role":"system","content":f"TypeCode={typecode_qp.strip()}"})
    st.session_state.messages.append({"role":"assistant","content":ASST_SEED})

if ("messages" not in st.session_state) or clear:
    init_session()

# ---------- 유효성 검사(첫 입력 전용) ----------
FIRST_INPUT_RE = re.compile(r"^\s*([^,]+)\s*,\s*([12])\s*,\s*([12])\s*,\s*([12])\s*$")

def validate_first_input(text: str) -> bool:
    return bool(FIRST_INPUT_RE.match(text or ""))

# ---------- OpenAI 호출(스트리밍) ----------
def stream_chat(messages, model, temperature):
    with client.chat.completions.stream(model=model, messages=messages, temperature=temperature) as stream:
        full = ""
        for event in stream:
            if event.type == "token":
                full += event.token
                yield event.token
            elif event.type == "completed":
                break
        return full

# ---------- 렌더 ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- 입력 처리 ----------
placeholder = "이름, 성별번호, 업무번호, 어조번호 형식으로 입력해주세요 (예: 이민용, 1, 1, 2)"
if user_text := st.chat_input(placeholder):
    # 1) 첫 입력 검증
    if not st.session_state.got_first_input:
        if not validate_first_input(user_text):
            with st.chat_message("assistant"):
                st.markdown("입력 형식이 올바르지 않습니다")
            # (시드 유지, 대화 지속)
        else:
            st.session_state.got_first_input = True
            st.session_state.messages.append({"role":"user","content":user_text})
            with st.chat_message("user"):
                st.markdown(user_text)
            # GPT로 규칙에 맞는 소개/과제1 제시를 생성
            with st.chat_message("assistant"):
                holder = st.empty()
                chunks = stream_chat(st.session_state.messages, model, temperature)
                if chunks:
                    text = ""
                    for c in chunks:
                        text += c
                        holder.markdown(text)
                    st.session_state.messages.append({"role":"assistant","content":text})
    else:
        # 2) 이후 입력: 정답/답변 처리 메시지도 GPT가 규칙대로 생성(행성 크기/생명 가능성 질문 포함 → GPT 사용)
        st.session_state.messages.append({"role":"user","content":user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            holder = st.empty()
            chunks = stream_chat(st.session_state.messages, model, temperature)
            if chunks:
                text = ""
                for c in chunks:
                    text += c
                    holder.markdown(text)
                st.session_state.messages.append({"role":"assistant","content":text})
