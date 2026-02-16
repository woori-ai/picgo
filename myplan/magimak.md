# Antigravity AI 에이전트 팀 구축 가이드 (Final)

**목표**: Antigravity AI 환경(로컬 서버)에서 Ollama를 엔진으로 사용하는 '나만의 AI 에이전트 팀'을 구축하고 실행하는 전체 과정을 순차적으로 정리합니다.

## 1. 사전 준비 (Prerequisites)

먼저 `Ollama`가 설치되어 있고 실행 중이어야 합니다.
Antigravity AI 터미널(PowerShell)을 열고 진행합니다.

### 1-1. 필수 모델 다운로드 (Models Pull)
우리가 정한 **최적의 팀 구성**을 위해 필요한 모델들을 다운로드합니다.

```powershell
# 1. PM & Senior Dev (EXAONE 3.5 - 7.8B)
ollama pull exaone3.5

# 2. Code Reviewer (Llama 3 - 8B)
ollama pull llama3

# 3. Writer/Tester (Phi-3 - 3.8B)
ollama pull phi3
```

---

## 2. 개발 환경 설정 (Python Environment)

에이전트 팀을 조율할 **오케스트레이터(Orchestrator)** 도구로 **CrewAI** 라이브러리를 사용합니다. (가장 직관적이고 역할 부여가 쉽습니다.)

### 2-1. 라이브러리 설치
```powershell
pip install crewai crewai-tools langchain_community
```

---

## 3. 에이전트 팀 코드 작성 (Python)

이제 실제로 일하는 팀을 만드는 Python 코드를 작성합니다. 파일명은 `agent_team.py`로 저장한다고 가정합니다.

```python
import os
from crewai import Agent, Task, Crew, Process
from crewai import LLM

# ---------------------------------------------------------
# 1. 모델(두뇌) 연결 설정
# - CrewAI 최신 버전에서는 LLM 클래스를 사용하여 Ollama를 직접 지정합니다.
# - 형식: "ollama/모델명"
# ---------------------------------------------------------

# PM & Senior Dev용 (EXAONE 3.5)
exaone = LLM(model="ollama/exaone3.5", base_url="http://localhost:11434")

# Code Reviewer용 (Llama 3)
llama3 = LLM(model="ollama/llama3", base_url="http://localhost:11434")

# Tester용 (Phi-3)
phi3 = LLM(model="ollama/phi3", base_url="http://localhost:11434")


# ---------------------------------------------------------
# 2. 에이전트(직원) 채용 및 역할 부여
# ---------------------------------------------------------

# [PM]: 프로젝트 매니저 (한국어 소통 & 기획)
project_manager = Agent(
    role='Project Manager',
    goal='프로젝트 요구사항을 명확히 정의하고 개발 방향을 지시',
    backstory='당신은 경험이 풍부한 PM입니다. 한국어로 팀원들과 소통하며, 사용자의 모호한 요구사항을 구체적인 개발 스펙으로 변환합니다.',
    llm=exaone,  # EXAONE 사용
    verbose=True
)

# [Senior Dev]: 수석 개발자 (구현 & 로직)
senior_developer = Agent(
    role='Senior Python Developer',
    goal='PM의 기획서에 따라 고품질의 Python 코드를 작성',
    backstory='당신은 코딩 실력이 뛰어난 수석 개발자입니다. 복잡한 로직도 수학적 사고로 풀어내며, 간결하고 효율적인 코드를 작성합니다.',
    llm=exaone,  # EXAONE 사용 (코딩/추론)
    verbose=True
)

# [Reviewer]: 코드 리뷰어 (검증)
code_reviewer = Agent(
    role='Code Reviewer',
    goal='작성된 코드의 버그를 찾고 개선점을 제안',
    backstory='당신은 꼼꼼한 코드 리뷰어입니다. 개발자가 놓친 버그나 보안 취약점, 비효율적인 로직을 찾아내어 직설적으로 지적합니다.',
    llm=llama3,  # Llama 3 사용 (교차 검증)
    verbose=True
)

# [Tester/Writer]: 테스터 (문서화 & 테스트)
tester = Agent(
    role='Quality Assurance Engineer',
    goal='작성된 코드에 대한 테스트 케이스를 만들고 사용 설명서를 작성',
    backstory='당신은 QA 엔지니어입니다. 코드가 잘 작동하는지 확인하기 위한 테스트 시나리오를 만들고, 읽기 쉬운 문서를 작성합니다.',
    llm=phi3,    # Phi-3 사용 (가벼운 작업)
    verbose=True
)


# ---------------------------------------------------------
# 3. 작업(Task) 정의
# ---------------------------------------------------------
# 사용자가 터미널에서 직접 주제를 입력하도록 변경
print("## 만들고 싶은 프로그램이나 주제를 입력하세요 ##")
user_input = input("입력: ")
user_topic = f"Python으로 만드는 {user_input}"
task1_plan = Task(
    description='사용자가 요청한 "Python으로 뱀 게임 만들기"에 대한 기능 명세서와 개발 계획을 한국어로 작성하세요.',
    expected_output='기능 목록과 단계별 개발 계획이 담긴 마크다운 문서',
    agent=project_manager
)

task2_code = Task(
    description='PM의 계획서를 바탕으로 완벽하게 동작하는 뱀 게임(Snake Game) Python 코드를 작성하세요.',
    expected_output='실행 가능한 Python 소스 코드 파일 (.py)',
    agent=senior_developer
)

task3_review = Task(
    description='작성된 코드를 분석하여 버그나 개선할 점을 찾으세요. 만약 심각한 문제가 있다면 수정 코드를 제안하세요.',
    expected_output='코드 리뷰 보고서 및 수정 제안',
    agent=code_reviewer
)

task4_doc = Task(
    description='최종 코드에 대한 실행 방법(README)과 테스트 케이스를 간단히 작성하세요.',
    expected_output='README.md 내용 및 테스트 케이스 목록',
    agent=tester
)


# ---------------------------------------------------------
# 4. 팀 결성 및 프로젝트 시작 (Kick-off)
# ---------------------------------------------------------
dev_team = Crew(
    agents=[project_manager, senior_developer, code_reviewer, tester],
    tasks=[task1_plan, task2_code, task3_review, task4_doc],
    process=Process.sequential,  # 순차적으로 작업 (PM -> Dev -> Review -> QA)
    verbose=True
)

print("### Antigravity Agent Team 프로젝트 시작 ###")
result = dev_team.kickoff()

print("\n\n################################################")
print("## 최종 결과물 ##")
print(result)
```

---

## 4. 실행 및 결과 확인 (Execution)

1.  위 코드를 `agent_team.py`로 저장합니다.
2.  터미널에서 실행합니다.

```powershell
python agent_team.py
```

### 🎯 실행 방법 (가상환경 + 파일 기반 대화)

이제 **터미널 입력 대신 파일(`team_talk.txt`)**을 통해 에이전트 팀과 소통합니다.

1.  **질문/주제 작성**:
    - `team_talk.txt` 파일을 열고, 맨 아래에 원하시는 주제(예: "테트리스 게임 만들기")를 적고 저장하세요.
    - (파일이 없으면 실행 시 자동으로 생성됩니다.)

2.  **에이전트 팀 실행**:
    ```powershell
    # 가상환경의 Python으로 실행
    .\venv\Scripts\python agent_team.py
    ```

3.  **결과 확인**:
    - 실행이 끝나면, 결과가 **`team_talk.txt` 파일 맨 아래에 자동으로 추가(Append)**됩니다.
    - 이 파일 하나로 질문과 답변 기록을 모두 관리할 수 있습니다.

### 🎯 실행 과정 (예상)
1.  **PM (EXAONE)**이 `team_talk.txt`에 적힌 주제를 읽고 기획서를 작성합니다.
2.  **Dev (EXAONE)**이 기획서를 읽고 Python 코드를 짭니다.
3.  **Reviewer (Llama 3)**가 코드를 보고 비평합니다.
4.  **Tester (Phi-3)**가 사용 설명서를 작성합니다.
5.  최종 결과가 `team_talk.txt`에 저장됩니다. (대화 기록처럼 누적됩니다.)

이것으로 Antigravity AI 환경에서 **나만의 AI 팀과 파일로 소통하는 시스템**이 완성됩니다.
